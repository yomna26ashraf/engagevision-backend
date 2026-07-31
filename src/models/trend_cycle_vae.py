"""
TrendCycleVAE — Section III-D of the paper.

Pipeline per branch (trend / cycle), matching the paper's description:
  1. 1-D convolutional stack with residual connections (stabilizes training,
     mitigates vanishing gradients).
  2. Transformer encoder over the convolved sequence.
  3. Two parallel linear heads compute (mu, log-variance) of the latent
     Gaussian from the (pooled) encoder output.
  4. Reparameterization trick samples the latent vector.
  5. The latent vector is projected back to the temporal domain and refined
     by a Transformer decoder that attends to the encoder output, to
     reconstruct the original (trend or cycle) sequence.

The paper explicitly calls this a "dual-branch VAE": trend and cycle each
get their own encode/decode pathway (separate weights), which is what
`TrendCycleVAE` below instantiates via two `_SingleComponentVAE`s.

Output A (Eq. 12) = [mu_trend, logvar_trend, mu_cycle, logvar_cycle],
later mapped to a continuous engagement score S by an external FC/MLP
head (see models/mlatte.py), following Eq. 13.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from .fft_decomposition import fft_trend_cycle_decompose


class ConvResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1), :]


class _SingleComponentVAE(nn.Module):
    """One branch (trend OR cycle) of the dual-branch VAE."""

    def __init__(self, input_dim: int, d_model: int = 256, n_heads: int = 8,
                 num_layers: int = 4, latent_dim: int = 128,
                 conv_channels=(128, 256, 256), dropout: float = 0.1):
        super().__init__()
        channels = [input_dim] + list(conv_channels)
        conv_blocks = []
        for i in range(len(channels) - 1):
            conv_blocks.append(ConvResidualBlock1D(channels[i], channels[i + 1]))
        # final projection to d_model if last conv channel differs
        if channels[-1] != d_model:
            conv_blocks.append(nn.Conv1d(channels[-1], d_model, kernel_size=1))
        self.conv_stack = nn.Sequential(*conv_blocks)

        self.pos_enc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

        self.latent_to_seq = nn.Linear(latent_dim, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

        self.d_model = d_model
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor):
        # x: (B, T, input_dim)
        h = x.transpose(1, 2)          # (B, input_dim, T)
        h = self.conv_stack(h)         # (B, d_model, T)
        h = h.transpose(1, 2)          # (B, T, d_model)
        h = self.pos_enc(h)
        enc_out = self.encoder(h)      # (B, T, d_model)
        pooled = enc_out.mean(dim=1)   # (B, d_model) — window-level summary
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        return mu, logvar, enc_out

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, enc_out: torch.Tensor, seq_len: int) -> torch.Tensor:
        z_seq = self.latent_to_seq(z).unsqueeze(1).repeat(1, seq_len, 1)  # (B, T, d_model)
        z_seq = self.pos_enc(z_seq)
        dec_out = self.decoder(tgt=z_seq, memory=enc_out)
        return self.output_proj(dec_out)  # (B, T, input_dim)

    def forward(self, x: torch.Tensor):
        mu, logvar, enc_out = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, enc_out, seq_len=x.shape[1])
        return recon, mu, logvar, z


class TrendCycleVAE(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 256, n_heads: int = 8,
                 num_layers: int = 4, latent_dim: int = 128,
                 conv_channels=(128, 256, 256),
                 fft_trend_cutoff_ratio: float = 0.1, fft_num_peaks: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.fft_trend_cutoff_ratio = fft_trend_cutoff_ratio
        self.fft_num_peaks = fft_num_peaks

        self.trend_vae = _SingleComponentVAE(
            input_dim, d_model, n_heads, num_layers, latent_dim, conv_channels, dropout
        )
        self.cycle_vae = _SingleComponentVAE(
            input_dim, d_model, n_heads, num_layers, latent_dim, conv_channels, dropout
        )
        self.latent_dim = latent_dim
        # A = [mu_trend, logvar_trend, mu_cycle, logvar_cycle]
        self.output_dim = latent_dim * 4

    def forward(self, F_fusion: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        F_fusion: (B, T, D) windowed fused multimodal features.
        """
        trend, cycle = fft_trend_cycle_decompose(
            F_fusion, trend_cutoff_ratio=self.fft_trend_cutoff_ratio,
            num_peaks=self.fft_num_peaks,
        )

        recon_trend, mu_t, logvar_t, z_t = self.trend_vae(trend)
        recon_cycle, mu_c, logvar_c, z_c = self.cycle_vae(cycle)

        A = torch.cat([mu_t, logvar_t, mu_c, logvar_c], dim=-1)  # (B, 4*latent_dim) — Eq. 12
        recon_fusion = recon_trend + recon_cycle                  # reconstruction of F_fusion

        return {
            "A": A,
            "recon": recon_fusion,
            "target": F_fusion,
            "trend_target": trend,
            "cycle_target": cycle,
            "recon_trend": recon_trend,
            "recon_cycle": recon_cycle,
            "mu_trend": mu_t, "logvar_trend": logvar_t,
            "mu_cycle": mu_c, "logvar_cycle": logvar_c,
        }
