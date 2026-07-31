"""
Full M-LATTE assembly (Fig. 2 / Eqs. 1-5).

Two variants, both sharing TrendCycleVAE:

  MLATTEFull        — visual + audio + text, with ViCEF fusion. Used for
                       RoomReader / CMU-MOSI where all three modalities
                       are available. Outputs a continuous score S in
                       roughly [-2, 2] (Eq. 5), via regression.

  MLATTEVisualOnly   — visual modality only, ViCEF omitted (Section
                       IV-C-2, used for DAiSEE / EngageNet). We keep the
                       regression head (continuous score) as the primary
                       output — matching the paper's stated data
                       processing of mapping DAiSEE labels to {0, 0.25,
                       0.5, 1.0} and training with MSE — and additionally
                       expose `score_to_class()` to bucket predictions
                       into the four DAiSEE levels for accuracy reporting
                       (Table VI), since the paper reports both a
                       regression-style training setup and a classification
                       accuracy number for this dataset.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .trend_cycle_vae import TrendCycleVAE
from .vicef import ViCEF


class RegressionHead(nn.Module):
    """Eq. 13: S = FC(mu_trend, logvar_trend, mu_cycle, logvar_cycle, x_t).

    `x_t` (the raw fused feature at the current step) is optionally
    concatenated in; we default to using it since the paper's Eq. 13
    includes it explicitly.
    """

    def __init__(self, vae_output_dim: int, fusion_dim: Optional[int] = None,
                 hidden_dim: int = 256, output_range: Optional[tuple] = None):
        super().__init__()
        in_dim = vae_output_dim + (fusion_dim or 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.output_range = output_range  # e.g. (-2, 2); None = unconstrained

    def forward(self, A: torch.Tensor, x_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = torch.cat([A, x_t], dim=-1) if x_t is not None else A
        out = self.mlp(feat).squeeze(-1)  # (B,)
        if self.output_range is not None:
            lo, hi = self.output_range
            out = lo + (hi - lo) * torch.sigmoid(out)
        return out


class MLATTEFull(nn.Module):
    """Trimodal pipeline: F_V, F_A, F_T -> ViCEF -> TrendCycleVAE -> score."""

    def __init__(self, visual_dim: int, audio_dim: int, text_dim: int,
                 vicef_d_model: int = 256, vicef_heads: int = 8,
                 vae_d_model: int = 256, vae_heads: int = 8, vae_layers: int = 4,
                 vae_latent_dim: int = 128, vae_conv_channels=(128, 256, 256),
                 fft_trend_cutoff_ratio: float = 0.1, fft_num_peaks: int = 3,
                 output_range: Optional[tuple] = (-2.0, 2.0)):
        super().__init__()
        self.vicef = ViCEF(visual_dim, audio_dim, text_dim, vicef_d_model, vicef_heads)
        self.trend_cycle_vae = TrendCycleVAE(
            input_dim=self.vicef.output_dim, d_model=vae_d_model, n_heads=vae_heads,
            num_layers=vae_layers, latent_dim=vae_latent_dim,
            conv_channels=vae_conv_channels,
            fft_trend_cutoff_ratio=fft_trend_cutoff_ratio, fft_num_peaks=fft_num_peaks,
        )
        self.head = RegressionHead(
            vae_output_dim=self.trend_cycle_vae.output_dim,
            fusion_dim=self.vicef.output_dim,
            output_range=output_range,
        )

    def forward(self, F_V: torch.Tensor, F_A: torch.Tensor, F_T: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        F_V: (B, T, visual_dim), F_A: (B, T, audio_dim), F_T: (B, T, text_dim)
        Returns dict with "score" (B,) and the raw vae_out for loss computation.
        """
        f_fusion = self.vicef(F_V, F_A, F_T)          # (B, T, d_model)  Eq. 3
        vae_out = self.trend_cycle_vae(f_fusion)        # Eq. 4
        x_t = f_fusion[:, -1, :]                          # current-step fused feature
        score = self.head(vae_out["A"], x_t)             # Eq. 5 / Eq. 13
        vae_out["score"] = score
        return vae_out


class MLATTEVisualOnly(nn.Module):
    """Visual-only pipeline for DAiSEE / EngageNet (ViCEF omitted, Section IV-C-2)."""

    def __init__(self, visual_dim: int,
                 vae_d_model: int = 256, vae_heads: int = 8, vae_layers: int = 4,
                 vae_latent_dim: int = 128, vae_conv_channels=(128, 256, 256),
                 fft_trend_cutoff_ratio: float = 0.1, fft_num_peaks: int = 3,
                 class_values=(0.0, 0.25, 0.5, 1.0)):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, vae_d_model)
        self.trend_cycle_vae = TrendCycleVAE(
            input_dim=vae_d_model, d_model=vae_d_model, n_heads=vae_heads,
            num_layers=vae_layers, latent_dim=vae_latent_dim,
            conv_channels=vae_conv_channels,
            fft_trend_cutoff_ratio=fft_trend_cutoff_ratio, fft_num_peaks=fft_num_peaks,
        )
        self.head = RegressionHead(
            vae_output_dim=self.trend_cycle_vae.output_dim,
            fusion_dim=vae_d_model,
            output_range=(0.0, 1.0),
        )
        self.register_buffer(
            "class_values", torch.tensor(class_values, dtype=torch.float32), persistent=False
        )

    def forward(self, F_V: torch.Tensor) -> Dict[str, torch.Tensor]:
        """F_V: (B, T, visual_dim). Returns dict with "score" (B,) in [0,1]."""
        v = self.visual_proj(F_V)               # (B, T, d_model)
        vae_out = self.trend_cycle_vae(v)
        x_t = v[:, -1, :]
        score = self.head(vae_out["A"], x_t)
        vae_out["score"] = score
        return vae_out

    def score_to_class(self, score: torch.Tensor) -> torch.Tensor:
        """Bucket a continuous score into the nearest DAiSEE class value,
        for accuracy reporting (Table VI). Works regardless of which
        device `score` is on (e.g. CPU tensors accumulated for metrics,
        even though the model itself lives on GPU)."""
        class_values = self.class_values.to(score.device)
        diffs = (score.unsqueeze(-1) - class_values.unsqueeze(0)).abs()
        idx = diffs.argmin(dim=-1)
        return class_values[idx]
