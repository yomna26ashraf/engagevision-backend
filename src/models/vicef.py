"""
Vision-guided Cross-modal Engagement Fusion (ViCEF) — Section III-C.

Visual features act as the Query; audio and text act as Key/Value in two
independent cross-attention branches. Results are summed with the visual
residual (Eqs. 6-10 in the paper):

    Q_{v->a} = W_Q^cross . F_V        K_a = W_K^cross . F_A   V_a = W_V^cross . F_A
    CrossAttn_{v->a} = softmax(Q_{v->a} K_a^T / sqrt(d_v)) . V_a

    Q_{v->t} = W_Q^cross . F_V        K_t = W_K^cross . F_T   V_t = W_V^cross . F_T
    CrossAttn_{v->t} = softmax(Q_{v->t} K_t^T / sqrt(d_v)) . V_t

    F_fusion = F_V + CrossAttn_{v->t} + CrossAttn_{v->a}

Note the paper uses separate weight sets per branch conceptually; we give
each modality-pair its own projection matrices (not shared) since audio
and text generally have different raw dimensionalities anyway.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """Single-head-configurable cross attention: visual queries, one
    auxiliary modality provides keys/values."""

    def __init__(self, d_model: int, aux_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(aux_dim, d_model)
        self.v_proj = nn.Linear(aux_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """
        visual: (B, T, d_model)  -- query source
        aux:    (B, T, aux_dim)  -- key/value source
        returns: (B, T, d_model)
        """
        b, t, _ = visual.shape
        q = self.q_proj(visual).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(aux).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(aux).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.out_proj(out)


class ViCEF(nn.Module):
    def __init__(self, visual_dim: int, audio_dim: int, text_dim: int,
                 d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Project visual features into the shared fusion space first,
        # since visual_dim (4096) is typically larger than d_model.
        self.visual_proj = nn.Linear(visual_dim, d_model)

        self.attn_v2a = CrossModalAttention(d_model, audio_dim, n_heads, dropout)
        self.attn_v2t = CrossModalAttention(d_model, text_dim, n_heads, dropout)

        self.output_dim = d_model

    def forward(self, F_V: torch.Tensor, F_A: torch.Tensor, F_T: torch.Tensor) -> torch.Tensor:
        """
        F_V: (B, T, visual_dim)
        F_A: (B, T, audio_dim)
        F_T: (B, T, text_dim)
        returns F_fusion: (B, T, d_model)
        """
        v = self.visual_proj(F_V)  # (B, T, d_model)
        cross_v2a = self.attn_v2a(v, F_A)
        cross_v2t = self.attn_v2t(v, F_T)
        return v + cross_v2t + cross_v2a
