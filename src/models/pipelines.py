"""
End-to-end pipelines wiring raw inputs -> feature extractors -> M-LATTE.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .visual_backbone import DualBranchVisualEncoder
from .mlatte import MLATTEVisualOnly, MLATTEFull


class DAiSEEPipeline(nn.Module):
    """raw frames (B, T, C, H, W) -> DualBranchVisualEncoder -> MLATTEVisualOnly."""

    def __init__(self, visual_encoder: DualBranchVisualEncoder,
                 vae_d_model: int = 256, vae_heads: int = 8, vae_layers: int = 4,
                 vae_latent_dim: int = 128, vae_conv_channels=(128, 256, 256),
                 fft_trend_cutoff_ratio: float = 0.1, fft_num_peaks: int = 3,
                 class_values=(0.0, 0.25, 0.5, 1.0)):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.mlatte = MLATTEVisualOnly(
            visual_dim=visual_encoder.output_dim,
            vae_d_model=vae_d_model, vae_heads=vae_heads, vae_layers=vae_layers,
            vae_latent_dim=vae_latent_dim, vae_conv_channels=vae_conv_channels,
            fft_trend_cutoff_ratio=fft_trend_cutoff_ratio, fft_num_peaks=fft_num_peaks,
            class_values=class_values,
        )

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """frames: (B, T, C, H, W)."""
        F_V = self.visual_encoder(frames)   # (B, T, visual_dim)
        return self.mlatte(F_V)

    def score_to_class(self, score: torch.Tensor) -> torch.Tensor:
        return self.mlatte.score_to_class(score)


class TrimodalPipeline(nn.Module):
    """raw frames + audio windows + text -> feature extractors -> MLATTEFull.

    Feature extraction for audio/text is comparatively expensive; for
    large datasets it's strongly recommended to pre-extract F_A / F_T
    offline and feed cached tensors directly into MLATTEFull instead of
    running this end-to-end wrapper every step. This class is provided
    for completeness / smaller datasets (e.g. CMU-MOSI).
    """

    def __init__(self, visual_encoder, audio_encoder, text_encoder, mlatte_full: MLATTEFull):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.mlatte = mlatte_full

    def forward(self, frames: torch.Tensor, audio_batch, text_batch) -> Dict[str, torch.Tensor]:
        F_V = self.visual_encoder(frames)  # (B, T, visual_dim)
        # NOTE: audio_batch / text_batch handling depends on your exact
        # windowing; see src/data/mosi_dataset.py for the expected format.
        F_A = audio_batch
        F_T = text_batch
        return self.mlatte(F_V, F_A, F_T)
