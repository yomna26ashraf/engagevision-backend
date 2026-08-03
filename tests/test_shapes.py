"""
Sanity-check the architecture with synthetic tensors — no real data or
pretrained weights needed. Run this FIRST on your machine right after
`pip install -r requirements.txt`, before touching real datasets:

    pytest tests/test_shapes.py -v

If everything here passes, the tensor plumbing (FFT decomposition, ViCEF,
TrendCycleVAE, regression heads) is wired correctly end-to-end.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.fft_decomposition import fft_trend_cycle_decompose
from src.models.vicef import ViCEF
from src.models.trend_cycle_vae import TrendCycleVAE
from src.models.mlatte import MLATTEFull, MLATTEVisualOnly
from src.losses import total_loss


def test_fft_decomposition_shapes():
    x = torch.randn(4, 32, 64)  # (B, T, D)
    trend, cycle = fft_trend_cycle_decompose(x, trend_cutoff_ratio=0.1, num_peaks=3)
    assert trend.shape == x.shape
    assert cycle.shape == x.shape


def test_vicef_forward():
    b, t = 2, 16
    vicef = ViCEF(visual_dim=4096, audio_dim=128, text_dim=768, d_model=256, n_heads=8)
    F_V = torch.randn(b, t, 4096)
    F_A = torch.randn(b, t, 128)
    F_T = torch.randn(b, t, 768)
    out = vicef(F_V, F_A, F_T)
    assert out.shape == (b, t, 256)


def test_trend_cycle_vae_forward():
    b, t, d = 2, 32, 256
    vae = TrendCycleVAE(input_dim=d, d_model=128, n_heads=4, num_layers=2, latent_dim=32)
    x = torch.randn(b, t, d)
    out = vae(x)
    assert out["A"].shape == (b, 32 * 4)  # 4 * latent_dim
    assert out["recon"].shape == x.shape


def test_mlatte_full_forward_and_loss():
    b, t = 2, 32
    model = MLATTEFull(visual_dim=4096, audio_dim=128, text_dim=768,
                        vae_d_model=64, vae_heads=4, vae_layers=2, vae_latent_dim=16)
    F_V = torch.randn(b, t, 4096)
    F_A = torch.randn(b, t, 128)
    F_T = torch.randn(b, t, 768)
    out = model(F_V, F_A, F_T)
    assert out["score"].shape == (b,)

    target = torch.rand(b) * 4 - 2  # in [-2, 2]
    losses = total_loss(out["score"], target, out, kl_weight=1.0)
    assert torch.isfinite(losses["loss"])


def test_mlatte_visual_only_forward():
    b, t = 3, 10
    model = MLATTEVisualOnly(visual_dim=4096, vae_d_model=64, vae_heads=4,
                              vae_layers=2, vae_latent_dim=16)
    F_V = torch.randn(b, t, 4096)
    out = model(F_V)
    assert out["score"].shape == (b,)
    classes = model.score_to_class(out["score"])
    assert classes.shape == (b,)
    for c in classes.tolist():
        assert c in (0.0, 0.5, 1.0)


if __name__ == "__main__":
    test_fft_decomposition_shapes()
    test_vicef_forward()
    test_trend_cycle_vae_forward()
    test_mlatte_full_forward_and_loss()
    test_mlatte_visual_only_forward()
    print("All shape sanity checks passed.")
