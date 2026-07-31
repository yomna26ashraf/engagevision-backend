"""
Time-series decomposition via FFT — Algorithm 1 in the paper.

    F        = FFT(X)
    F_trend  = keep only |freq| < f_trend           (low frequencies)
    T_t      = IFFT(F_trend)
    peaks    = DetectPeaks(F)                       (dominant seasonal freqs)
    F_season = keep only frequencies at `peaks`
    C_t      = IFFT(F_season)

Operates along the time dimension of a (B, T, D) tensor, independently
per feature channel D and batch element B.
"""
from __future__ import annotations

import torch


def _topk_peaks(magnitude: torch.Tensor, num_peaks: int, exclude_dc: bool = True) -> torch.Tensor:
    """magnitude: (B, F, D) real-valued FFT magnitudes.
    Returns a boolean mask (B, F, D) selecting the top-`num_peaks`
    frequency bins per (batch, channel), excluding the DC (0-freq) bin
    which already belongs to the trend."""
    mag = magnitude.clone()
    if exclude_dc:
        mag[:, 0, :] = -float("inf")
    b, f, d = mag.shape
    k = min(num_peaks, f)
    _, idx = torch.topk(mag, k=k, dim=1)  # (B, k, D)
    mask = torch.zeros_like(mag, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def fft_trend_cycle_decompose(x: torch.Tensor, trend_cutoff_ratio: float = 0.1,
                               num_peaks: int = 3):
    """
    x: (B, T, D) real-valued time series (fused multimodal features per
       time step).
    trend_cutoff_ratio: fraction of the (positive) frequency spectrum kept
       as the low-frequency "trend" band.
    num_peaks: number of dominant frequency bins (outside the trend band)
       kept to reconstruct the cyclical component.

    Returns:
        trend: (B, T, D)
        cycle: (B, T, D)
    """
    b, t, d = x.shape

    # cuFFT's half-precision kernels only support power-of-two signal
    # lengths; our clip lengths (e.g. 10-frame DAiSEE clips, 32-step
    # RoomReader windows) generally aren't. FFT is also precision-sensitive
    # by nature, so we always compute it in float32 regardless of any
    # surrounding autocast/mixed-precision context, then cast back.
    orig_dtype = x.dtype
    with torch.autocast(device_type=x.device.type, enabled=False):
        x32 = x.float()
        # rfft along the time axis -> (B, F, D) complex, F = T//2 + 1
        Xf = torch.fft.rfft(x32, dim=1)
        num_freqs = Xf.shape[1]

        cutoff = max(1, int(round(num_freqs * trend_cutoff_ratio)))

        # --- Trend: keep only the lowest `cutoff` frequency bins (incl. DC) ---
        trend_mask = torch.zeros(num_freqs, dtype=torch.bool, device=x.device)
        trend_mask[:cutoff] = True
        trend_mask_b = trend_mask.view(1, -1, 1).expand_as(Xf)
        Xf_trend = Xf.masked_fill(~trend_mask_b, 0)
        trend = torch.fft.irfft(Xf_trend, n=t, dim=1)

        # --- Cycle: keep the top-`num_peaks` magnitude bins outside the trend band ---
        magnitude = Xf.abs()  # (B, F, D)
        magnitude_outside_trend = magnitude.clone()
        magnitude_outside_trend[:, :cutoff, :] = -float("inf")
        peak_mask = _topk_peaks(magnitude_outside_trend, num_peaks=num_peaks, exclude_dc=False)
        Xf_cycle = Xf.masked_fill(~peak_mask, 0)
        cycle = torch.fft.irfft(Xf_cycle, n=t, dim=1)

    return trend.to(orig_dtype), cycle.to(orig_dtype)
