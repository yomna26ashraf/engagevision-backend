"""
Precomputed-matrix version of the trend/cycle FFT decomposition
(Algorithm 1), mathematically identical to fft_decomposition.py's
torch.fft-based version but built entirely from matrix multiplications +
top-k selection — so it exports cleanly to ONNX (torch.fft ops have poor
ONNX Runtime support, and none at all for fixed odd-length real signals
like ours: clip_len=10).

THE TRICK: for a FIXED sequence length T (true for our deployment — the
clip length is always the same at inference, e.g. 10 for DAiSEE), the
forward real-FFT (rfft) and its inverse (irfft) are both *linear*,
T-only-dependent operators. A linear operator is fully determined by its
action on the standard basis vectors, so we derive its matrix form ONCE,
eagerly, by feeding basis vectors through the real torch.fft functions —
this happens outside of (and before) any exported graph, using ordinary
PyTorch that still has full FFT support. The resulting matrices are then
frozen as constants (buffers) and reused as plain matmuls forever after,
which *is* well supported everywhere, including ONNX Runtime, mobile
backends, etc.

Only the peak/trend-band SELECTION is genuinely data-dependent (which
frequency bins count as the "cycle"); that stays as ordinary top-k +
masking, both of which are standard, well-supported ONNX ops.

CORRECTNESS: this is not an approximation. `tests/test_onnx_fft_equivalence.py`
asserts this module's output matches `fft_decomposition.py`'s torch.fft-based
output to float32 precision on random inputs — run it after installing
torch on a machine with FFT support before trusting an ONNX export built
on top of this module.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def build_dft_matrices(seq_len: int):
    """Derives the (F, T) forward and (T, F) inverse real/imaginary DFT
    matrices for sequence length `seq_len`, by feeding standard basis
    vectors through torch.fft.rfft / torch.fft.irfft once. F = T//2 + 1.

    Returns (dft_real, dft_imag, idft_real, idft_imag), each a float32
    tensor. This function itself still needs a torch build with FFT
    support (i.e. run it once, eagerly, at model-build time) — its OUTPUT
    is what gets baked into an ONNX-exportable module, not this function.
    """
    t = seq_len
    f = t // 2 + 1

    eye_t = torch.eye(t, dtype=torch.float64)
    Xf = torch.fft.rfft(eye_t, dim=0)  # (F, T) complex; column n = rfft(e_n)
    dft_real = Xf.real.contiguous()    # (F, T)
    dft_imag = Xf.imag.contiguous()    # (F, T)

    eye_f = torch.eye(f, dtype=torch.float64)
    zeros_f = torch.zeros(f, f, dtype=torch.float64)
    spec_real_basis = torch.complex(eye_f, zeros_f)   # (F, F): col k = e_k + 0i
    spec_imag_basis = torch.complex(zeros_f, eye_f)   # (F, F): col k = 0 + i*e_k

    idft_real = torch.fft.irfft(spec_real_basis, n=t, dim=0)  # (T, F)
    idft_imag = torch.fft.irfft(spec_imag_basis, n=t, dim=0)  # (T, F)

    return (
        dft_real.float(), dft_imag.float(),
        idft_real.float(), idft_imag.float(),
    )


class MatmulFFTTrendCycleDecomposer(nn.Module):
    """Fixed-T, ONNX-exportable replacement for
    `fft_decomposition.fft_trend_cycle_decompose()`. Numerically identical
    to the torch.fft-based version for the given `seq_len` (by
    construction — see module docstring), but uses no torch.fft calls at
    all, so it's safe to include inside an exported graph.

    Usage is a drop-in match for the free function it replaces:
        decomposer = MatmulFFTTrendCycleDecomposer(seq_len=10)
        trend, cycle = decomposer(x)   # x: (B, T, D)
    """

    def __init__(self, seq_len: int, trend_cutoff_ratio: float = 0.1, num_peaks: int = 3):
        super().__init__()
        self.seq_len = seq_len
        self.num_peaks = num_peaks

        dft_real, dft_imag, idft_real, idft_imag = build_dft_matrices(seq_len)
        self.register_buffer("dft_real", dft_real, persistent=False)    # (F, T)
        self.register_buffer("dft_imag", dft_imag, persistent=False)    # (F, T)
        self.register_buffer("idft_real", idft_real, persistent=False)  # (T, F)
        self.register_buffer("idft_imag", idft_imag, persistent=False)  # (T, F)

        num_freqs = seq_len // 2 + 1
        cutoff = max(1, round(num_freqs * trend_cutoff_ratio))
        trend_mask = torch.zeros(num_freqs, dtype=torch.bool)
        trend_mask[:cutoff] = True
        self.register_buffer("trend_mask", trend_mask, persistent=False)
        self.cutoff = cutoff

    def forward(self, x: torch.Tensor):
        """x: (B, T, D). Returns (trend, cycle), each (B, T, D)."""
        # Forward "FFT" via matmul: Xr/Xi are the real/imag spectrum, (B, F, D).
        Xr = torch.einsum("ft,btd->bfd", self.dft_real, x)
        Xi = torch.einsum("ft,btd->bfd", self.dft_imag, x)

        mask = self.trend_mask.view(1, -1, 1)  # (1, F, 1), broadcasts over batch/channels

        # --- Trend: low-frequency band, fixed regardless of input content ---
        Xr_trend = Xr.masked_fill(~mask, 0.0)
        Xi_trend = Xi.masked_fill(~mask, 0.0)
        trend = (
            torch.einsum("tf,bfd->btd", self.idft_real, Xr_trend)
            + torch.einsum("tf,bfd->btd", self.idft_imag, Xi_trend)
        )

        # --- Cycle: top-k magnitude bins outside the trend band (data-dependent) ---
        magnitude = Xr.pow(2) + Xi.pow(2)  # squared magnitude — same top-k ranking as |X|, no sqrt needed
        magnitude_outside_trend = magnitude.masked_fill(mask, float("-inf"))
        k = min(self.num_peaks, self.seq_len // 2 + 1)
        _, idx = torch.topk(magnitude_outside_trend, k=k, dim=1)  # idx: (B, k, D)

        # Build the boolean peak mask via one_hot + sum instead of an
        # in-place scatter_ — mathematically identical, but scatter_'s
        # bool-tensor overload isn't traceable by torch.onnx.export's
        # legacy (TorchScript-based) exporter, while one_hot is standard
        # and well supported.
        num_freqs = magnitude.shape[1]
        one_hot = torch.nn.functional.one_hot(idx, num_classes=num_freqs)  # (B, k, D, F)
        peak_mask = one_hot.sum(dim=1).permute(0, 2, 1).bool()  # (B, F, D)

        Xr_cycle = Xr.masked_fill(~peak_mask, 0.0)
        Xi_cycle = Xi.masked_fill(~peak_mask, 0.0)
        cycle = (
            torch.einsum("tf,bfd->btd", self.idft_real, Xr_cycle)
            + torch.einsum("tf,bfd->btd", self.idft_imag, Xi_cycle)
        )

        return trend, cycle
