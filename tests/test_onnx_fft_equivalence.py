"""
Sanity check: MatmulFFTTrendCycleDecomposer (fft_matrix.py, used for ONNX
export) must produce numerically identical output to the original
torch.fft-based fft_trend_cycle_decompose() (fft_decomposition.py), for
every fixed sequence length we actually use.

Run this BEFORE trusting any ONNX export built on top of the matmul
version — if this fails, the ONNX model's predictions cannot be trusted
either, no matter how "successful" the export itself looked.

    pytest tests/test_onnx_fft_equivalence.py -v
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.fft_decomposition import fft_trend_cycle_decompose
from src.models.fft_matrix import MatmulFFTTrendCycleDecomposer


def _check_equivalence(seq_len: int, dim: int = 32, batch: int = 4,
                        trend_cutoff_ratio: float = 0.1, num_peaks: int = 3,
                        atol: float = 1e-4):
    torch.manual_seed(0)
    x = torch.randn(batch, seq_len, dim)

    trend_ref, cycle_ref = fft_trend_cycle_decompose(
        x, trend_cutoff_ratio=trend_cutoff_ratio, num_peaks=num_peaks
    )

    decomposer = MatmulFFTTrendCycleDecomposer(
        seq_len=seq_len, trend_cutoff_ratio=trend_cutoff_ratio, num_peaks=num_peaks
    )
    trend_mm, cycle_mm = decomposer(x)

    trend_diff = (trend_ref - trend_mm).abs().max().item()
    cycle_diff = (cycle_ref - cycle_mm).abs().max().item()

    assert trend_diff < atol, (
        f"[seq_len={seq_len}] trend mismatch: max abs diff {trend_diff} >= {atol}"
    )
    # NOTE: the "cycle" component can legitimately differ if two frequency
    # bins are tied in magnitude and torch.topk / our matmul-based topk
    # break the tie differently — this is extremely unlikely with random
    # float inputs (measure-zero event), but if this assertion ever fails,
    # check for tied magnitudes before assuming a real bug.
    assert cycle_diff < atol, (
        f"[seq_len={seq_len}] cycle mismatch: max abs diff {cycle_diff} >= {atol}. "
        f"If inputs weren't random (e.g. contain exact duplicate frequency "
        f"magnitudes), this can be a tie-breaking difference rather than a bug."
    )
    return trend_diff, cycle_diff


def test_equivalence_daisee_clip_len():
    """The length that actually matters: DAiSEE's 10-frame clips."""
    trend_diff, cycle_diff = _check_equivalence(seq_len=10)
    print(f"seq_len=10: trend_diff={trend_diff:.2e}  cycle_diff={cycle_diff:.2e}")


def test_equivalence_roomreader_window():
    """The paper's optimal RoomReader window length, in case that path
    ever gets ONNX-exported too."""
    trend_diff, cycle_diff = _check_equivalence(seq_len=32)
    print(f"seq_len=32: trend_diff={trend_diff:.2e}  cycle_diff={cycle_diff:.2e}")


def test_equivalence_various_lengths():
    for seq_len in [4, 5, 8, 16, 20]:
        trend_diff, cycle_diff = _check_equivalence(seq_len=seq_len)
        print(f"seq_len={seq_len}: trend_diff={trend_diff:.2e}  cycle_diff={cycle_diff:.2e}")


if __name__ == "__main__":
    test_equivalence_daisee_clip_len()
    test_equivalence_roomreader_window()
    test_equivalence_various_lengths()
    print("All FFT-vs-matmul equivalence checks passed.")
