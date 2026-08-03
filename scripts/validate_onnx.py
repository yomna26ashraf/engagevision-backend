"""
Validates an exported ONNX model against the original PyTorch checkpoint:
runs the SAME random input through both, and asserts the scores match
within a small tolerance. This is the real proof the export is trustworthy
— a "successful" export with no errors doesn't guarantee correct numbers.

Usage:
    pip install onnxruntime  # if not already installed
    python scripts/validate_onnx.py \
        --checkpoint ./checkpoints/daisee_mlatte_best.pt \
        --onnx ./checkpoints/daisee_mlatte.onnx \
        --num_samples 5
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from scripts.export_onnx import build_onnx_safe_model  # noqa: E402
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from export_onnx import build_onnx_safe_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except Exception as e:
        print(f"Could not import onnxruntime — real error was:\n  {type(e).__name__}: {e}\n"
              f"If it's not simply 'not installed', try: pip install --force-reinstall onnxruntime")
        sys.exit(1)

    device = torch.device("cpu")
    pt_model, clip_len = build_onnx_safe_model(args.checkpoint, device)
    pt_model.eval()

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    torch.manual_seed(123)
    max_abs_diff = 0.0
    all_close = True

    for i in range(args.num_samples):
        frames = torch.randn(1, clip_len, 3, 224, 224)

        with torch.no_grad():
            pt_score = pt_model(frames)["score"].numpy()

        onnx_score = session.run(["score"], {"frames": frames.numpy().astype(np.float32)})[0]

        diff = np.abs(pt_score - onnx_score).max()
        max_abs_diff = max(max_abs_diff, diff)
        status = "OK" if diff < args.atol else "MISMATCH"
        if diff >= args.atol:
            all_close = False
        print(f"sample {i}: pytorch={pt_score.item():.6f}  onnx={onnx_score.item():.6f}  "
              f"diff={diff:.2e}  [{status}]")

    print(f"\nMax abs diff across {args.num_samples} random samples: {max_abs_diff:.2e} "
          f"(tolerance: {args.atol})")
    if all_close:
        print("✅ ONNX export matches the PyTorch model. Safe to deploy.")
    else:
        print("❌ ONNX export does NOT match closely enough — do not deploy this yet. "
              "Check for un-exported randomness, unsupported ops silently falling back, "
              "or a seq_len mismatch between the checkpoint's clip_len and the export.")
        sys.exit(1)


if __name__ == "__main__":
    main()
