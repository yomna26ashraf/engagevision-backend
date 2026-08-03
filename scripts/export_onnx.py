"""
Exports the trained DAiSEE M-LATTE pipeline to ONNX.

Loads your trained checkpoint, rebuilds the model in "onnx_safe" mode
(matmul-based FFT instead of torch.fft — see src/models/fft_matrix.py),
copies the trained weights over, and exports to a fixed-shape ONNX graph
(batch size is dynamic; clip length is FIXED at the value used during
training, since the FFT matrices are baked in for that exact length).

Usage:
    python scripts/export_onnx.py \
        --checkpoint ./checkpoints/daisee_mlatte_best.pt \
        --out ./checkpoints/daisee_mlatte.onnx

Then validate it matches the original PyTorch model:
    python scripts/validate_onnx.py \
        --checkpoint ./checkpoints/daisee_mlatte_best.pt \
        --onnx ./checkpoints/daisee_mlatte.onnx
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.visual_backbone import DualBranchVisualEncoder, EmotionBranch, BehaviorBranch  # noqa: E402
from src.models.pipelines import DAiSEEPipeline  # noqa: E402


def build_onnx_safe_model(checkpoint_path: str, device: torch.device) -> DAiSEEPipeline:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = state["config"]
    vae_cfg = cfg["trend_cycle_vae"]
    daisee_cfg = cfg["daisee"]
    clip_len = daisee_cfg["clip_seconds"]

    encoder = DualBranchVisualEncoder(
        EmotionBranch(num_classes=7, pretrained_imagenet=False),
        BehaviorBranch(num_classes=3, pretrained_imagenet=False),
        freeze_backbones=True,
    )
    model = DAiSEEPipeline(
        visual_encoder=encoder,
        vae_d_model=vae_cfg["d_model"],
        vae_heads=vae_cfg["n_heads"],
        vae_layers=vae_cfg["transformer_layers"],
        vae_latent_dim=vae_cfg["latent_dim"],
        vae_conv_channels=tuple(vae_cfg["conv_channels"]),
        fft_trend_cutoff_ratio=vae_cfg["fft_trend_cutoff_ratio"],
        fft_num_peaks=vae_cfg["fft_num_peaks"],
        class_values=tuple(daisee_cfg["label_values"]),
        onnx_safe=True,          # <-- the important bit: matmul FFT, no randomness at eval
        seq_len=clip_len,
    ).to(device)

    model.load_state_dict(state["model_state"])
    model.eval()  # also makes the VAE's reparameterization deterministic (see trend_cycle_vae.py)
    return model, clip_len


class _ScoreOnlyWrapper(torch.nn.Module):
    """torch.onnx.export wants clean tensor outputs, not the full dict
    DAiSEEPipeline.forward() returns (which also carries VAE internals
    used only for the training loss). This wrapper exposes just the
    final engagement score."""

    def __init__(self, pipeline: DAiSEEPipeline):
        super().__init__()
        self.pipeline = pipeline

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.pipeline(frames)["score"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="./checkpoints/daisee_mlatte.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    device = torch.device("cpu")  # export from CPU for a CPU-deployment-friendly graph
    model, clip_len = build_onnx_safe_model(args.checkpoint, device)
    export_model = _ScoreOnlyWrapper(model).eval()

    dummy = torch.randn(1, clip_len, 3, 224, 224, device=device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.onnx.export(
        export_model,
        (dummy,),
        args.out,
        input_names=["frames"],
        output_names=["score"],
        dynamic_axes={"frames": {0: "batch"}, "score": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {args.out} (clip_len={clip_len} fixed, batch dynamic)")
    print("Next: python scripts/validate_onnx.py "
          f"--checkpoint {args.checkpoint} --onnx {args.out}")


if __name__ == "__main__":
    main()
