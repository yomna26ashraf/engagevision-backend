"""
Standalone evaluation of a trained DAiSEE checkpoint on the test split.

Usage:
    python scripts/evaluate_daisee.py --checkpoint ./checkpoints/daisee_mlatte_best.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.daisee_dataset import build_daisee_dataloaders  # noqa: E402
from src.models.visual_backbone import DualBranchVisualEncoder, EmotionBranch, BehaviorBranch  # noqa: E402
from src.models.pipelines import DAiSEEPipeline  # noqa: E402
from src.utils import get_device, mse, mae, classification_accuracy  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = get_device()
    state = torch.load(args.checkpoint, map_location=device)
    cfg = state["config"]

    encoder = DualBranchVisualEncoder(
        EmotionBranch(num_classes=7, pretrained_imagenet=False),
        BehaviorBranch(num_classes=3, pretrained_imagenet=False),
        freeze_backbones=True,
    )
    model = DAiSEEPipeline(
        visual_encoder=encoder,
        vae_d_model=cfg["trend_cycle_vae"]["d_model"],
        vae_heads=cfg["trend_cycle_vae"]["n_heads"],
        vae_layers=cfg["trend_cycle_vae"]["transformer_layers"],
        vae_latent_dim=cfg["trend_cycle_vae"]["latent_dim"],
        vae_conv_channels=tuple(cfg["trend_cycle_vae"]["conv_channels"]),
        fft_trend_cutoff_ratio=cfg["trend_cycle_vae"]["fft_trend_cutoff_ratio"],
        fft_num_peaks=cfg["trend_cycle_vae"]["fft_num_peaks"],
        class_values=tuple(cfg["daisee"]["label_values"]),
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    _, _, test_loader = build_daisee_dataloaders(
        daisee_root=cfg["paths"]["daisee_root"],
        frames_root=cfg["paths"]["daisee_frames_cache"],
        batch_size=cfg["daisee"]["batch_size"],
        clip_len=cfg["daisee"]["clip_seconds"],
        num_workers=cfg["train"]["num_workers"],
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            frames = batch["frames"].to(device)
            labels = batch["label"].to(device)
            out = model(frames)
            all_preds.append(out["score"].cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    pred_classes = model.score_to_class(preds)
    label_classes = model.score_to_class(labels)

    print(f"Test MSE: {mse(preds, labels):.4f}")
    print(f"Test MAE: {mae(preds, labels):.4f}")
    print(f"Test Accuracy (nearest-class): {classification_accuracy(pred_classes, label_classes):.4f}")
    print("Paper reference: DAiSEE engagement accuracy = 61.37%")


if __name__ == "__main__":
    main()
