"""
Main training script: M-LATTE visual-only pipeline on DAiSEE.

Usage:
    python scripts/train_daisee.py --config configs/config.yaml \
        --emotion_ckpt ./checkpoints/emotion_branch.pt \
        --behavior_ckpt ./checkpoints/behavior_branch.pt
"""
import argparse
import os
import sys

import torch
import yaml
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.daisee_dataset import build_daisee_dataloaders, DAISEE_RAW_TO_MERGED_CLASS, MERGED_CLASS_LABELS  # noqa: E402
from src.losses import total_loss  # noqa: E402
from src.models.visual_backbone import DualBranchVisualEncoder, EmotionBranch, BehaviorBranch  # noqa: E402
from src.models.pipelines import DAiSEEPipeline  # noqa: E402
from src.utils import set_seed, get_device, save_checkpoint, EarlyStopping, mse, mae, classification_accuracy  # noqa: E402


def build_visual_encoder(emotion_ckpt, behavior_ckpt, device, freeze_override: bool = None):
    emotion_branch = EmotionBranch(num_classes=7, pretrained_imagenet=True)
    behavior_branch = BehaviorBranch(num_classes=3, pretrained_imagenet=True)

    emotion_pretrained = bool(emotion_ckpt and os.path.exists(emotion_ckpt))
    behavior_pretrained = bool(behavior_ckpt and os.path.exists(behavior_ckpt))

    if emotion_pretrained:
        state = torch.load(emotion_ckpt, map_location="cpu")
        emotion_branch.load_state_dict(state["model_state"])
        print(f"Loaded emotion branch checkpoint from {emotion_ckpt} (val_acc={state.get('val_acc')})")
    else:
        print("WARNING: no emotion branch checkpoint provided/found — using ImageNet init only. "
              "Run scripts/pretrain_visual_branch.py first for paper-faithful results.")

    if behavior_pretrained:
        state = torch.load(behavior_ckpt, map_location="cpu")
        behavior_branch.load_state_dict(state["model_state"])
        print(f"Loaded behavior branch checkpoint from {behavior_ckpt} (val_acc={state.get('val_acc')})")
    else:
        print("WARNING: no behavior branch checkpoint provided/found — using ImageNet init only. "
              "Leaving it UNFROZEN so it can still adapt to DAiSEE during training (see README).")

    # Default policy: freeze a branch only if it has a real domain-pretrained
    # checkpoint; leave ImageNet-only branches trainable so they aren't dead
    # weight. `freeze_override`, if given, forces the same choice for both.
    freeze_emotion = freeze_override if freeze_override is not None else emotion_pretrained
    freeze_behavior = freeze_override if freeze_override is not None else behavior_pretrained

    encoder = DualBranchVisualEncoder(
        emotion_branch, behavior_branch,
        freeze_emotion=freeze_emotion, freeze_behavior=freeze_behavior,
    )
    return encoder.to(device)


def compute_class_weights(dataset, num_classes: int = 3, device=None, max_weight: float = 8.0) -> torch.Tensor:
    """Inverse-frequency weight per merged DAiSEE class (0=Low Engagement,
    1=Engaged, 2=Highly Engaged — see src/data/daisee_dataset.py for the
    very-low/low merge). NOTE: unused by default now that oversampling
    (build_class_balanced_sampler) handles imbalance instead — kept here
    for reference / in case you want to experiment with combining both."""
    counts = torch.zeros(num_classes)
    for row in dataset.rows:
        counts[DAISEE_RAW_TO_MERGED_CLASS[int(row["Engagement"])]] += 1
    counts = counts.clamp(min=1)  # avoid div-by-zero for any unseen class
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    weights = weights.clamp(max=max_weight)
    weights = weights / weights.mean()  # re-normalize after capping
    print(f"Class counts (train): {counts.tolist()}  ->  weights (capped at {max_weight}): "
          f"{[round(w, 3) for w in weights.tolist()]}")
    return weights.to(device) if device is not None else weights


def run_epoch(model, loader, optimizer, scaler, device, cfg, train: bool, class_weights=None):
    model.train() if train else model.eval()
    total_mse, total_mae, n = 0.0, 0.0, 0
    all_preds, all_labels = [], []
    accum_steps = max(1, cfg["train"].get("grad_accum_steps", 1)) if train else 1

    if train:
        optimizer.zero_grad()

    for step, batch in enumerate(loader):
        frames = batch["frames"].to(device, non_blocking=True)  # (B, T, C, H, W)
        labels = batch["label"].to(device, non_blocking=True)   # (B,)

        sample_weights = None
        if train and class_weights is not None:
            raw_labels = batch["raw_label"].to(device, non_blocking=True)  # (B,) ints 0..3
            sample_weights = class_weights[raw_labels]

        with torch.set_grad_enabled(train):
            with autocast(device.type, enabled=cfg["train"]["mixed_precision"]):
                out = model(frames)
                score = out["score"]
                losses = total_loss(
                    score, labels, out,
                    kl_weight=cfg["train"]["kl_weight"],
                    smoothness_weight=0.0,  # single-window batches here; see README for session-level variant
                    sample_weights=sample_weights,
                )
                loss = losses["loss"] / accum_steps

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        bsz = frames.size(0)
        total_mse += mse(score.detach(), labels) * bsz
        total_mae += mae(score.detach(), labels) * bsz
        n += bsz
        all_preds.append(score.detach().cpu())
        all_labels.append(labels.detach().cpu())

    # flush any leftover accumulated gradients from a partial final batch
    if train and (len(loader) % accum_steps != 0):
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return {
        "mse": total_mse / max(n, 1),
        "mae": total_mae / max(n, 1),
        "preds": all_preds,
        "labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--emotion_ckpt", default=None)
    parser.add_argument("--behavior_ckpt", default=None)
    parser.add_argument("--freeze_visual_backbone", dest="freeze_visual_backbone",
                         action="store_true", help="freeze both branches (default if pretrained checkpoints given)")
    parser.add_argument("--unfreeze_visual_backbone", dest="freeze_visual_backbone",
                         action="store_false", help="let both branches fine-tune during DAiSEE training")
    parser.set_defaults(freeze_visual_backbone=None)  # decided below based on checkpoints
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device()
    print(f"Using device: {device}")

    encoder = build_visual_encoder(args.emotion_ckpt, args.behavior_ckpt, device,
                                    freeze_override=args.freeze_visual_backbone)
    model = DAiSEEPipeline_wrap(encoder, cfg).to(device)

    train_loader, val_loader, test_loader = build_daisee_dataloaders(
        daisee_root=cfg["paths"]["daisee_root"],
        frames_root=cfg["paths"]["daisee_frames_cache"],
        batch_size=cfg["daisee"]["batch_size"],
        clip_len=cfg["daisee"]["clip_seconds"],
        num_workers=cfg["train"]["num_workers"],
        balance_classes=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=cfg["daisee"]["lr"])
    scaler = GradScaler(device.type, enabled=cfg["train"]["mixed_precision"] and device.type == "cuda")

    # NOTE: class imbalance is now handled by oversampling the rare classes
    # at the DataLoader level (see build_daisee_dataloaders(balance_classes=True)
    # in src/data/daisee_dataset.py), which gave real additional training
    # exposure to "Not Engaged" clips. We tried loss-reweighting alone
    # first and it barely moved the confusion matrix while hurting overall
    # accuracy, so it's disabled here (class_weights=None) to avoid
    # stacking two imbalance-correction mechanisms at once.
    class_weights = None

    early_stopper = EarlyStopping(patience=cfg["daisee"]["early_stopping_patience"])
    ckpt_path = os.path.join(cfg["paths"]["checkpoints"], "daisee_mlatte_best.pt")

    epoch_history = []
    for epoch in range(cfg["daisee"]["max_epochs"]):
        train_stats = run_epoch(model, train_loader, optimizer, scaler, device, cfg, train=True,
                                 class_weights=class_weights)
        val_stats = run_epoch(model, val_loader, None, None, device, cfg, train=False)

        val_classes = model.score_to_class(val_stats["preds"])
        label_classes = model.score_to_class(val_stats["labels"])
        val_acc = classification_accuracy(val_classes, label_classes)

        print(f"epoch {epoch+1}/{cfg['daisee']['max_epochs']}  "
              f"train_mse={train_stats['mse']:.4f}  val_mse={val_stats['mse']:.4f}  "
              f"val_mae={val_stats['mae']:.4f}  val_acc={val_acc:.4f}")

        epoch_history.append({
            "epoch": epoch + 1,
            "train_mse": train_stats["mse"],
            "val_mse": val_stats["mse"],
            "val_mae": val_stats["mae"],
            "val_acc": val_acc,
        })

        is_best = early_stopper.step(val_stats["mse"])
        if is_best:
            save_checkpoint({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_mse": val_stats["mse"],
                "val_acc": val_acc,
                "config": cfg,
            }, ckpt_path)
        if early_stopper.should_stop:
            print(f"Early stopping at epoch {epoch+1} (best val_mse={early_stopper.best_value:.4f})")
            break

    # Final test evaluation with best checkpoint
    best_state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_state["model_state"])
    test_stats = run_epoch(model, test_loader, None, None, device, cfg, train=False)
    test_classes = model.score_to_class(test_stats["preds"])
    test_label_classes = model.score_to_class(test_stats["labels"])
    test_acc = classification_accuracy(test_classes, test_label_classes)
    print(f"\nFINAL TEST — mse={test_stats['mse']:.4f}  mae={test_stats['mae']:.4f}  acc={test_acc:.4f}")
    print("(paper reference: DAiSEE engagement accuracy = 61.37%)")

    _save_results_json(cfg, epoch_history, test_stats, test_classes, test_label_classes, test_acc)


LEVEL_LABELS = MERGED_CLASS_LABELS  # ["Low Engagement", "Engaged", "Highly Engaged"]


def _save_results_json(cfg, epoch_history, test_stats, test_classes, test_label_classes, test_acc):
    """Writes checkpoints/results.json, read by the backend's /api/performance
    endpoint so the frontend dashboard shows real numbers instead of mocks."""
    import json
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    label_values = cfg["daisee"]["label_values"]
    value_to_idx = {v: i for i, v in enumerate(label_values)}
    y_true = [value_to_idx[float(v)] for v in test_label_classes.tolist()]
    y_pred = [value_to_idx[float(v)] for v in test_classes.tolist()]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_values)))).tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(label_values))), zero_division=0
    )
    per_class_metrics = [
        {
            "name": LEVEL_LABELS[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(label_values))
    ]

    results = {
        "epochs": epoch_history,
        "confusion_matrix": cm,
        "per_class_metrics": per_class_metrics,
        "final_test": {
            "mse": test_stats["mse"],
            "mae": test_stats["mae"],
            "accuracy": test_acc,
        },
    }
    out_path = os.path.join(cfg["paths"]["checkpoints"], "results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved dashboard/performance results to {out_path}")


def DAiSEEPipeline_wrap(encoder, cfg):
    """Thin factory so `model.score_to_class` stays reachable after wrapping."""
    from src.models.pipelines import DAiSEEPipeline
    return DAiSEEPipeline(
        visual_encoder=encoder,
        vae_d_model=cfg["trend_cycle_vae"]["d_model"],
        vae_heads=cfg["trend_cycle_vae"]["n_heads"],
        vae_layers=cfg["trend_cycle_vae"]["transformer_layers"],
        vae_latent_dim=cfg["trend_cycle_vae"]["latent_dim"],
        vae_conv_channels=tuple(cfg["trend_cycle_vae"]["conv_channels"]),
        fft_trend_cutoff_ratio=cfg["trend_cycle_vae"]["fft_trend_cutoff_ratio"],
        fft_num_peaks=cfg["trend_cycle_vae"]["fft_num_peaks"],
        class_values=tuple(cfg["daisee"]["label_values"]),
    )


if __name__ == "__main__":
    main()
