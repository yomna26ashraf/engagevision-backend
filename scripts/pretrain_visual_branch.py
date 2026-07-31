"""
Pretrain one branch (emotion OR behavior) of the dual-branch visual
encoder — reproduces Table III's training recipe:
    100 epochs, Adam-less (paper doesn't specify optimizer here; we use
    Adam for consistency with the rest of the pipeline), initial
    lr = 1e-5, decayed by a factor of 10 every 25 epochs, cross-entropy
    loss.

Expects an ImageFolder-style directory layout:
    <data_root>/train/<class_name>/*.jpg
    <data_root>/val/<class_name>/*.jpg

For the emotion branch this is RAF-DB (7 classes: neutral, happiness,
sadness, surprise, fear, disgust, anger) or a substitute such as
FER2013 restructured into class folders.

For the behavior branch this is StudentEngagementDataset (3 classes:
screen-viewing, writing, distraction) or your own annotated substitute.

Usage:
    python scripts/pretrain_visual_branch.py \
        --branch emotion --data_root ./data/RAFDB --num_classes 7 \
        --out_path ./checkpoints/emotion_branch.pt
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.preprocessing import default_face_transform  # noqa: E402
from src.models.visual_backbone import EmotionBranch, BehaviorBranch  # noqa: E402
from src.utils import set_seed, get_device, save_checkpoint  # noqa: E402


class _ImageFolderTransform:
    """Picklable wrapper (module-level class, not a lambda/closure) so this
    works with num_workers > 0 DataLoader multiprocessing on Windows,
    where worker processes are spawned and must pickle the transform."""

    def __init__(self, tf):
        self.tf = tf

    def __call__(self, img):
        return self.tf(_pil_to_np(img))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=["emotion", "behavior"], required=True)
    parser.add_argument("--data_root", required=True, help="dir with train/ and val/ subfolders")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_decay_factor", type=float, default=10.0)
    parser.add_argument("--lr_decay_every", type=int, default=25)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4,
                         help="set to 0 on Windows if you still hit multiprocessing errors")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    train_tf = default_face_transform(args.image_size, train=True)
    val_tf = default_face_transform(args.image_size, train=False)

    train_ds = datasets.ImageFolder(os.path.join(args.data_root, "train"),
                                     transform=_ImageFolderTransform(train_tf))
    val_ds = datasets.ImageFolder(os.path.join(args.data_root, "val"),
                                   transform=_ImageFolderTransform(val_tf))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True,
                               persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=args.num_workers > 0)

    ModelCls = EmotionBranch if args.branch == "emotion" else BehaviorBranch
    model = ModelCls(num_classes=args.num_classes, pretrained_imagenet=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_every, gamma=1.0 / args.lr_decay_factor
    )

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        scheduler.step()
        train_acc = correct / max(total, 1)
        train_loss = running_loss / max(total, 1)

        model.eval()
        v_total, v_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += images.size(0)
        val_acc = v_correct / max(v_total, 1)

        print(f"epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "branch": args.branch,
                "num_classes": args.num_classes,
            }, args.out_path)

    print(f"Best val_acc = {best_val_acc:.4f}. Checkpoint saved to {args.out_path}")


def _pil_to_np(img):
    import numpy as np
    return np.array(img.convert("RGB"))


if __name__ == "__main__":
    main()
