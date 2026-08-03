"""
DAiSEE dataset loader (visual-only path, Section IV-C-2 + Data Processing).

Expects:
  - Pre-extracted 1-fps frames under `frames_root/<split>/<clip_id>/frame_*.jpg`
    (see scripts/extract_daisee_frames.py)
  - Official label CSVs (`Labels/TrainLabels.csv` etc.) with columns:
    ClipID, Boredom, Engagement, Confusion, Frustration (each 0-3)

DEVIATION FROM THE PAPER (documented, evidence-based): the paper maps all
four raw Engagement codes (very-low, low, high, very-high) to four
continuous values. In our DAiSEE split, "very-low" has only ~33 training
clips (~0.7%) vs. ~2500 for "high" — far too few to learn from, and this
matches a well-documented issue in the literature (e.g. Zheng et al. 2024,
Dewan et al. 2018): several DAiSEE studies merge "very-low" and "low"
into one "Low Engagement" class because (a) very-low is chronically
under-represented, and (b) multiple papers report that human annotators
themselves struggle to reliably distinguish very-low from low in 10s
clips. We follow that same practice here, giving three continuous target
levels: Low Engagement (0.0), Engaged (0.5), Highly Engaged (1.0).
"""
from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import default_face_transform, load_frame_as_rgb, pad_or_truncate_frames

# Raw DAiSEE Engagement code (0=very-low, 1=low, 2=high, 3=very-high) ->
# continuous target. 0 and 1 are merged into a single "Low Engagement"
# level (see module docstring).
DAISEE_ENGAGEMENT_MAP = {0: 0.0, 1: 0.0, 2: 0.5, 3: 1.0}

# Same merge, expressed as a 0..2 bucket index — used wherever we need an
# integer class id (oversampling, class-weighting, confusion matrices).
DAISEE_RAW_TO_MERGED_CLASS = {0: 0, 1: 0, 2: 1, 3: 2}
MERGED_CLASS_LABELS = ["Low Engagement", "Engaged", "Highly Engaged"]


class DAiSEEDataset(Dataset):
    def __init__(self, labels_csv: str, frames_root: str, split: str,
                 clip_len: int = 10, image_size: int = 224, train: bool = False):
        """
        labels_csv: path to e.g. Labels/TrainLabels.csv
        frames_root: root dir containing <split>/<clip_id>/frame_*.jpg
        split: "Train" | "Validation" | "Test" (must match the frames_root subfolder)
        clip_len: number of frames per clip after padding/truncation (paper:
                  10-second clips at 1 fps -> 10 frames)
        """
        df = pd.read_csv(labels_csv)
        df.columns = [c.strip() for c in df.columns]
        self.df = df
        self.frames_root = os.path.join(frames_root, split)
        self.clip_len = clip_len
        self.transform = default_face_transform(image_size=image_size, train=train)

        # Filter out clips whose frame directory doesn't exist (e.g. not
        # yet extracted, or a failed video read).
        valid_rows = []
        for _, row in self.df.iterrows():
            clip_id = os.path.splitext(str(row["ClipID"]).strip())[0]
            clip_dir = os.path.join(self.frames_root, clip_id)
            if os.path.isdir(clip_dir) and len(os.listdir(clip_dir)) > 0:
                valid_rows.append(row)
        if len(valid_rows) == 0:
            raise RuntimeError(
                f"No valid extracted clips found under {self.frames_root}. "
                f"Did you run scripts/extract_daisee_frames.py first?"
            )
        self.rows = valid_rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        clip_id = os.path.splitext(str(row["ClipID"]).strip())[0]
        clip_dir = os.path.join(self.frames_root, clip_id)

        frame_files = sorted(f for f in os.listdir(clip_dir) if f.endswith(".jpg"))
        frames = []
        for f in frame_files:
            img = load_frame_as_rgb(os.path.join(clip_dir, f))
            frames.append(self.transform(img))
        frames = torch.stack(frames, dim=0)  # (T, C, H, W)
        frames = pad_or_truncate_frames(frames, self.clip_len)

        raw_label = DAISEE_RAW_TO_MERGED_CLASS[int(row["Engagement"])]  # 0=Low, 1=Engaged, 2=Highly
        label = DAISEE_ENGAGEMENT_MAP[int(row["Engagement"])]

        return {
            "clip_id": clip_id,
            "frames": frames,                      # (clip_len, C, H, W)
            "label": torch.tensor(label, dtype=torch.float32),
            "raw_label": raw_label,                 # merged 0..2 class index
        }


def build_class_balanced_sampler(dataset, num_classes: int = 3):
    """Builds a WeightedRandomSampler that oversamples the rarer merged
    DAiSEE classes (see module docstring for the very-low/low merge).
    Unlike loss-reweighting alone, this changes how often each sample is
    actually *drawn* during training — rare-class clips get seen (and
    backpropagated through, with fresh augmentation each time) far more
    often per epoch, giving the model real training signal on them
    instead of just a bigger loss penalty on rare misses.

    Samples with replacement; epoch length stays equal to len(dataset), so
    training time per epoch is unchanged — only the class composition of
    each epoch shifts toward balance.
    """
    from torch.utils.data import WeightedRandomSampler

    counts = torch.zeros(num_classes)
    raw_labels = [DAISEE_RAW_TO_MERGED_CLASS[int(row["Engagement"])] for row in dataset.rows]
    for label in raw_labels:
        counts[label] += 1
    counts = counts.clamp(min=1)
    class_weight = 1.0 / counts  # higher weight = drawn more often
    sample_weights = torch.tensor([class_weight[label] for label in raw_labels])

    print(f"[oversampling] class counts: {counts.tolist()} "
          f"-> per-draw class weight: {[round(w, 4) for w in class_weight.tolist()]}")

    return WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)


def build_daisee_dataloaders(daisee_root: str, frames_root: str, batch_size: int = 32,
                              clip_len: int = 10, image_size: int = 224,
                              num_workers: int = 4, balance_classes: bool = True):
    from torch.utils.data import DataLoader

    labels_dir = os.path.join(daisee_root, "Labels")
    train_ds = DAiSEEDataset(
        os.path.join(labels_dir, "TrainLabels.csv"), frames_root, "Train",
        clip_len=clip_len, image_size=image_size, train=True,
    )
    val_ds = DAiSEEDataset(
        os.path.join(labels_dir, "ValidationLabels.csv"), frames_root, "Validation",
        clip_len=clip_len, image_size=image_size, train=False,
    )
    test_ds = DAiSEEDataset(
        os.path.join(labels_dir, "TestLabels.csv"), frames_root, "Test",
        clip_len=clip_len, image_size=image_size, train=False,
    )

    if balance_classes:
        sampler = build_class_balanced_sampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                   num_workers=num_workers, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
