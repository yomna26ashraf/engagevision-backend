"""
DAiSEE dataset loader (visual-only path, Section IV-C-2 + Data Processing).

Expects:
  - Pre-extracted 1-fps frames under `frames_root/<split>/<clip_id>/frame_*.jpg`
    (see scripts/extract_daisee_frames.py)
  - Official label CSVs (`Labels/TrainLabels.csv` etc.) with columns:
    ClipID, Boredom, Engagement, Confusion, Frustration (each 0-3)

Engagement labels (0,1,2,3 = very-low..very-high) are mapped to continuous
values {0.0, 0.25, 0.5, 1.0} exactly as described in the paper's Data
Processing subsection.
"""
from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import default_face_transform, load_frame_as_rgb, pad_or_truncate_frames

DAISEE_ENGAGEMENT_MAP = {0: 0.0, 1: 0.25, 2: 0.5, 3: 1.0}


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

        raw_label = int(row["Engagement"])
        label = DAISEE_ENGAGEMENT_MAP[raw_label]

        return {
            "clip_id": clip_id,
            "frames": frames,                      # (clip_len, C, H, W)
            "label": torch.tensor(label, dtype=torch.float32),
            "raw_label": raw_label,
        }


def build_daisee_dataloaders(daisee_root: str, frames_root: str, batch_size: int = 32,
                              clip_len: int = 10, image_size: int = 224,
                              num_workers: int = 4):
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
