"""Shared preprocessing utilities: frame extraction and image transforms."""
from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_face_transform(image_size: int = 224, train: bool = False) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def extract_frames_at_fps(video_path: str, out_dir: str, target_fps: int = 1) -> List[str]:
    """Extract frames from `video_path` at `target_fps` and save as JPEGs in
    `out_dir`. Returns the list of saved frame paths (sorted). Idempotent:
    skips extraction if `out_dir` already contains frames.
    """
    os.makedirs(out_dir, exist_ok=True)
    existing = sorted(f for f in os.listdir(out_dir) if f.endswith(".jpg"))
    if existing:
        return [os.path.join(out_dir, f) for f in existing]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, round(native_fps / target_fps))

    saved_paths = []
    frame_idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = os.path.join(out_dir, f"frame_{saved_idx:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_paths.append(out_path)
            saved_idx += 1
        frame_idx += 1
    cap.release()
    return saved_paths


def load_frame_as_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Could not read frame: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pad_or_truncate_frames(frames: torch.Tensor, target_len: int) -> torch.Tensor:
    """frames: (T, C, H, W). Pads by repeating the last frame, or truncates
    (uniform subsampling) to exactly `target_len` frames."""
    t = frames.shape[0]
    if t == target_len:
        return frames
    if t > target_len:
        idx = torch.linspace(0, t - 1, target_len).long()
        return frames[idx]
    # pad by repeating last frame
    pad = frames[-1:].repeat(target_len - t, 1, 1, 1)
    return torch.cat([frames, pad], dim=0)
