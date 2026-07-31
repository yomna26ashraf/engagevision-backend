"""
Loads the trained (or, if unavailable, untrained-but-functional) M-LATTE
DAiSEE pipeline once at startup, and exposes a simple `predict()` used by
the API layer.

If no checkpoint is found at MLATTE_CHECKPOINT (env var, default
`../checkpoints/daisee_mlatte_best.pt`), we still build the model with
ImageNet-initialized weights so the API is usable end-to-end for
frontend/integration testing before training finishes — every response
clearly reports `model_status: "untrained_demo"` in that case so nobody
mistakes placeholder scores for real ones.
"""
from __future__ import annotations

import os
import sys
from typing import List

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.preprocessing import default_face_transform, pad_or_truncate_frames  # noqa: E402
from src.models.visual_backbone import DualBranchVisualEncoder, EmotionBranch, BehaviorBranch  # noqa: E402
from src.models.pipelines import DAiSEEPipeline  # noqa: E402

try:
    from .schemas import LEVEL_LABELS, LEVEL_VALUES
except ImportError:
    from schemas import LEVEL_LABELS, LEVEL_VALUES

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "daisee_mlatte_best.pt")


def _maybe_download_checkpoint(checkpoint_path: str):
    """If the checkpoint isn't present locally but MLATTE_CHECKPOINT_URL is
    set (e.g. a direct link to the file on Hugging Face Hub / S3 / a
    release asset), download it once at startup. Handy for deployment,
    where you don't want a large binary checkpoint sitting in git."""
    if os.path.exists(checkpoint_path):
        return
    url = os.environ.get("MLATTE_CHECKPOINT_URL")
    if not url:
        return
    import urllib.request
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    print(f"Downloading checkpoint from {url} -> {checkpoint_path} ...")
    urllib.request.urlretrieve(url, checkpoint_path)
    print("Checkpoint download complete.")


class MLatteService:
    def __init__(self, checkpoint_path: str = None, config_path: str = DEFAULT_CONFIG_PATH):
        checkpoint_path = checkpoint_path or os.environ.get("MLATTE_CHECKPOINT", DEFAULT_CHECKPOINT)
        _maybe_download_checkpoint(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.model_status = "untrained_demo"
        state = None
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            cfg = state.get("config", cfg)
            self.model_status = "trained"

        vae_cfg = cfg["trend_cycle_vae"]
        daisee_cfg = cfg["daisee"]

        encoder = DualBranchVisualEncoder(
            EmotionBranch(num_classes=7, pretrained_imagenet=(state is None)),
            BehaviorBranch(num_classes=3, pretrained_imagenet=(state is None)),
            freeze_backbones=True,
        )
        self.model = DAiSEEPipeline(
            visual_encoder=encoder,
            vae_d_model=vae_cfg["d_model"],
            vae_heads=vae_cfg["n_heads"],
            vae_layers=vae_cfg["transformer_layers"],
            vae_latent_dim=vae_cfg["latent_dim"],
            vae_conv_channels=tuple(vae_cfg["conv_channels"]),
            fft_trend_cutoff_ratio=vae_cfg["fft_trend_cutoff_ratio"],
            fft_num_peaks=vae_cfg["fft_num_peaks"],
            class_values=tuple(daisee_cfg["label_values"]),
        ).to(self.device)

        if state is not None:
            self.model.load_state_dict(state["model_state"])

        self.model.eval()
        self.clip_len = daisee_cfg["clip_seconds"]
        self.transform = default_face_transform(image_size=224, train=False)

    @torch.no_grad()
    def predict_from_frames(self, frames_bgr: List[np.ndarray]):
        """frames_bgr: list of HxWx3 BGR numpy arrays (as read by cv2)."""
        if not frames_bgr:
            raise ValueError("No frames provided")

        tensors = []
        for f in frames_bgr:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(rgb))
        clip = torch.stack(tensors, dim=0)               # (T, C, H, W)
        clip = pad_or_truncate_frames(clip, self.clip_len)
        clip = clip.unsqueeze(0).to(self.device)          # (1, T, C, H, W)

        out = self.model(clip)
        score = out["score"].item()

        # Bucket into nearest DAiSEE level + a distance-based pseudo-confidence.
        diffs = [abs(score - v) for v in LEVEL_VALUES]
        best_idx = int(np.argmin(diffs))
        level = LEVEL_LABELS[best_idx]

        # Pseudo-probabilities: inverse-distance softmax over the 4 levels
        # (NOT a calibrated classifier — the underlying model is a
        # regressor; this is purely for the UI's probability bars).
        inv = [1.0 / (d + 1e-3) for d in diffs]
        total = sum(inv)
        probs = [v / total for v in inv]
        confidence = probs[best_idx]

        return {
            "engagement_score": score,
            "engagement_level": level,
            "confidence": confidence,
            "level_probabilities": [
                {"label": LEVEL_LABELS[i], "p": probs[i]} for i in range(len(LEVEL_LABELS))
            ],
            "num_frames_used": clip.shape[1],
            "model_status": self.model_status,
        }

    def predict_from_video_bytes(self, video_bytes: bytes, target_fps: int = 1):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        try:
            cap = cv2.VideoCapture(tmp_path)
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            interval = max(1, round(native_fps / target_fps))
            frames = []
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % interval == 0:
                    frames.append(frame)
                idx += 1
            cap.release()
        finally:
            os.unlink(tmp_path)
        return self.predict_from_frames(frames)

    def predict_from_image_bytes(self, image_bytes: bytes):
        """Single-image fallback: replicate the frame across the clip
        window (documented limitation — no real temporal signal)."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
        return self.predict_from_frames([frame] * self.clip_len)


_service_singleton: MLatteService = None


def get_service() -> MLatteService:
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = MLatteService()
    return _service_singleton
