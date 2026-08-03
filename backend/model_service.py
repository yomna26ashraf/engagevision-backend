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

import gc
import os
import sys
from typing import List

import cv2
import numpy as np
import torch
import yaml

# Keep PyTorch's CPU thread pool small — each thread carries its own
# working-memory overhead, which adds up on a memory-capped host (e.g. a
# 512MB deployment tier). Must be set before any tensor ops run.
torch.set_num_threads(int(os.environ.get("MLATTE_NUM_THREADS", "1")))

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

# Set MLATTE_LOW_MEMORY=1 (e.g. as a Render/Railway env var) to trade some
# speed and a little precision for a much smaller memory footprint:
#   - frames go through each CNN branch 2-at-a-time instead of all at once
#   - the VAE/regression Linear layers are dynamically int8-quantized
# Leave unset for normal (GPU/local) use — this is a deployment-only knob.
LOW_MEMORY_MODE = os.environ.get("MLATTE_LOW_MEMORY", "0") == "1"


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
            inference_chunk_size=2 if LOW_MEMORY_MODE else None,
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

        if LOW_MEMORY_MODE and self.device.type == "cpu":
            # Dynamic quantization (int8) of Linear layers only — safe for
            # the VAE/Transformer/regression-head parts of the model
            # (Conv2d layers in the ResNet branches aren't touched, since
            # dynamic quantization doesn't meaningfully help convolutions).
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("[low-memory mode] Linear layers dynamically quantized to int8; "
                  "CNN branches chunked 2 frames at a time.")

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

        result = {
            "engagement_score": score,
            "engagement_level": level,
            "confidence": confidence,
            "level_probabilities": [
                {"label": LEVEL_LABELS[i], "p": probs[i]} for i in range(len(LEVEL_LABELS))
            ],
            "num_frames_used": clip.shape[1],
            "model_status": self.model_status,
        }
        if LOW_MEMORY_MODE:
            del clip, out, tensors
            gc.collect()
        return result

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


_service_singleton = None


def get_service():
    """Returns the running inference service — either the full-PyTorch
    MLatteService (default) or the leaner MLatteOnnxService, selected via
    the MLATTE_USE_ONNX=1 environment variable. Only switch to ONNX after
    you've run scripts/export_onnx.py + scripts/validate_onnx.py and
    confirmed the export matches — see onnx_model_service.py."""
    global _service_singleton
    if _service_singleton is None:
        if os.environ.get("MLATTE_USE_ONNX", "0") == "1":
            try:
                from .onnx_model_service import MLatteOnnxService
            except ImportError:
                from onnx_model_service import MLatteOnnxService
            _service_singleton = MLatteOnnxService()
        else:
            _service_singleton = MLatteService()
    return _service_singleton
