"""
ONNX Runtime-backed version of MLatteService — same predict_from_frames /
predict_from_video_bytes / predict_from_image_bytes interface as
model_service.py, but runs inference through onnxruntime instead of full
PyTorch. onnxruntime's CPU footprint is substantially smaller than
torch+torchvision (no autograd engine, leaner runtime), which is the
whole point: this is the deployment-memory fix for hosts too small for
the full PyTorch model_service.py (e.g. Render's free 512MB tier).

Requires an exported .onnx file — see scripts/export_onnx.py — and that
you've already run scripts/validate_onnx.py to confirm it matches the
original PyTorch model's predictions.

HONEST CAVEAT: this still imports `torch` for two small, non-model
things — stacking frame tensors and the pad/truncate helper — because
our preprocessing pipeline (src/data/preprocessing.py) is torchvision-
based. So this does NOT eliminate the PyTorch dependency entirely; it
eliminates it for the *heavy* part (the dual-ResNet-50 + Transformer VAE
forward pass and its autograd bookkeeping), which is where almost all of
the memory actually goes. If you need to shave off every remaining MB,
the next step would be rewriting default_face_transform/
pad_or_truncate_frames in plain NumPy/Pillow — a separate, smaller task.

Enable this backend by setting MLATTE_USE_ONNX=1 (see backend/app.py).
"""
from __future__ import annotations

import os
import sys
from typing import List

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.preprocessing import default_face_transform, pad_or_truncate_frames  # noqa: E402

try:
    from .schemas import LEVEL_LABELS, LEVEL_VALUES
except ImportError:
    from schemas import LEVEL_LABELS, LEVEL_VALUES

DEFAULT_ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "daisee_mlatte.onnx")
DEFAULT_CLIP_LEN = int(os.environ.get("MLATTE_CLIP_LEN", "10"))  # must match the exported graph


def _maybe_download_onnx(onnx_path: str):
    """Same pattern as model_service.py's checkpoint auto-download: if the
    .onnx file isn't present locally but MLATTE_ONNX_URL is set (e.g. a
    direct download link from the same Hugging Face model repo you
    uploaded the .pt checkpoint to), fetch it once at startup."""
    if os.path.exists(onnx_path):
        return
    url = os.environ.get("MLATTE_ONNX_URL")
    if not url:
        return
    import urllib.request
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    print(f"Downloading ONNX model from {url} -> {onnx_path} ...")
    urllib.request.urlretrieve(url, onnx_path)
    print("ONNX model download complete.")


class MLatteOnnxService:
    def __init__(self, onnx_path: str = None):
        import onnxruntime as ort

        onnx_path = onnx_path or os.environ.get("MLATTE_ONNX_PATH", DEFAULT_ONNX_PATH)
        _maybe_download_onnx(onnx_path)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"No ONNX model found at {onnx_path}, and no MLATTE_ONNX_URL was set to "
                f"download one. Run scripts/export_onnx.py first, then "
                f"scripts/validate_onnx.py to confirm it's correct, before enabling "
                f"MLATTE_USE_ONNX."
            )

        # int8 dynamic quantization gives onnxruntime a further memory/speed
        # win on top of the format switch itself, at effectively no
        # accuracy cost for CPU inference — cheap to try, easy to disable
        # (MLATTE_ONNX_QUANTIZE=0) if you ever see it hurt scores.
        if os.environ.get("MLATTE_ONNX_QUANTIZE", "1") == "1":
            onnx_path = self._maybe_quantize(onnx_path)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = int(os.environ.get("MLATTE_NUM_THREADS", "1"))
        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        self.model_status = "trained"  # an ONNX export only exists once you've trained + exported
        self.device = "cpu"  # onnxruntime CPUExecutionProvider — kept for parity with MLatteService's API
        self.clip_len = DEFAULT_CLIP_LEN
        self.transform = default_face_transform(image_size=224, train=False)

    @staticmethod
    def _maybe_quantize(onnx_path: str) -> str:
        quantized_path = onnx_path.replace(".onnx", ".quant.onnx")
        if os.path.exists(quantized_path):
            return quantized_path
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QInt8)
            print(f"[onnx] Quantized model written to {quantized_path}")
            return quantized_path
        except Exception as e:  # pragma: no cover - quantization is a best-effort optimization
            print(f"[onnx] Dynamic quantization failed ({e}); using unquantized model.")
            return onnx_path

    def predict_from_frames(self, frames_bgr: List[np.ndarray]):
        if not frames_bgr:
            raise ValueError("No frames provided")

        tensors = []
        for f in frames_bgr:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(rgb))
        import torch  # only for the same padding helper + tensor->numpy convenience
        clip = torch.stack(tensors, dim=0)
        clip = pad_or_truncate_frames(clip, self.clip_len)
        clip = clip.unsqueeze(0).numpy().astype(np.float32)  # (1, T, C, H, W)

        (score_arr,) = self.session.run(["score"], {"frames": clip})
        score = float(score_arr.reshape(-1)[0])

        diffs = [abs(score - v) for v in LEVEL_VALUES]
        best_idx = int(np.argmin(diffs))
        level = LEVEL_LABELS[best_idx]

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
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
        return self.predict_from_frames([frame] * self.clip_len)
