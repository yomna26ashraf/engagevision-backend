"""
Lightweight service dispatcher.

This module intentionally has NO heavy top-level imports (no torch, no cv2,
no onnxruntime). It only decides, at call time, which concrete service class
to import and instantiate, based on the MLATTE_USE_ONNX environment
variable. This keeps `import torch` (pulled in by model_service.py) from
ever executing when MLATTE_USE_ONNX=1 — the whole point of this file.
"""
from __future__ import annotations

import os

_service_singleton = None


def get_service():
    """Returns the running inference service — either the full-PyTorch
    MLatteService (default) or the leaner MLatteOnnxService, selected via
    the MLATTE_USE_ONNX=1 environment variable.

    Crucially, the *import* of model_service.py (which pulls in torch) or
    onnx_model_service.py (which pulls in onnxruntime) only happens inside
    the relevant branch below — never both, and never at module load time.
    """
    global _service_singleton
    if _service_singleton is None:
        if os.environ.get("MLATTE_USE_ONNX", "0") == "1":
            try:
                from .onnx_model_service import MLatteOnnxService
            except ImportError:
                from onnx_model_service import MLatteOnnxService
            _service_singleton = MLatteOnnxService()
        else:
            try:
                from .model_service import MLatteService
            except ImportError:
                from model_service import MLatteService
            _service_singleton = MLatteService()
    return _service_singleton
