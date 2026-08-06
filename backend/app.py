"""
EngageVision AI backend — FastAPI service wrapping the M-LATTE DAiSEE
pipeline for the frontend (TanStack Start app).

Run locally:
    cd backend
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000

The frontend expects this at http://localhost:8000 by default (see
src/lib/api.ts on the frontend side — VITE_API_BASE_URL overrides it).

"""
from __future__ import annotations
from contextlib import asynccontextmanager


from collections import Counter
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    from . import storage
    from .service_factory import get_service
    from .schemas import (
        DashboardResponse, DistributionItem, EpochMetric, HistoryItem,
        LEVEL_LABELS, PerformanceResponse, PredictionResponse,
    )
except ImportError:
    # Running as a top-level script (`uvicorn app:app` from inside backend/)
    import storage
    from service_factory import get_service
    from schemas import (
        DashboardResponse, DistributionItem, EpochMetric, HistoryItem,
        LEVEL_LABELS, PerformanceResponse, PredictionResponse,
    )

import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # يُنفذ هذا الجزء فور بدء تشغيل السيرفر وقبل استقبال أي طلبات
    print("Loading model on startup...")
    get_service()  # استدعاء دالة التحميل مسبقاً
    print("Model loaded successfully!")
    
    yield  # يتوقف هنا أثناء عمل التطبيق
    
    # يمكن إضافه أي عملية تنظيف (Clean up) هنا عند إيقاف السيرفر إن وجدت

# يمرر الـ lifespan للتطبيق عند إنشائه
app = FastAPI(title="EngageVision AI API", version="0.1.0", lifespan=lifespan)

# In dev this defaults to "*" (any origin). For a deployed portfolio demo,
# set ALLOWED_ORIGINS to your frontend's exact URL(s), comma-separated,
# e.g. ALLOWED_ORIGINS="https://your-site.pages.dev,https://your-site.com"
_allowed = os.environ.get("ALLOWED_ORIGINS", "*")
allowed_origins = ["*"] if _allowed == "*" else [o.strip() for o in _allowed.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.api_route("/api/health", methods=["GET", "HEAD"])
def health():
    service = get_service()
    return {"status": "ok", "model_status": service.model_status, "device": str(service.device)}


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    service = get_service()
    content = await file.read()
    content_type = (file.content_type or "").lower()

    try:
        if content_type.startswith("video/"):
            result = service.predict_from_video_bytes(content)
        elif content_type.startswith("image/"):
            result = service.predict_from_image_bytes(content)
        else:
            # best-effort: try video decode first, fall back to image
            try:
                result = service.predict_from_video_bytes(content)
            except Exception:
                result = service.predict_from_image_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process file: {e}")

    storage.log_prediction(file.filename or "upload", result)
    return result


@app.get("/api/performance", response_model=PerformanceResponse)
def performance():
    service = get_service()
    results = storage.read_training_results()

    if results is None:
        # No training run finished yet — return an honest empty shell
        # rather than fabricated curves.
        return PerformanceResponse(
            model_status="untrained_demo",
            epochs=[],
            confusion_matrix=None,
            per_class_metrics=None,
            final_test=None,
        )

    return PerformanceResponse(
        model_status=service.model_status,
        epochs=[EpochMetric(**e) for e in results.get("epochs", [])],
        confusion_matrix=results.get("confusion_matrix"),
        per_class_metrics=results.get("per_class_metrics"),
        final_test=results.get("final_test"),
    )


@app.get("/api/dashboard", response_model=DashboardResponse)
def dashboard(limit: int = 50):
    service = get_service()
    entries = storage.read_predictions(limit=limit)

    counts = Counter(e["label"] for e in entries)
    distribution = [DistributionItem(label=lbl, count=counts.get(lbl, 0)) for lbl in LEVEL_LABELS]

    avg_score: Optional[float] = None
    if entries:
        avg_score = sum(e["score"] for e in entries) / len(entries)

    history = [
        HistoryItem(
            id=e["id"], filename=e["filename"], label=e["label"],
            confidence=e["confidence"], score=e["score"], timestamp=e["timestamp"],
        )
        for e in entries
    ]

    return DashboardResponse(
        model_status=service.model_status,
        total_predictions=len(entries),
        avg_engagement_score=avg_score,
        distribution=distribution,
        history=history,
    )
