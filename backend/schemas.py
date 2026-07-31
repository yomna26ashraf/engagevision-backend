"""Pydantic response/request schemas for the EngageVision API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

# DAiSEE-aligned engagement levels (matches src/data/daisee_dataset.py mapping)
LEVEL_LABELS = ["Not Engaged", "Barely Engaged", "Engaged", "Highly Engaged"]
LEVEL_VALUES = [0.0, 0.25, 0.5, 1.0]


class LevelProbability(BaseModel):
    label: str
    p: float


class PredictionResponse(BaseModel):
    engagement_score: float = Field(..., description="Continuous score in [0, 1]")
    engagement_level: str = Field(..., description="Nearest DAiSEE-style bucket")
    confidence: float = Field(..., description="Confidence in the bucketed level, in [0, 1]")
    level_probabilities: List[LevelProbability]
    num_frames_used: int
    model_status: str = Field(..., description="'trained' or 'untrained_demo'")


class EpochMetric(BaseModel):
    epoch: int
    train_mse: Optional[float] = None
    val_mse: Optional[float] = None
    val_mae: Optional[float] = None
    val_acc: Optional[float] = None


class PerformanceResponse(BaseModel):
    model_status: str
    epochs: List[EpochMetric]
    confusion_matrix: Optional[List[List[int]]] = None
    class_labels: List[str] = LEVEL_LABELS
    per_class_metrics: Optional[List[dict]] = None
    final_test: Optional[dict] = None
    paper_reference_accuracy: float = 0.6137


class HistoryItem(BaseModel):
    id: str
    filename: str
    label: str
    confidence: float
    score: float
    timestamp: str


class DistributionItem(BaseModel):
    label: str
    count: int


class DashboardResponse(BaseModel):
    model_status: str
    total_predictions: int
    avg_engagement_score: Optional[float] = None
    distribution: List[DistributionItem]
    history: List[HistoryItem]
