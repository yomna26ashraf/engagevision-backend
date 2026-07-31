"""
Minimal file-based storage:
  - predictions.jsonl: one line per /predict call, powers the dashboard.
  - results.json: written by scripts/train_daisee.py at the end of
    training, powers the performance page (epoch curves, confusion
    matrix, final test metrics).

Good enough for a single-instructor / small-deployment demo; swap for a
real DB later if this grows into a multi-user product.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import List, Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.jsonl")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "results.json")

os.makedirs(DATA_DIR, exist_ok=True)


def log_prediction(filename: str, result: dict):
    entry = {
        "id": f"PRD-{int(datetime.now().timestamp() * 1000) % 1_000_000}",
        "filename": filename,
        "label": result["engagement_level"],
        "score": result["engagement_score"],
        "confidence": result["confidence"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(PREDICTIONS_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_predictions(limit: int = 50) -> List[dict]:
    if not os.path.exists(PREDICTIONS_PATH):
        return []
    with open(PREDICTIONS_PATH) as f:
        lines = f.readlines()
    entries = [json.loads(line) for line in lines[-limit:]]
    entries.reverse()  # most recent first
    return entries


def read_training_results() -> Optional[dict]:
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)


def write_training_results(results: dict):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
