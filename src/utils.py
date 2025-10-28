# src/utils.py
from __future__ import annotations

import json
import random

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float | None]:
    """Compute accuracy, weighted F1, and optional multi-class ROC-AUC."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    auc: float | None = None
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        except Exception:
            auc = None
    return {"accuracy": float(acc), "f1": float(f1), "auc": auc}


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
