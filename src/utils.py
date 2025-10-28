"""Utility helpers for training/evaluation.

- compute_metrics: accuracy / weighted F1 / (optional) ROC-AUC OvR
- seed_everything: reproducibility across NumPy / PyTorch / cuDNN
- load_yaml / save_json: simple IO helpers
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_metrics(y_true, y_pred, y_prob=None) -> dict[str, float | None]:
    """Compute accuracy, weighted F1, and optional multi-class ROC-AUC."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except Exception:
            auc = None
    return {"accuracy": acc, "f1": f1, "auc": auc}


def seed_everything(seed: int = 42) -> None:
    """Ensure reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(d: dict, path: str | Path) -> None:
    """Save dictionary as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2))


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file into a Python dictionary."""
    import yaml  # local import to keep base deps light

    return yaml.safe_load(Path(path).read_text())
