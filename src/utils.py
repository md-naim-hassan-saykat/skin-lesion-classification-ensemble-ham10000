"""Utility helpers for training/evaluation.

- compute_metrics: accuracy / weighted F1 / (optional) ROC-AUC OvR
- seed_everything: reproducibility across NumPy / PyTorch / cuDNN
- load_yaml / save_json: simple IO helpers
"""
from __future__ import annotations
import os
import json
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
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
    import numpy as _np
    import torch as _torch

    random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False


def save_json(d: Dict, path: os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def load_yaml(path: os.PathLike) -> Dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)
