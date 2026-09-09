from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

CANONICAL_CLASSES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Maximum-confidence multiclass Expected Calibration Error using equal-width bins.
    """

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    confidence = np.max(y_prob, axis=1)
    prediction = np.argmax(y_prob, axis=1)
    correct = (prediction == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidence >= edges[i]) & (confidence <= edges[i + 1])
        else:
            mask = (confidence >= edges[i]) & (confidence < edges[i + 1])

        if not np.any(mask):
            continue

        bin_accuracy = correct[mask].mean()
        bin_confidence = confidence[mask].mean()
        ece += mask.mean() * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_metrics(
    y_true,
    y_pred=None,
    y_prob=None,
    dataset: str = "ham10000",
    ece_bins: int = 15,
) -> dict[str, float | None]:
    """
    Compute manuscript-aligned metrics.

    HAM10000:
      accuracy
      weighted F1
      macro OvR ROC-AUC
      micro OvR ROC-AUC
      ECE

    ISIC 2019:
      accuracy
      weighted F1
      macro OvR ROC-AUC
      weighted OvR ROC-AUC
    """

    y_true = np.asarray(y_true, dtype=int)

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)

        if y_prob.ndim != 2 or y_prob.shape[1] != 7:
            raise ValueError(f"Expected probability matrix of shape (N, 7), got {y_prob.shape}")

        if y_pred is None:
            y_pred = np.argmax(y_prob, axis=1)

    if y_pred is None:
        raise ValueError("Either y_pred or y_prob must be provided.")

    y_pred = np.asarray(y_pred, dtype=int)

    result: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    if y_prob is None:
        return result

    labels = np.arange(y_prob.shape[1])

    try:
        result["macro_roc_auc_ovr"] = float(
            roc_auc_score(
                y_true,
                y_prob,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        result["macro_roc_auc_ovr"] = None

    ds = dataset.lower().replace("-", "").replace("_", "")

    if ds in {"ham10000", "ham"}:
        try:
            binary = label_binarize(
                y_true,
                classes=labels,
            )

            result["micro_roc_auc_ovr"] = float(
                roc_auc_score(
                    binary,
                    y_prob,
                    average="micro",
                )
            )
        except ValueError:
            result["micro_roc_auc_ovr"] = None

        result["ece"] = expected_calibration_error(
            y_true,
            y_prob,
            n_bins=ece_bins,
        )

    elif ds in {"isic2019", "isic"}:
        try:
            result["weighted_roc_auc_ovr"] = float(
                roc_auc_score(
                    y_true,
                    y_prob,
                    labels=labels,
                    multi_class="ovr",
                    average="weighted",
                )
            )
        except ValueError:
            result["weighted_roc_auc_ovr"] = None

    else:
        raise ValueError("dataset must be 'ham10000' or 'isic2019'")

    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
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
