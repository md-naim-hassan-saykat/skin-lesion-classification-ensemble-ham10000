# src/ensemble.py
from __future__ import annotations

import argparse
import csv
from typing import List, Tuple

import numpy as np

from src.utils import compute_metrics, save_json


def load_probs(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, probs) from a prediction CSV."""
    ys: List[int] = []
    probs: List[List[float]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        # assume columns: y_true, p_0, p_1, ... p_K
        for row in r:
            ys.append(int(row["y_true"]))
            row_probs = [float(v) for k, v in row.items() if k.startswith("p_")]
            probs.append(row_probs)
    return np.array(ys), np.array(probs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True, help="List of per-model prediction CSVs.")
    ap.add_argument("--out", required=True, help="Path to write ensemble metrics JSON.")
    args = ap.parse_args()

    y_true_ref: np.ndarray | None = None
    prob_stack: List[np.ndarray] = []

    for path in args.csvs:
        y_true, probs = load_probs(path)
        if y_true_ref is None:
            y_true_ref = y_true
        else:
            if not np.array_equal(y_true_ref, y_true):
                raise ValueError(f"y_true mismatch across CSVs: {path}")
        prob_stack.append(probs)

    assert y_true_ref is not None
    mean_probs = np.mean(prob_stack, axis=0)
    y_pred = mean_probs.argmax(axis=1)

    metrics = compute_metrics(y_true_ref, y_pred, y_prob=mean_probs)
    save_json(metrics, args.out)


if __name__ == "__main__":
    main()
