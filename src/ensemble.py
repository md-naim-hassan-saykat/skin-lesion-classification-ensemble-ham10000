"""Average probabilities from multiple prediction CSVs (same rows/order).

Each CSV should have columns: index,true,pred,p_0,p_1,...,p_{K-1}

Usage:
  python src/ensemble.py --csvs path/a.csv path/b.csv path/c.csv --out ./outputs/ensemble_metrics.json
"""
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
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        prob_idx = [i for i, h in enumerate(header) if h.startswith("p_")]
        true_idx = header.index("true") if "true" in header else 2
        for row in reader:
            ys.append(int(row[true_idx]))
            probs.append([float(row[i]) for i in prob_idx])
    return np.array(ys), np.array(probs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ensemble prediction CSVs by averaging probabilities.")
    ap.add_argument("--csvs", nargs="+", required=True, help="List of prediction CSVs.")
    ap.add_argument("--out", required=True, help="Where to save ensemble metrics JSON.")
    args = ap.parse_args()

    y0, p0 = load_probs(args.csvs[0])
    probs_sum = p0.astype(np.float64)
    y_true = y0

    for path in args.csvs[1:]:
        yi, pi = load_probs(path)
        if yi.shape != y_true.shape or not np.array_equal(yi, y_true):
            raise ValueError(f"True labels mismatch between {args.csvs[0]} and {path}.")
        if pi.shape != probs_sum.shape:
            raise ValueError(f"Probability shape mismatch: {probs_sum.shape} vs {pi.shape} for {path}")
        probs_sum += pi

    probs_avg = probs_sum / len(args.csvs)
    y_pred = probs_avg.argmax(axis=1)

    metrics = compute_metrics(y_true, y_pred, y_prob=probs_avg)
    save_json(metrics, args.out)
    print("Ensemble metrics saved to:", args.out, metrics)


if __name__ == "__main__":
    main()
