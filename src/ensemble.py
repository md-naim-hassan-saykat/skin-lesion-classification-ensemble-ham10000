"""Average probabilities from multiple prediction CSVs (same rows/order).

Each CSV should have columns: index,true,pred,p_0,p_1,...,p_{K-1}

Usage:
  python src/ensemble.py --csvs path/a.csv path/b.csv path/c.csv --out ./outputs/ensemble_metrics.json
"""
from __future__ import annotations

import argparse
import csv
import numpy as np
from utils import compute_metrics, save_json


def read_csv(path: str):
    probs = []
    y_true = []
    y_pred = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        p_cols = [i for i, h in enumerate(header) if h.startswith("p_")]
        for row in reader:
            y_true.append(int(row[1]))
            y_pred.append(int(row[2]))
            probs.append([float(row[i]) for i in p_cols])
    return np.array(y_true), np.array(y_pred), np.array(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True, help="List of prediction CSVs to average") 
    ap.add_argument("--out", required=True, type=str, help="Where to write ensemble metrics JSON")
    args = ap.parse_args()

    y_trues, prob_list = [], []
    for i, p in enumerate(args.csvs):
        y_true, _, probs = read_csv(p)
        prob_list.append(probs)
        y_trues.append(y_true)

    # Sanity check: all y_true identical
    for i in range(1, len(y_trues)):
        if not np.array_equal(y_trues[0], y_trues[i]):
            raise ValueError("CSV ground-truth rows don't match. Ensure the same ordering.")

    y_true = y_trues[0]
    avg_probs = np.mean(np.stack(prob_list, axis=0), axis=0)
    y_pred = avg_probs.argmax(1)

    metrics = compute_metrics(y_true, y_pred, y_prob=avg_probs)
    save_json(metrics, args.out)
    print("Ensemble metrics:", metrics, "-> saved to", args.out)


if __name__ == "__main__":
    main()
