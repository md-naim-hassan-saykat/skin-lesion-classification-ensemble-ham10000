# src/ensemble.py
import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


def _is_int_like(x: str, num_classes: int) -> bool:
    """Return True if x is an integer in [0, num_classes-1] or a float extremely close to such an int."""
    try:
        # exact int string
        if re.fullmatch(r"\d+", x.strip()):
            v = int(x)
            return 0 <= v < num_classes
        # float close to int
        v = float(x)
        if not (0 <= v < num_classes):
            return False
        r = round(v)
        return abs(v - r) < 1e-6
    except Exception:
        return False


def _parse_label(x: str) -> int:
    """Parse a label previously validated by _is_int_like (standard 0.5 rounds up)."""
    v = float(x)
    return int(math.floor(v + 0.5))


def _find_prob_indices(header: list[str], num_classes: int) -> list[int]:
    """Prefer named p_0..p_{C-1}; otherwise, fall back to the last C columns."""
    lower = [h.strip().lower() for h in header]
    p_idx = []
    for i in range(num_classes):
        name = f"p_{i}"
        if name in lower:
            p_idx.append(lower.index(name))
        else:
            p_idx = []
            break
    if p_idx:
        return p_idx
    # fallback: last C columns
    if len(header) >= num_classes:
        return list(range(len(header) - num_classes, len(header)))
    return []


# fmt: off
def read_probs_and_labels(path: str, num_classes: int) -> tuple[np.ndarray | None, np.ndarray]:  # noqa: C901
# fmt: on

    """
    Read a CSV of per-sample probabilities (and an optional integer label column).
    - Prob columns: prefer named p_0..p_{C-1}; else the last C columns.
    - Label column: use 'y_true' if present; fall back to 'true' ONLY if values are integer-like in [0..C-1].
      Otherwise ignore the column (labels=None).
    Returns:
      (Y, P) where Y is np.ndarray[int] of shape (N,) or None, and P is np.ndarray[float] of shape (N, C).
    """
    Y: list[int] = []
    P: list[list[float]] = []

    with open(path, newline="") as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration:
            return None, np.zeros((0, num_classes), dtype=float)

        lower = [h.strip().lower() for h in header]
        # Probabilities
        prob_idx = _find_prob_indices(header, num_classes)
        if len(prob_idx) != num_classes:
            raise ValueError(f"{path}: could not locate {num_classes} probability columns.")

        # Label column detection (strict)
        label_idx = None
        if "y_true" in lower:
            label_idx = lower.index("y_true")
        elif "true" in lower:
            # We'll tentatively allow 'true' only if its values are integer-like.
            label_idx = lower.index("true")

        have_valid_labels = True if label_idx is not None else False

        for row in r:
            if not row or not any(cell.strip() for cell in row):
                continue
            try:
                probs = [float(row[i]) for i in prob_idx]
            except Exception:
                # skip bad row
                continue
            if len(probs) != num_classes or any(math.isnan(x) for x in probs):
                continue
            P.append(probs)

            if label_idx is not None and label_idx < len(row):
                lab = row[label_idx].strip()
                if _is_int_like(lab, num_classes):
                    Y.append(_parse_label(lab))
                else:
                    have_valid_labels = False
            else:
                have_valid_labels = False

    P_arr = np.asarray(P, dtype=float)
    if P_arr.ndim != 2 or P_arr.shape[1] != num_classes:
        raise ValueError(f"{path}: probabilities array malformed; got shape {P_arr.shape}")

    if have_valid_labels and len(Y) == len(P_arr):
        return np.asarray(Y, dtype=int), P_arr
    else:
        return None, P_arr


def main():
    ap = argparse.ArgumentParser(
        description="Average probabilities across multiple model CSVs and (optionally) compute metrics."
    )
    ap.add_argument("--csvs", nargs="+", required=True, help="List of per-model CSVs.")
    ap.add_argument("--out", required=True, help="Path to output JSON metrics file.")
    ap.add_argument("--num_classes", type=int, default=7)
    args = ap.parse_args()

    outputs = [read_probs_and_labels(p, args.num_classes) for p in args.csvs]
    Ys, Ps = zip(*outputs)

    # Align by truncating to the smallest N
    n = min((P.shape[0] for P in Ps), default=0)
    Ps = [P[:n] for P in Ps]

    if n == 0:
        out_json = {
            "samples": 0,
            "num_classes": int(args.num_classes),
            "has_ground_truth": False,
            "message": "No valid probability rows found.",
        }
        Path(args.out).write_text(json.dumps(out_json, indent=2))
        print(f"Wrote {args.out} (empty)")
        return

    # Average probabilities across models: shape (N, C)
    P = np.mean(np.stack(Ps, axis=0), axis=0)
    y_pred = P.argmax(axis=1)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always write ensemble probabilities and predictions
    probs_csv = out_dir / "ensemble_probs.csv"
    preds_csv = out_dir / "ensemble_preds.csv"
    with open(probs_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"p_{i}" for i in range(args.num_classes)])
        for row in P:
            w.writerow([f"{x:.8f}" for x in row])
    with open(preds_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y_pred"])
        for yp in y_pred:
            w.writerow([int(yp)])

    # Ground truth (use the first CSV that provides VALID labels for all N rows)
    Y = next((Y for Y in Ys if Y is not None and len(Y) >= n), None)
    if Y is not None and len(Y) > n:
        Y = Y[:n]

    out_json = {
        "samples": int(n),
        "num_classes": int(args.num_classes),
        "has_ground_truth": bool(Y is not None),
    }

    if Y is not None:
        acc = float((y_pred == Y).mean())
        out_json["accuracy"] = acc
        try:
            from sklearn.metrics import (
                classification_report,
                confusion_matrix,
                f1_score,
                precision_score,
                recall_score,
            )
            out_json["macro_f1"] = float(f1_score(Y, y_pred, average="macro", zero_division=0))
            out_json["micro_f1"] = float(f1_score(Y, y_pred, average="micro", zero_division=0))
            out_json["macro_precision"] = float(
                precision_score(Y, y_pred, average="macro", zero_division=0)
            )
            out_json["macro_recall"] = float(
                recall_score(Y, y_pred, average="macro", zero_division=0)
            )
            out_json["classification_report"] = classification_report(
                Y, y_pred, output_dict=True, zero_division=0
            )
            out_json["confusion_matrix"] = confusion_matrix(Y, y_pred).tolist()
        except Exception:
            out_json["message"] = "Metrics computation failed (missing sklearn)."
    else:
        out_json["message"] = (
            "Averaged probabilities across CSVs (no valid integer label column detected). "
            "Regenerate per-model CSVs with header 'y_true' containing integer class IDs."
        )

    Path(args.out).write_text(json.dumps(out_json, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
