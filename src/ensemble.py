# src/ensemble.py
from __future__ import annotations

# ruff: noqa: E402  # allow imports after the path shim

# --- path shim (lets `python src/xyz.py` import `src.*`) ---
import sys
from pathlib import Path as _P

_PROJECT_ROOT = _P(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# -----------------------------------------------------------

# stdlib / third-party
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
        if re.fullmatch(r"\d+", x.strip()):
            v = int(x)
            return 0 <= v < num_classes
        v = float(x)
        if not (0 <= v < num_classes):
            return False
        return abs(v - round(v)) < 1e-6
    except Exception:
        return False


def _parse_label(x: str) -> int:
    v = float(x)
    return int(math.floor(v + 0.5))


def _find_prob_indices(header: list[str], num_classes: int) -> list[int]:
    lower = [h.strip().lower() for h in header]
    p_idx: list[int] = []
    for i in range(num_classes):
        name = f"p_{i}"
        if name in lower:
            p_idx.append(lower.index(name))
        else:
            p_idx = []
            break
    if p_idx:
        return p_idx
    if len(header) >= num_classes:
        return list(range(len(header) - num_classes, len(header)))
    return []


def read_probs_and_labels(path: str, num_classes: int) -> tuple[np.ndarray | None, np.ndarray]:
    """Return (Y, P) where Y may be None; P is (N, C) float array."""
    Y: list[int] = []
    P: list[list[float]] = []

    with open(path, newline="") as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration:
            return None, np.zeros((0, num_classes), dtype=float)

        lower = [h.strip().lower() for h in header]
        prob_idx = _find_prob_indices(header, num_classes)
        if len(prob_idx) != num_classes:
            raise ValueError(f"{path}: could not locate {num_classes} probability columns.")

        label_idx = None
        if "y_true" in lower:
            label_idx = lower.index("y_true")
        elif "true" in lower:
            label_idx = lower.index("true")

        have_valid_labels = label_idx is not None

        for row in r:
            if not row or not any(cell.strip() for cell in row):
                continue
            try:
                probs = [float(row[i]) for i in prob_idx]
            except Exception:
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
    return None, P_arr


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Average probabilities across multiple model CSVs and (optionally) compute metrics."
    )
    ap.add_argument("--csvs", nargs="+", required=True, help="List of per-model CSVs.")
    ap.add_argument("--out", required=True, help="Path to output JSON metrics file.")
    ap.add_argument("--num_classes", type=int, default=7)
    args = ap.parse_args()

    outputs = [read_probs_and_labels(p, args.num_classes) for p in args.csvs]
    Ys, Ps = zip(*outputs, strict=False)

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

    P = np.mean(np.stack(Ps, axis=0), axis=0)  # (N, C)
    y_pred = P.argmax(axis=1)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # write ensemble outputs
    with open(out_dir / "ensemble_probs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"p_{i}" for i in range(args.num_classes)])
        for row in P:
            w.writerow([f"{x:.8f}" for x in row])
    with open(out_dir / "ensemble_preds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y_pred"])
        for yp in y_pred:
            w.writerow([int(yp)])

    # pick first CSV that had valid labels for all rows
    Y = next((Y for Y in Ys if Y is not None and len(Y) >= n), None)
    if Y is not None and len(Y) > n:
        Y = Y[:n]

    out = {
        "samples": int(n),
        "num_classes": int(args.num_classes),
        "has_ground_truth": bool(Y is not None),
    }
    if Y is not None:
        out["accuracy"] = float((y_pred == Y).mean())
    else:
        out["message"] = (
            "Averaged probabilities across CSVs (no valid integer label column detected). "
            "Regenerate per-model CSVs with header 'y_true' containing integer class IDs."
        )

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
