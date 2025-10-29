# src/ensemble.py
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read_probs_and_labels(path, num_classes):
    """
    Reads per-sample probabilities (and optional labels) from a CSV file.
    Compatible with headers like: y_true,p_0,p_1,p_2,...,p_6
    If no label column is found, returns Y=None.
    """
    Y, P = [], []

    with open(path, newline="") as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration:
            return None, np.zeros((0, num_classes), dtype=float)

        header_lower = [h.strip().lower() for h in header]
        label_idx = None
        for cand in ("y_true", "true"):
            if cand in header_lower:
                label_idx = header_lower.index(cand)
                break

        for row in r:
            if not row or not any(row):
                continue
            # last num_classes columns are probabilities
            probs = row[-num_classes:]
            try:
                probs = [float(x) for x in probs]
            except Exception:
                continue
            P.append(probs)

            if label_idx is not None and label_idx < len(row):
                try:
                    Y.append(int(float(row[label_idx])))
                except Exception:
                    Y.append(None)

    P = np.asarray(P, dtype=float)
    if P.ndim == 1 and P.size % num_classes == 0:
        P = P.reshape(-1, num_classes)

    # Clean label array
    if len(Y) == len(P) and all(y is not None for y in Y):
        Y = np.array(Y, dtype=int)
    else:
        Y = None

    return Y, P


def main():
    ap = argparse.ArgumentParser(
        description="Average probabilities across multiple model CSVs and compute ensemble metrics."
    )
    ap.add_argument("--csvs", nargs="+", required=True, help="List of per-model CSVs.")
    ap.add_argument("--out", required=True, help="Path to output JSON metrics file.")
    ap.add_argument("--num_classes", type=int, default=7)
    args = ap.parse_args()

    pairs = [read_probs_and_labels(p, args.num_classes) for p in args.csvs]
    Ys, Ps = zip(*pairs)

    # Trim to smallest length (in case CSVs differ)
    n = min((P.shape[0] for P in Ps), default=0)
    Ps = [P[:n] for P in Ps]

    if n == 0:
        out_json = {
            "samples": 0,
            "num_classes": args.num_classes,
            "has_ground_truth": False,
            "message": "No valid probability rows found."
        }
        Path(args.out).write_text(json.dumps(out_json, indent=2))
        print(f"Wrote {args.out} (empty)")
        return

    # Average probabilities across models
    P = np.mean(np.stack(Ps, axis=0), axis=0)  # shape [N, C]
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

    # Try to find ground truth labels
    Y = next((Y for Y in Ys if Y is not None and len(Y) == n), None)

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
                f1_score,
                precision_score,
                recall_score,
                confusion_matrix,
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
        out_json["message"] = "Averaged probabilities across CSVs (no label column detected)."

    Path(args.out).write_text(json.dumps(out_json, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
