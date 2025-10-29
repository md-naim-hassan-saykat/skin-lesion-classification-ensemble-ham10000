# src/ensemble.py
import argparse, csv, json
from pathlib import Path
import numpy as np

def read_probs_and_labels(path, num_classes):
    """
    Lenient reader:
    - If a label column exists ('y_true' or 'true'), use it.
    - Otherwise, return labels=None.
    - For each row, probabilities are taken from the LAST num_classes columns.
    """
    Y = []
    P = []
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
            if not row or not any(c.strip() for c in row):
                continue
            # take last C columns as probabilities
            cols = row[-num_classes:]
            if len(cols) != num_classes:
                continue
            try:
                probs = [float(x) for x in cols]
            except Exception:
                continue
            P.append(probs)

            if label_idx is not None and label_idx < len(row):
                try:
                    Y.append(int(float(row[label_idx])))  # cope with "3" vs "3.0"
                except Exception:
                    Y.append(None)

    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        if P.size % num_classes == 0:
            P = P.reshape(-1, num_classes)
        else:
            P = np.zeros((0, num_classes), dtype=float)

    Y = np.array(Y, dtype=object) if Y else None
    # If labels exist but contain None, drop them (treat as no labels)
    if Y is not None and (len(Y) != len(P) or any(y is None for y in Y)):
        Y = None
    return Y, P

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_classes", type=int, default=7)
    args = ap.parse_args()

    pairs = [read_probs_and_labels(p, args.num_classes) for p in args.csvs]
    Ys, Ps = zip(*pairs)

    # Trim all to same length
    n = min((P.shape[0] for P in Ps), default=0)
    Ps = [P[:n] for P in Ps]
    if n == 0:
        out_dir = Path(args.out).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "samples": 0,
            "num_classes": int(args.num_classes),
            "has_ground_truth": False,
            "message": "No parseable rows; ensure last C columns are probabilities."
        }, indent=2))
        print(f"Wrote {args.out} (empty)")
        return

    # Average probs across models
    P = np.mean(np.stack(Ps, axis=0), axis=0)  # [N, C]
    y_pred = P.argmax(axis=1)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ensemble_probs.csv").write_text(
        "p_" + ",p_".join(str(i) for i in range(args.num_classes)) + "\n" +
        "\n".join(",".join(f"{x:.8f}" for x in row) for row in P)
    )
    (out_dir / "ensemble_preds.csv").write_text(
        "y_pred\n" + "\n".join(str(int(x)) for x in y_pred)
    )

    # If any input had labels (and shapes match), compute metrics
    Y = next((Y for Y in Ys if Y is not None and len(Y) == n), None)
    out_json = {
        "samples": int(n),
        "num_classes": int(args.num_classes),
        "has_ground_truth": bool(Y is not None),
    }
    if Y is not None:
        Y = np.asarray(Y, dtype=int)
        acc = float((y_pred == Y).mean())
        out_json["accuracy"] = acc
        try:
            from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
            out_json["macro_f1"] = float(f1_score(Y, y_pred, average="macro", zero_division=0))
            out_json["micro_f1"] = float(f1_score(Y, y_pred, average="micro", zero_division=0))
            out_json["macro_precision"] = float(precision_score(Y, y_pred, average="macro", zero_division=0))
            out_json["macro_recall"] = float(recall_score(Y, y_pred, average="macro", zero_division=0))
            out_json["classification_report"] = classification_report(Y, y_pred, output_dict=True, zero_division=0)
            out_json["confusion_matrix"] = confusion_matrix(Y, y_pred).tolist()
        except Exception:
            pass
    else:
        out_json["message"] = "Averaged probabilities across CSVs (no label column detected)."

    Path(args.out).write_text(json.dumps(out_json, indent=2))
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
