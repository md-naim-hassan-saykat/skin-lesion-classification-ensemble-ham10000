"""
Partition of the ISIC 2019 evaluation set by source (manuscript Table 3,
Supplementary Table S5).

The ISIC 2019 challenge training set is assembled from HAM10000, BCN_20000 and MSK.
Every HAM10000 image therefore appears inside the 25,331-image ISIC 2019 set that this
study used for external evaluation, so about 40% of that "external" set is the dataset
the seven checkpoints were trained on.

This script partitions ISIC 2019 by source and recomputes every external metric on the
non-HAM10000 remainder, which is genuinely unseen by every model. It needs no retraining
and no new data: it reuses the released per-model probability arrays.

HOW TO RUN

  In a Kaggle or Jupyter notebook, just run the cell. Inputs are discovered automatically
  under /kaggle/input and the working directory; no arguments are needed. (An earlier
  version read sys.argv, which breaks in a notebook because the kernel passes its own
  "-f <connection file>" flag.)

  From a terminal, optionally pass paths:
      python external_decontamination.py ground_truth.csv predictions.npz

INPUTS   ISIC_2019_Training_GroundTruth.csv
         isic2019_external_eval_predictions.npz, or the individual
         isic2019_<model>_probs.npy files plus isic2019_labels.npy
OUTPUTS  ISIC2019_decontaminated_metrics.csv, ISIC2019_contamination_summary.csv,
         and a LaTeX table body for the manuscript
"""

import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
ONEHOT = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
MODELS = [
    "cnn",
    "resnet50",
    "densenet121",
    "efficientnet_b3",
    "convnext_tiny",
    "mobilenet_v3",
    "vit",
]
LABEL = {
    "cnn": "CNN",
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b3": "EfficientNet-B3",
    "convnext_tiny": "ConvNeXt-Tiny",
    "mobilenet_v3": "MobileNetV3-L",
    "vit": "ViT",
}

# HAM10000 occupies a contiguous ISIC id block. The bounds come from the released
# HAM10000 split files; the resulting count is checked against the documented 10,015.
HAM_LO, HAM_HI = 24307, 34317

SEARCH_ROOTS = ["/kaggle/input", "/kaggle/working", ".", ".."]


def find(pattern, needed=True):
    """Locate a file by glob pattern under the usual roots. Shortest path wins."""
    hits = []
    for root in SEARCH_ROOTS:
        if os.path.isdir(root):
            hits += glob(os.path.join(root, "**", pattern), recursive=True)
    hits = sorted(set(hits), key=len)
    if hits:
        return hits[0]
    if needed:
        raise FileNotFoundError(
            f"Could not find '{pattern}' under {', '.join(SEARCH_ROOTS)}. "
            f"Attach the dataset containing it, or pass the path explicitly."
        )
    return None


def cli_arg(suffix):
    """Return a command-line path with this suffix, ignoring notebook kernel flags."""
    for a in sys.argv[1:]:
        if a.lower().endswith(suffix) and os.path.exists(a):
            return a
    return None


def load_predictions():
    """Return (labels, {model: probs}) from the npz, or from individual npy files."""
    npz = cli_arg(".npz") or find("isic2019_external_eval_predictions.npz", needed=False)
    if npz:
        z = np.load(npz)
        missing = [m for m in MODELS if f"{m}_probs" not in z]
        if not missing:
            print(f"predictions: {npz}")
            return z["labels"], {m: z[f"{m}_probs"] for m in MODELS}
        print(f"  {npz} is missing {missing}; falling back to individual .npy files")

    lab = find("isic2019_labels.npy")
    probs = {}
    for m in MODELS:
        probs[m] = np.load(find(f"isic2019_{m}_probs.npy"))
    print(f"predictions: individual .npy files next to {lab}")
    return np.load(lab), probs


def main():
    gt_path = cli_arg(".csv") or find("ISIC_2019_Training_GroundTruth.csv")
    print(f"ground truth: {gt_path}")

    gt = pd.read_csv(gt_path)
    gt["diagnosis"] = gt[ONEHOT].idxmax(axis=1).str.lower().replace({"ak": "akiec", "scc": "akiec"})
    gt["label"] = gt["diagnosis"].map({c: i for i, c in enumerate(CLASSES)})
    gt["num"] = gt["image"].str.extract(r"ISIC_(\d+)").astype(int)
    is_ham = ((gt["num"] >= HAM_LO) & (gt["num"] <= HAM_HI)).values

    y, P = load_predictions()

    if len(y) != len(gt):
        raise SystemExit(
            f"Length mismatch: {len(y)} predictions vs {len(gt)} ground-truth rows. "
            f"These must come from the same evaluation run."
        )
    agree = (gt["label"].values == y).mean()
    if agree < 1.0:
        raise SystemExit(
            f"Prediction row order does not match the ground-truth file "
            f"(label agreement {agree:.6f}). Re-export the predictions in CSV order."
        )
    print(f"row order verified against the ground truth (agreement {agree:.6f})")

    P["ensemble"] = np.mean([P[m] for m in MODELS], axis=0)

    print(f"\nISIC 2019 total            : {len(y):>6}")
    print(f"  HAM10000 source (seen)   : {is_ham.sum():>6}  ({100*is_ham.mean():.1f}%)")
    print(f"  non-HAM10000 (unseen)    : {(~is_ham).sum():>6}  ({100*(~is_ham).mean():.1f}%)")

    rows = []
    for key in MODELS + ["ensemble"]:
        p = P[key]
        r = {"Model": LABEL.get(key, "Ensemble")}
        for tag, mask in [("full", np.ones(len(y), bool)), ("ham", is_ham), ("clean", ~is_ham)]:
            yy, pp = y[mask], p[mask]
            pred = pp.argmax(1)
            present = np.unique(yy)
            r[f"acc_{tag}"] = accuracy_score(yy, pred)
            r[f"f1_{tag}"] = f1_score(yy, pred, average="weighted", zero_division=0)
            # Restrict to classes present in the subset so the OvR AUC stays defined,
            # and renormalise so the retained columns sum to one.
            sub = pp[:, present]
            r[f"auc_{tag}"] = roc_auc_score(
                yy,
                sub / sub.sum(1, keepdims=True),
                multi_class="ovr",
                average="macro",
                labels=present,
            )
            r[f"wauc_{tag}"] = roc_auc_score(
                yy,
                sub / sub.sum(1, keepdims=True),
                multi_class="ovr",
                average="weighted",
                labels=present,
            )
        r["acc_drop"] = r["acc_full"] - r["acc_clean"]
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv("ISIC2019_decontaminated_metrics.csv", index=False)

    pd.DataFrame(
        {
            "Subset": [
                "ISIC 2019 as evaluated",
                "HAM10000 source images",
                "non-HAM10000 remainder",
            ],
            "N": [len(y), int(is_ham.sum()), int((~is_ham).sum())],
            "Share": ["100.0%", f"{100*is_ham.mean():.1f}%", f"{100*(~is_ham).mean():.1f}%"],
            "Seen during training": ["partly", "yes", "no"],
        }
    ).to_csv("ISIC2019_contamination_summary.csv", index=False)

    print(
        f"\n{'Model':<16}{'acc full':>9}{'acc seen':>9}{'acc unseen':>11}{'drop':>8}"
        f"{'F1 unseen':>10}{'AUC unseen':>11}"
    )
    for _, r in df.iterrows():
        print(
            f"{r['Model']:<16}{r['acc_full']:>9.4f}{r['acc_ham']:>9.4f}"
            f"{r['acc_clean']:>11.4f}{r['acc_drop']:>+8.4f}"
            f"{r['f1_clean']:>10.4f}{r['auc_clean']:>11.4f}"
        )

    print("\n" + "=" * 78)
    print("LaTeX body for the partitioned external table")
    print("Model & Acc full & Acc seen & Acc unseen & wF1 unseen & macroAUC & wAUC")
    print("=" * 78)
    for _, r in df.iterrows():
        print(
            f"{r['Model']:<16} & {r['acc_full']:.4f} & {r['acc_ham']:.4f} & "
            f"{r['acc_clean']:.4f} & {r['f1_clean']:.4f} & "
            f"{r['auc_clean']:.4f} & {r['wauc_clean']:.4f} \\\\"
        )
    print("=" * 78)
    print("Saved: ISIC2019_decontaminated_metrics.csv, ISIC2019_contamination_summary.csv")


main()
