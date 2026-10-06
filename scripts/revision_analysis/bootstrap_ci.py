"""
Point estimates and 95% bootstrap percentile confidence intervals on the
standardized 2,003-image HAM10000 evaluation cohort (Supplementary Table S1,
results/tables/HAM10000_master_results_with_95CI.csv).

Procedure, as in the research notebook:

  * point estimates are computed once on the complete cohort;
  * each metric is bootstrapped separately with 1,000 resamples of the cohort,
    drawn with replacement using numpy.random.default_rng;
  * the generator is seeded per metric from a base seed of 42:
        accuracy 42, weighted F1 43, macro ROC-AUC 44, micro ROC-AUC 45;
  * a resample on which a metric is undefined (a class absent, so OvR ROC-AUC
    cannot be computed) is recorded as missing and excluded from the
    percentiles.

Because the four metrics use different seeds, their intervals come from
different resample sets. Running all four on one shared stream gives intervals
that differ in the third or fourth decimal place; the published table uses the
per-metric seeds above.

Usage:
    python scripts/revision_analysis/bootstrap_ci.py <data_dir> [out.csv]

<data_dir> holds canonical_ham10000_labels.npy and canonical_<model>_probs.npy
for the seven models and the ensemble (Zenodo doi:10.5281/zenodo.17390952).
"""

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

MODELS = {
    "CNN": "cnn",
    "ResNet-50": "resnet50",
    "DenseNet-121": "densenet121",
    "EfficientNet-B3": "efficientnet_b3",
    "ConvNeXt-Tiny": "convnext_tiny",
    "MobileNetV3-L": "mobilenet_v3",
    "ViT": "vit",
    "Ensemble": "ensemble",
}
BASE_SEED = 42
N_BOOTSTRAP = 1000


def _acc(y, p):
    return accuracy_score(y, p.argmax(1))


def _f1w(y, p):
    return f1_score(y, p.argmax(1), average="weighted")


def _auc_macro(y, p):
    return roc_auc_score(y, p, multi_class="ovr", average="macro")


def _auc_micro(y, p):
    return roc_auc_score(y, p, multi_class="ovr", average="micro")


METRICS = [
    ("Accuracy", _acc),
    ("Weighted F1", _f1w),
    ("Macro ROC-AUC (OvR)", _auc_macro),
    ("Micro ROC-AUC (OvR)", _auc_micro),
]


def point_and_ci(y, probs, fn, seed, n_bootstrap=N_BOOTSTRAP):
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            stats.append(fn(y[idx], probs[idx]))
        except ValueError:
            stats.append(np.nan)
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return fn(y, probs), lo, hi


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else "HAM10000_master_results_with_95CI.csv"

    y = np.load(f"{data_dir}/canonical_ham10000_labels.npy")
    rows = []
    for name, key in MODELS.items():
        probs = np.load(f"{data_dir}/canonical_{key}_probs.npy")
        row = {"Model": name}
        for j, (label, fn) in enumerate(METRICS):
            pt, lo, hi = point_and_ci(y, probs, fn, seed=BASE_SEED + j)
            row[label] = f"{pt:.4f} [{lo:.4f}, {hi:.4f}]"
        rows.append(row)
        print(" | ".join(str(v) for v in row.values()))

    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
