"""
Verifies that the near-chance ISIC 2019 results originally reported for ViT and
ResNet-50 were caused by a class-index mismatch, not by the models.

Both models were trained under the unsorted dx mapping in the notebook,
    {label: idx for idx, label in enumerate(df['dx'].unique())}
which follows the insertion order of HAM10000_metadata.csv:
    ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']
while the original ISIC evaluation scored them against the canonical sorted order:
    ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

Reindexing column k of the model output to canonical class k with META below
recovers the expected performance, and a brute-force search over all 5,040
permutations recovers META as the optimum for ViT.

Usage:
    python scripts/revision_analysis/verify_label_order.py isic2019_all_model_predictions.csv

The input is the per-image prediction export of the original (pre-correction)
ISIC 2019 evaluation, distributed in the Zenodo archive.
"""

import ast
import sys
from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PRED_CSV = sys.argv[1] if len(sys.argv) > 1 else "isic2019_all_model_predictions.csv"
MODELS = ["CNN", "ResNet50", "DenseNet121", "EfficientNetB3", "ConvNeXtTiny", "MobileNetV3", "ViT"]
BROKEN = ["ResNet50", "ViT"]
META = (6, 5, 0, 2, 3, 1, 4)  # canonical index -> column in the model's own order

d = pd.read_csv(PRED_CSV)
y = d["label"].values
P = {m: np.array([ast.literal_eval(s) for s in d[m + "_probs"]]) for m in MODELS}


def report(name, probs):
    pred = probs.argmax(1)
    acc = accuracy_score(y, pred)
    wf1 = f1_score(y, pred, average="weighted")
    auc = roc_auc_score(y, probs, multi_class="ovr")
    print(f"{name:<24} acc {acc:.4f}  wF1 {wf1:.4f}  AUC {auc:.4f}")


print("=== brute-force search over all 7! permutations ===")
for m in BROKEN:
    best = max(
        ((accuracy_score(y, P[m][:, list(p)].argmax(1)), p) for p in permutations(range(7))),
        key=lambda t: t[0],
    )
    ident = accuracy_score(y, P[m].argmax(1))
    print(
        f"{m:<12} identity acc {ident:.4f} -> best perm {best[1]} acc {best[0]:.4f}"
        f"  (META match: {best[1] == META})"
    )

print("\n=== per-model, as published vs reordered ===")
for m in BROKEN:
    report(m + " as-published", P[m])
    report(m + " REORDERED", P[m][:, list(META)])

print("\n=== ensemble ===")
Q = dict(P)
for m in BROKEN:
    Q[m] = P[m][:, list(META)]
report("ensemble as-published", np.mean([P[m] for m in MODELS], axis=0))
report("ensemble fixed", np.mean([Q[m] for m in MODELS], axis=0))
