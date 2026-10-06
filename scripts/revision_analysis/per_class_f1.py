"""
Per-class F1 on the full ISIC 2019 set and on the genuinely unseen remainder
(Supplementary Tables S4 and S5).

The unseen remainder is the 15,320 BCN_20000 and MSK images, after removing the
10,011 HAM10000 images (identifier block ISIC_0024307 to ISIC_0034317) that every
checkpoint saw during training. Inputs (ISIC_2019_Training_GroundTruth.csv and
isic2019_external_eval_predictions.npz) are located automatically under
/kaggle/working, /kaggle/input and the current folder.

Usage:
    cd /path/to/inputs && python /path/to/scripts/revision_analysis/per_class_f1.py
"""

import os
from glob import glob

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SEARCH_ROOTS = ["/kaggle/working", "/kaggle/input", "."]
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MODELS = [
    "cnn",
    "resnet50",
    "densenet121",
    "efficientnet_b3",
    "convnext_tiny",
    "mobilenet_v3",
    "vit",
]
LABEL = dict(
    zip(
        MODELS,
        [
            "CNN",
            "ResNet-50",
            "DenseNet-121",
            "EfficientNet-B3",
            "ConvNeXt-Tiny",
            "MobileNetV3-L",
            "ViT",
        ],
    )
)
HAM_LO, HAM_HI = 24307, 34317


def find(name):
    hits = []
    for root in SEARCH_ROOTS:
        if os.path.isdir(root):
            hits += glob(os.path.join(root, "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found under {SEARCH_ROOTS}")
    return sorted(set(hits), key=len)[0]


gt = pd.read_csv(find("ISIC_2019_Training_GroundTruth.csv"))
z = np.load(find("isic2019_external_eval_predictions.npz"))
y = z["labels"]
P = {m: z[f"{m}_probs"] for m in MODELS}
P["ensemble"] = np.mean([P[m] for m in MODELS], axis=0)

num = gt["image"].str.extract(r"ISIC_(\d+)")[0].astype(int).values
is_ham = (num >= HAM_LO) & (num <= HAM_HI)
assert len(y) == len(gt) and is_ham.sum() == 10011 and (~is_ham).sum() == 15320

tables = {}
for title, mask in [
    ("S4: full ISIC 2019 set (n = 25,331)", np.ones(len(y), bool)),
    ("S5: unseen remainder (n = 15,320)", ~is_ham),
]:
    rows = {
        LABEL.get(k, "Ensemble"): f1_score(
            y[mask], p[mask].argmax(1), average=None, labels=range(7), zero_division=0
        )
        for k, p in P.items()
    }
    df = pd.DataFrame(rows, index=[c.upper() for c in CLASSES]).T
    tables[title] = df
    print(f"\nPer-class F1, {title}")
    print(df.round(2).to_string())
    print(
        "Highest per class:",
        ", ".join(f"{c}: {df[c].idxmax()} ({df[c].max():.4f})" for c in df.columns),
    )

full, unseen = tables.values()
higher = (full.values > unseen.values).sum()
print(f"\nFull-set F1 higher than unseen F1 in {higher} of {full.size} model-class cells")
unseen.round(4).to_csv("ISIC2019_unseen_per_class_F1.csv")
print("Saved: ISIC2019_unseen_per_class_F1.csv")
