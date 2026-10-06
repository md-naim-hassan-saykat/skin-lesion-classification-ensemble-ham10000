"""
Post-hoc temperature scaling of the equal-weight ensemble and of each model
(manuscript Table 5, Supplementary Table S7).

Question: does a single temperature applied directly to the ensemble
probabilities reduce Expected Calibration Error while leaving classification
performance unchanged?

Inputs are the released prediction arrays only; nothing is retrained:

    canonical_ham10000_labels.npy
    canonical_<model>_probs.npy        (seven models)
    canonical_ensemble_probs.npy
    isic2019_external_eval_predictions.npz

Usage:
    python scripts/revision_analysis/calibration_analysis.py <data_dir>

<data_dir> is the folder holding the arrays above (default: current folder).
They are distributed in the Zenodo archive (doi:10.5281/zenodo.17390952).

Temperature scaling is applied to recovered logits L = log(p); softmax is
invariant to an additive constant, so log(p) is an exact logit representative.
A single positive temperature is a monotone rescaling, so the predicted class,
and therefore accuracy and weighted F1, cannot change; only the confidence
distribution moves.

Two estimates are reported:
  (a) in-sample T, fitted and evaluated on the whole cohort (optimistic);
  (b) cross-fitted T, 5-fold stratified with seed 42: T is fitted on four
      folds and ECE is measured on the held-out fold, then pooled. This is the
      estimate reported in the manuscript.

ECE uses 15 equal-width maximum-confidence bins, right-closed: (lo, hi].
"""

import sys

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

D = sys.argv[1] if len(sys.argv) > 1 else "."
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NBINS = 15
EPS = 1e-12


def ece(probs, y, n_bins=NBINS):
    """15-bin maximum-confidence multiclass ECE (Guo et al. 2017)."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += (m.sum() / len(y)) * abs(correct[m].mean() - conf[m].mean())
    return total


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_T(probs, y):
    """Minimise multiclass NLL over a single scalar temperature."""
    L = np.log(np.clip(probs, EPS, None))

    def nll(logT):
        p = softmax(L / np.exp(logT))
        return -np.log(np.clip(p[np.arange(len(y)), y], EPS, None)).mean()

    r = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(r.x))


def apply_T(probs, T):
    return softmax(np.log(np.clip(probs, EPS, None)) / T)


def crossfit(probs, y, n_splits=5, seed=42):
    """Out-of-fold temperature scaling: fit T on 4 folds, score the 5th."""
    out = np.zeros_like(probs)
    Ts = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(probs, y):
        T = fit_T(probs[tr], y[tr])
        Ts.append(T)
        out[te] = apply_T(probs[te], T)
    return out, Ts


def metrics(probs, y):
    pred = probs.argmax(1)
    return (
        accuracy_score(y, pred),
        f1_score(y, pred, average="weighted"),
        roc_auc_score(y, probs, multi_class="ovr", average="macro"),
        ece(probs, y),
    )


# ---------------------------------------------------------------- HAM10000
names = [
    "cnn",
    "resnet50",
    "densenet121",
    "efficientnet_b3",
    "convnext_tiny",
    "mobilenet_v3",
    "vit",
]
label = {
    "cnn": "CNN",
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b3": "EfficientNet-B3",
    "convnext_tiny": "ConvNeXt-Tiny",
    "mobilenet_v3": "MobileNetV3-L",
    "vit": "ViT",
}

y = np.load(f"{D}/canonical_ham10000_labels.npy")
P = {n: np.load(f"{D}/canonical_{n}_probs.npy") for n in names}
ens = np.load(f"{D}/canonical_ensemble_probs.npy")

print(f"HAM10000 cohort: n={len(y)}  ensemble array {ens.shape}")
print(
    f"reconstruction check |mean(P) - ensemble| max = "
    f"{np.abs(np.mean([P[n] for n in names], axis=0) - ens).max():.2e}"
)

a, f, auc, e = metrics(ens, y)
print(f"\nEnsemble as published : acc {a:.4f}  wF1 {f:.4f}  macroAUC {auc:.4f}  ECE {e*100:.2f}%")
print("  (paper reports acc 0.9386, wF1 0.9374, AUC 0.9951, ECE 11.54%)")

print("\n=== Temperature scaling applied to the ENSEMBLE ===")
T_all = fit_T(ens, y)
ens_in = apply_T(ens, T_all)
a1, f1_, auc1, e1 = metrics(ens_in, y)
print(
    f"(a) in-sample   T = {T_all:.4f} -> ECE {e1*100:.2f}%  (acc {a1:.4f}, wF1 {f1_:.4f}, AUC {auc1:.4f})"
)

ens_cv, Ts = crossfit(ens, y)
a2, f2, auc2, e2 = metrics(ens_cv, y)
print(
    f"(b) cross-fitted T = {np.mean(Ts):.4f} +/- {np.std(Ts):.4f} -> ECE {e2*100:.2f}%"
    f"  (acc {a2:.4f}, wF1 {f2:.4f}, AUC {auc2:.4f})"
)
print(f"    fold temperatures: {', '.join(f'{t:.3f}' for t in Ts)}")
print(
    f"    ECE reduction: {e*100:.2f}% -> {e2*100:.2f}%  "
    f"({(1 - e2/e)*100:.1f}% relative), accuracy unchanged at {a2:.4f}"
)

print("\n=== alternative: per-model temperature scaling, then average ===")
pm_cv = np.mean([crossfit(P[n], y)[0] for n in names], axis=0)
a3, f3, auc3, e3 = metrics(pm_cv, y)
print(
    f"per-model cross-fitted T then equal-weight average -> ECE {e3*100:.2f}%"
    f"  (acc {a3:.4f}, wF1 {f3:.4f}, AUC {auc3:.4f})"
)

print("\n=== per-model ECE before / after cross-fitted temperature scaling ===")
print(f"{'Model':<18}{'T':>7}{'ECE before':>13}{'ECE after':>12}")
for n in names:
    pc, Tn = crossfit(P[n], y)
    print(f"{label[n]:<18}{np.mean(Tn):>7.3f}{ece(P[n], y)*100:>12.2f}%{ece(pc, y)*100:>11.2f}%")
print(f"{'Ensemble':<18}{np.mean(Ts):>7.3f}{e*100:>12.2f}%{e2*100:>11.2f}%")

# ------------------------------------------------------------- ISIC 2019
print("\n\n=== ISIC 2019 (manuscript Table 5, external columns) ===")
z = np.load(f"{D}/isic2019_external_eval_predictions.npz")
yi = z["labels"]
Pi = {n: z[f"{n}_probs"] for n in names}
ensi = np.mean([Pi[n] for n in names], axis=0)
print(f"n={len(yi)}")
print(f"{'Model':<18}{'ECE':>9}{'ECE after TS':>15}")
for n in names:
    pc, _ = crossfit(Pi[n], yi)
    print(f"{label[n]:<18}{ece(Pi[n], yi)*100:>8.2f}%{ece(pc, yi)*100:>14.2f}%")
ens_i_cv, Ti = crossfit(ensi, yi)
print(f"{'Ensemble':<18}{ece(ensi, yi)*100:>8.2f}%{ece(ens_i_cv, yi)*100:>14.2f}%")
ai, fi, auci, _ = metrics(ensi, yi)
print(
    f"\nEnsemble ISIC: acc {ai:.4f}  wF1 {fi:.4f}  macroAUC {auci:.4f}"
    f"   (paper: 0.6830 / 0.6446 / 0.8981)"
)
print(f"ISIC ensemble cross-fitted T = {np.mean(Ti):.4f}")
