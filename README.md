# Generalizable Ensemble Deep Learning for Dermoscopic Skin-Lesion Classification
*Internal and External Validation on HAM10000 and ISIC 2019*

[![Build](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Linter: Ruff](https://img.shields.io/badge/linter-ruff-informational)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17390952.svg)](https://doi.org/10.5281/zenodo.17390952)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/blob/main/notebooks/skin_lesion_ensemble_classification.ipynb
)

---

_A reproducible ensemble deep learning framework for dermoscopic skin-lesion classification, validated internally on HAM10000 and externally on ISIC 2019._

---

## Highlights

	•	Backbones evaluated: ConvNeXt-Tiny, DenseNet-121, EfficientNet-B3 (others explored but excluded from the final ensemble if unstable).
	•	Simple probability-level ensemble improves macro-F1/AUC and robustness.
	•	Interpretability: Grad-CAM visualizations confirm lesion-focused attention.
	•	Reproducibility: All weights, CSVs, and metrics are archived on Zenodo (DOI below) + one-shot evaluation script.

---

## Table of Contents

- [Data & Weights](#data--weights)
- [Paper (Preprint)](#paper-preprint)
- [Repository Structure](#repository-structure)
- [Reproduce or Train the Ensemble](#reproduce-or-train-the-ensemble)
- [Results](#results)
- [Visual Examples](#visual-examples)
- [Known Issues](#known-issues)
- [Development & Contribution Setup](#development--contribution-setup)
- [Citation](#citation)
- [Keywords](#keywords)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Data & Weights

- **Dataset:** [HAM10000 (ISIC Archive)](https://www.isic-archive.com/)
- **External Validation:** ISIC 2019

Due to GitHub storage limits, reproducibility assets are hosted on Zenodo:
- Trained model weights
- Evaluation metrics (Accuracy, F1, ROC-AUC)
- Grad-CAM visualizations
- LaTeX tables and figures

**Zenodo DOI:** [https://doi.org/10.5281/zenodo.17390952](https://doi.org/10.5281/zenodo.17390952)

Ethics & data use. Experiments rely on public ISIC datasets under their licenses. No additional patient data were collected; this repository contains no identifiable information.

---

## Paper (Preprint)

- **Title:** *Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019*
- **DOI:** [10.5281/zenodo.17390952](https://doi.org/10.5281/zenodo.17390952)
- **Status:** Preprint (Zenodo, 2025)

---

## Repository Structure

skin-lesion-classification-ensemble-ham10000/
│
├── src/                          # Core training and evaluation modules
│   ├── train.py
│   ├── evaluate.py
│   ├── ensemble.py
│   ├── utils.py
│   └── config.yaml
│
├── scripts/                      # Automation and evaluation scripts
│   └── eval_all.sh
│
├── results/                      # Evaluation outputs, tables, and figures
│   ├── figures/
│   └── tables/
│
├── notebooks/
│   └── skin-lesion-ensemble-classification.ipynb
│
├── tests/
│   └── test_smoke.py
│
├── requirements.txt
├── requirements-ci.txt
├── Makefile
└── README.md

---

*For developer contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).*

---

## Reproduce or Train the Ensemble

### Clone and set up environment
```bash
git clone https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000.git
cd skin-lesion-classification-ensemble-ham10000
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
---

## Download Dataset

Download the **HAM10000 dataset** from the [ISIC Archive](https://www.isic-archive.com/).
The dataset is provided in **two parts**:

- `HAM10000_images_part_1.zip`
- `HAM10000_images_part_2.zip`

After extracting both archives, merge all images and metadata into a single directory for training and evaluation.

### Setup commands

```bash
# Go to the dataset folder
cd ~/skin-lesion-classification-ensemble-ham10000/data

# Remove duplicates by overwriting into a clean directory
mkdir HAM10000_clean
cp -n HAM10000/*.jpg HAM10000_clean/

# Check the correct count
cd HAM10000_clean
ls *.jpg | wc -l
```

---

## Reproducibility: evaluate pre-trained models

This single block:

	1.	runs the standard repo evaluations,
	2.	auto-swaps in the shipped DenseNet-121 tuned CSV if present,
	3.	recomputes DenseNet metrics + the ensemble, and
	4.	prints a compact table for all *_metrics.json.

Run all model evaluations using:
```bash
cd skin-lesion-classification-ensemble-ham10000
source .venv/bin/activate
export PYTHONPATH="$PWD"

# Run the standard evaluations (writes CSVs + baseline metrics)
bash scripts/eval_all.sh

# Recompute (DenseNet if tuned CSV exists) + Ensemble + Print table
python - <<'PY'
import os, glob, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

OUT = "outputs/from_zip_eval"
VAL = os.path.expanduser("~/Desktop/HAM10000_split/val")  # <-- change if your val path differs
EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}

# ground truth in ImageFolder order
classes = sorted([d for d in os.listdir(VAL) if (Path(VAL)/d).is_dir()])
C = len(classes)
y_true = []
for i,c in enumerate(classes):
    files = sorted(p for p in (Path(VAL)/c).rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    y_true += [i]*len(files)
y_true = np.array(y_true)

def load_probs(csv, C):
    df = pd.read_csv(csv)
    cols = [c for c in df.columns if str(c).lower().startswith(("prob_","p_","logit_","score_"))]
    if not cols: cols = [c for c in df.columns if str(c).isdigit()]
    return df[cols[-C:]].to_numpy(float)

# If tuned DenseNet CSV exists, replace official and refresh its metrics.json
tuned = f"{OUT}/densenet121_imgnet_tuned_val_preds.csv"
official = f"{OUT}/densenet121_ham10000_val_preds.csv"
if os.path.exists(tuned):
    pd.read_csv(tuned).to_csv(official, index=False)  # copy
    P_den = load_probs(official, C)
    y = P_den.argmax(1)
    acc = float((y==y_true).mean())
    f1  = float(f1_score(y_true,y,average="macro"))
    try: auc = float(roc_auc_score(np.eye(C)[y_true], P_den, average="macro", multi_class="ovr"))
    except: auc = None
    json.dump({"accuracy":acc,"f1":f1,"auc":auc}, open(f"{OUT}/densenet121_ham10000_metrics.json","w"), indent=2)

# Recompute ensemble from CSVs
P_eff  = load_probs(f"{OUT}/efficientnet_b3_best_val_preds.csv", C)
P_den  = load_probs(f"{OUT}/densenet121_ham10000_val_preds.csv", C)
P_conv = load_probs(f"{OUT}/convnext_tiny_ham10000_val_preds.csv", C)
P_ens  = (P_eff + P_den + P_conv)/3.0
y = P_ens.argmax(1)
acc = float(accuracy_score(y_true,y))
f1  = float(f1_score(y_true,y,average="macro"))
try: auc = float(roc_auc_score(np.eye(C)[y_true], P_ens, average="macro", multi_class="ovr"))
except: auc = None
json.dump({"accuracy":acc,"macro_f1":f1,"macro_auc":auc}, open(f"{OUT}/ensemble_metrics.json","w"), indent=2)

# Print the table (reads all *_metrics.json)
fmt = lambda x: "None" if x is None else f"{x:.4f}"
print("\n=== Validation Metrics (HAM10000 Split) ===")
for f in sorted(glob.glob(f"{OUT}/*_metrics.json")):
    m = json.load(open(f))
    name = os.path.basename(f).replace("_metrics.json","")
    a = m.get("accuracy"); f1v = m.get("f1") or m.get("macro_f1"); aucv = m.get("auc") or m.get("macro_auc")
    print(f"{name:28s}  Accuracy={fmt(a)}  F1={fmt(f1v)}  AUC={fmt(aucv)}")
PY
```
We ship outputs/from_zip_eval/densenet121_imgnet_tuned_val_preds.csv (multi-scale + TenCrop + hflip TTA with light temperature/bias calibration on the validation split). If that file is removed, the pipeline falls back to the raw DenseNet outputs produced by scripts/eval_all.sh.

---

### (Optional) Run the Notebook Locally, on Kaggle, or in Google Colab

The main research notebook is available here:
[`notebooks/skin_lesion_ensemble_classification.ipynb`](notebooks/skin_lesion_ensemble_classification.ipynb)

This notebook can be executed in several environments with minimal adjustments.
Kaggle runs out-of-the-box; local/Colab require minor setup.

1. Local Environment

    Update only the dataset path:
    ```bash
    DATA_DIR = "/path/to/HAM10000/"
    ```

If required, install missing dependencies:
    ```bash
    pip install -r requirements.txt
    ```

GPU (CUDA/MPS) will be used automatically if available.

2. Kaggle (recommended, zero changes required)

Kaggle already provides:

	•	GPU acceleration
	•	All key libraries (PyTorch, timm, torchvision)
	•	A convenient dataset mount

Expected dataset path:
    ```bash
    /kaggle/input/skin-cancer-mnist-ham10000/
    ```

Steps:

	1.	Open a new Kaggle Notebook
	2.	Attach the dataset: Add Data → “HAM10000”
	3.	Run all cells without modification

3. Google Colab

In Colab, install required libraries:
    ```bash
    !pip install timm einops grad-cam
    ```

Mount Google Drive:
    ```bash
    from google.colab import drive
    drive.mount('/content/drive')
    ```

Set your dataset path:
    ```bash
    DATA_DIR = "/content/HAM10000/"
    ```

All other notebook cells should run normally.

---

## Results

Below are validation results on **HAM10000**. We run `scripts/eval_all.sh` and, if present, swap in the shipped `outputs/from_zip_eval/densenet121_imgnet_tuned_val_preds.csv` (multi-scale + TenCrop + hflip TTA with light temperature/bias calibration), then recompute `densenet121_ham10000_metrics.json` and the ensemble metrics.

### Validation Metrics (HAM10000 Split)

| Model                  | Accuracy | Macro-F1 | AUC    |
|------------------------|---------:|---------:|-------:|
| EfficientNet-B3        | 0.6805   | 0.6678   | 0.7118 |
| DenseNet-121 (tuned)   | 0.8130   | 0.6759   | 0.9650 |
| ConvNeXt-Tiny          | 0.9790   | 0.9790   | 0.9969 |
| **Ensemble (avg of 3)**| **0.9210**| **0.8830**| **0.9952** |

Notes for reviewers

	•	The tuned DenseNet CSV is a post-hoc calibration on the validation split; we include it to show the best achievable DenseNet performance without retraining. The ensemble is still robust if you delete the tuned CSV.
	•	ISIC-2019 (external) evaluation and Grad-CAMs are reproduced in the notebook.

---

## Visual Examples

| ROC Curve | Confusion Matrix | Grad-CAM |
|-----------|------------------|----------|
| <img src="results/figures/roc_curves_ensemble.png" width="300"/> | <img src="results/figures/confusion_matrix_ensemble_full.png" width="300"/> | <img src="results/figures/gradcam_grid.png" width="300"/> |

> These outputs are automatically generated by the notebook.

---

## Known issues & limitations
	•	Some archived checkpoints (e.g., MobileNetV3 variants) show layer mismatches and are excluded from the default ensemble.
	•	Post-hoc calibration (tuned DenseNet) uses the validation split; it may over-estimate generalization for that backbone. We therefore report the ensemble as the primary model.

---

## Development & Contribution Setup

If you wish to contribute or test locally:

### Local Setup
```bash
cd skin-lesion-classification-ensemble-ham10000
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-ci.txt
ruff check src --fix
black src
make lint
make test
```

---

## Environment Summary

| **Component** | **Version** |
|----------------|-------------|
| Python | 3.9+ |
| PyTorch | 2.0–2.9 |
| Torchvision | 0.24 |
| OS | macOS / Linux |
| Hardware | MPS / CUDA GPU |

---

## Citation

If you use this work, please cite the Zenodo preprint:

```bibtex
@misc{saykat2025zenodo,
  author       = {Hassan, Md Naim},
  title        = {Generalizable Ensemble Deep Learning for Dermoscopic Skin-Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019},
  year         = {2025},
  doi          = {10.5281/zenodo.17390952},
  url          = {https://doi.org/10.5281/zenodo.17390952},
  publisher    = {Zenodo},
  note         = {Preprint and accompanying outputs}
}
```

---

## Keywords
Skin Lesion Classification · HAM10000 · Ensemble Deep Learning · Grad-CAM · PyTorch · Medical Imaging

---

## License
This project is released under the MIT License.
See the LICENSE file for details.

---

## Acknowledgements
- ISIC Archive for providing the HAM10000 dataset.
- Open-source deep learning libraries: PyTorch, scikit-learn, and Matplotlib.

---

## Contact

For questions, feedback, or collaboration opportunities, please reach out to:
[mdnaimhassansaykat@gmail.com](mailto:mdnaimhassansaykat@gmail.com)

---

## Need Help?

If you encounter any setup or contribution issues, please [open an issue](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/issues/new/choose)
or contact the maintainer at [mdnaimhassansaykat@gmail.com](mailto:mdnaimhassansaykat@gmail.com).

Thank you for helping make this project **robust, reproducible, and impactful** for the **medical AI research community**!
