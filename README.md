# Generalizable Ensemble Deep Learning for Skin Lesion Classification
### *Internal and External Validation on HAM10000 and ISIC 2019*

This repository provides a reproducible **PyTorch-based ensemble framework** for robust skin lesion classification across **HAM10000** and **ISIC 2019** datasets.
It integrates multiple deep learning backbones (ResNet, DenseNet, ViT, ConvNeXt, etc.), ensemble fusion for improved generalization, and Grad-CAM visualizations for interpretability.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/blob/main/notebooks/skin-lesion-ensemble-classification.ipynb)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Star](https://img.shields.io/github/stars/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000?style=social)](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/stargazers)
[![Watch](https://img.shields.io/github/watchers/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000?style=social)](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/watchers)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Linter: Ruff](https://img.shields.io/badge/linter-ruff-informational)
![Tests](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/actions/workflows/ci.yml/badge.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17390952.svg)](https://doi.org/10.5281/zenodo.17390952)

> A reproducible ensemble pipeline for high-performing skin lesion classification, including pretrained weights, evaluation scripts, and Grad-CAM interpretability tools.

---

## Highlights

- **7 architectures:** ResNet50, DenseNet121, ViT, EfficientNetB3, MobileNetV3, ConvNeXt, and CNN baseline
- **Ensemble fusion** improves generalization and AUC across datasets
- **Grad-CAM** visualizations for interpretable lesion attention
- **Pretrained weights, metrics, and outputs** hosted on Zenodo
- **Reproducible workflow** via Bash scripts or Makefile

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

- **Datasets:** [HAM10000 (ISIC Archive)](https://www.isic-archive.com/) and ISIC 2019 (external validation)
- **Assets hosted on Zenodo** due to GitHub storage limits:
  - Trained model weights
  - Evaluation metrics (Accuracy, F1, ROC-AUC)
  - Grad-CAM visualizations
  - LaTeX tables and figures

🔗 **Zenodo DOI:** [https://doi.org/10.5281/zenodo.17390952](https://doi.org/10.5281/zenodo.17390952)

---

## Paper (Preprint)

**Title:** *Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019*
**DOI:** [10.5281/zenodo.17390952](https://doi.org/10.5281/zenodo.17390952)
**Status:** Preprint (Zenodo, 2025)

---

## Repository Structure

skin-lesion-classification-ensemble-ham10000/
├── src/                          # Core training and evaluation modules
│   ├── train.py
│   ├── evaluate.py
│   ├── ensemble.py
│   ├── utils.py
│   ├── data.py
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
├── Dockerfile
├── Makefile
└── README.md

---

> For developer contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Reproduce or Train the Ensemble

### Setup and run all evaluations
```bash
cd ~/skin-lesion-classification-ensemble-ham10000
chmod +x scripts/eval_all.sh
export PYTHONPATH="$PWD"
bash scripts/eval_all.sh
```

---

## Download Dataset

Download HAM10000 from the ISIC Archive.
Merge both image parts after extraction:

```bash
# Create dataset folder
mkdir -p data/HAM10000

# Copy both parts (from Downloads or Desktop)
cp -r ~/Downloads/HAM10000_images_part_1/* data/HAM10000/ 2>/dev/null || true
cp -r ~/Downloads/HAM10000_images_part_2/* data/HAM10000/ 2>/dev/null || true
cp -r ~/Desktop/HAM10000_images_part_1/* data/HAM10000/ 2>/dev/null || true
cp -r ~/Desktop/HAM10000_images_part_2/* data/HAM10000/ 2>/dev/null || true

# Optionally unzip if still compressed
if [ -f ~/Downloads/HAM10000_images_part_1.zip ]; then
  unzip -n ~/Downloads/HAM10000_images_part_1.zip -d ~/Downloads/HAM10000_images_part_1
  cp -r ~/Downloads/HAM10000_images_part_1/* data/HAM10000/
fi
if [ -f ~/Downloads/HAM10000_images_part_2.zip ]; then
  unzip -n ~/Downloads/HAM10000_images_part_2.zip -d ~/Downloads/HAM10000_images_part_2
  cp -r ~/Downloads/HAM10000_images_part_2/* data/HAM10000/
fi

# Check total images (should print 10015)
find data/HAM10000 -type f -name "*.jpg" | wc -l
```

---

## Reproducibility: Evaluate Pretrained Models
```bash
cd skin-lesion-classification-ensemble-ham10000
source .venv/bin/activate
export PYTHONPATH="$PWD"
bash scripts/eval_all.sh
```
---

### (Optional) Train or Visualize Locally / in Colab

Open the notebook:
[`notebooks/skin-lesion-ensemble-classification.ipynb`](notebooks/skin-lesion-ensemble-classification.ipynb)

---

## Results

The ensemble consistently outperformed individual models in Accuracy, Weighted F1, and ROC-AUC on both internal (HAM10000) and external (ISIC 2019) validation.
Calibration via Expected Calibration Error (ECE) showed high reliability, and Grad-CAM confirmed clinically relevant lesion focus.

### Validation Metrics (HAM10000 Split)

Reproduced from `scripts/eval_all.sh`.

| Model                  | Accuracy | F1     | AUC    |
|------------------------|---------:|-------:|-------:|
| EfficientNet-B3        | 0.638    | 0.625  | 0.671  |
| DenseNet-121           | 0.809    | 0.791  | 0.968  |
| ConvNeXt-Tiny          | 0.979    | 0.979  | 0.998  |
| **Ensemble (above 3)** | **0.883**| **0.868**| **0.992** |

> Metrics are from `outputs/from_zip_eval/*_metrics.json` and `ensemble_metrics.json`.

---

## Visual Examples

| ROC Curve | Confusion Matrix | Grad-CAM |
|-----------|------------------|----------|
| <img src="results/figures/roc_curves_ensemble.png" width="300"/> | <img src="results/figures/confusion_matrix_ensemble_full.png" width="300"/> | <img src="results/figures/gradcam_grid.png" width="300"/> |

> These outputs are automatically generated by the notebook.

---

## Known Issues
	•	MobileNetV3 checkpoint: Shape mismatches (under retraining).
	•	ResNet50 / ViT models: Load but underperform on this split — excluded from default ensemble.

---


## Development & Contribution Setup
```bash
git clone https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000.git
cd skin-lesion-classification-ensemble-ham10000

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -r requirements-ci.txt

[ -f src/__init__.py ] || touch src/__init__.py
export PYTHONPATH="$PWD"

ruff clean
ruff check src scripts tests --fix
make lint
make test
```

### Local Development
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-ci.txt
pre-commit install
pytest -q && ruff check .
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
  title        = {Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019},
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
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements
	•	ISIC Archive for the HAM10000 dataset
	•	PyTorch, scikit-learn, Matplotlib — core open-source tools used throughout

---

## Contact

For questions, collaborations, or bug reports:
mdnaimhassansaykat@gmail.com

---
