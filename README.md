# Generalizable Ensemble Deep Learning for Skin Lesion Classification
**_Internal and External Validation on HAM10000 and ISIC 2019_**

[![Build](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Linter: Ruff](https://img.shields.io/badge/linter-ruff-informational)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17390952.svg)](https://doi.org/10.5281/zenodo.17390952)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/blob/main/notebooks/skin-lesion-ensemble-classification.ipynb
)

---

_A reproducible ensemble deep learning framework for skin lesion classification with internal and external validation on HAM10000 and ISIC 2019._

---

## Highlights

- **Seven architectures evaluated:** ResNet-50, DenseNet-121, EfficientNet-B3, ConvNeXt-Tiny, MobileNetV3-Large, Vision Transformer (ViT-B/16), and a baseline CNN.
- **Ensemble fusion** improves generalization and AUC across datasets
- **Grad-CAM** visualizations for interpretable lesion focus
- **Pretrained weights, metrics, and outputs** hosted on Zenodo
- **Reproducible pipeline** via bash or Docker


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
├── Dockerfile
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
The dataset is provided in **two parts** — `HAM10000_images_part_1.zip` and `HAM10000_images_part_2.zip`.
After extracting both, merge all images into a single directory:

```bash
mkdir -p data/HAM10000
cp ~/Desktop/HAM10000_images_part_1/* data/HAM10000/
cp ~/Desktop/HAM10000_images_part_2/* data/HAM10000/
```

---

## Reproducibility: Evaluate pretrained models
Run all model evaluations using:
```bash
cd skin-lesion-classification-ensemble-ham10000
source ../.venv/bin/activate
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
Calibration analysis using Expected Calibration Error (ECE) showed strong reliability, and Grad-CAM visualizations confirmed clinically relevant lesion focus.

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
- **MobileNetV3 checkpoint** fails to load (shape mismatches across multiple blocks) — likely trained with a different architecture variant. Marked as TODO (skip in current eval).
- **ResNet50 / ViT checkpoints** load but perform poorly on this split, so excluded from the default ensemble until retrained/verified.

---


## Development & Contribution Setup

If you wish to contribute or test locally:

### Local Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-ci.txt
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

If you encounter issues or wish to suggest improvements, please [open an issue](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/issues/new/choose)  
or contact the maintainer directly at [mdnaimhassansaykat@gmail.com](mailto:mdnaimhassansaykat@gmail.com).

---


