# Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/blob/main/notebooks/skin-lesion-ensemble-classification.ipynb)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Star](https://img.shields.io/github/stars/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000?style=social)](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/stargazers)
[![Watch](https://img.shields.io/github/watchers/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000?style=social)](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/watchers)
![CI](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/actions/workflows/ci.yml/badge.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17390952.svg)](https://doi.org/10.5281/zenodo.17390952)

> A reproducible PyTorch implementation and ensemble pipeline for robust skin lesion classification across HAM10000 and ISIC 2019 datasets. Includes trained weights, evaluation scripts, and Grad-CAM visualizations.

This repository contains code, trained model checkpoints, evaluation outputs, and the preprint paper for our study:
“Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019”

---

## Table of Contents

- [Data & Weights](#data--weights)
- [Paper (Preprint)](#paper-preprint)
- [Repository Structure](#repository-structure)
- [How to Reproduce](#how-to-reproduce)
- [Example Outputs](#example-outputs)
- [Results](#results)
  - [Validation Metrics (HAM10000 Split)](#validation-metrics-ham10000-split)
  - [Known Issues](#known-issues)
  - [Reproducibility](#reproducibility)
- [Citation](#citation)
- [Keywords](#keywords)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Data & Weights

- **Dataset**  
  HAM10000 (ISIC Archive): https://www.isic-archive.com/

- **Trained Weights & Artifacts**  
  Due to GitHub's file size limits, all reproducibility assets are hosted externally via Zenodo, including:  
  - Trained model weights (CNN, ResNet50, ViT, etc.)  
  - Evaluation metrics (Accuracy, ROC-AUC, etc.)  
  - Grad-CAM visualizations  
  - LaTeX tables and figures
  
  **Zenodo DOI**: [https://doi.org/10.5281/zenodo.17390952](https://doi.org/10.5281/zenodo.17390952)

---

## Paper (Preprint)

- **Title**: *Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019*  
- **DOI**: [10.5281/zenodo.17390952](https://doi.org/10.5281/zenodo.17390952)  
- **Status**: Preprint on Zenodo (not peer-reviewed or journal-published yet)

---

## Repository Structure

skin-lesion-classification-ensemble-ham10000/
│
├── main.tex                         # LaTeX source of the paper
├── references.bib                   # References for ESWA submission
├── README.md                        # Project documentation
├── .gitignore                       # Ignore large model files
│
├── notebooks/
│   └── skin-lesion-classification-ensemble-ham10000.ipynb   # End-to-end pipeline
│
├── results/
│   ├── figures/                     # Plots and visualizations
│   │   ├── confusion_matrix.png
│   │   ├── roc_curves_ensemble.png
│   │   ├── gradcam_grid.png
│   │   └── …
│   ├── tables/                      # LaTeX-ready tables
│   │   ├── master_results_with_95CI.tex
│   │   ├── delta_ensemble_vs_models.tex
│   │   └── mcnemar_ensemble_vs_models.tex
│   └── metrics_with_95CI.csv
│
└── paper/
└── skin_lesion_classification_ensemble_ham10000.pdf     # Compiled paper

---

## How to Reproduce

**Clone the repository**:
   ```bash
   git clone https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000.git
   cd skin-lesion-classification-ensemble-ham10000
   ```
---

## Install Dependencies
```bash
pip install -r requirements.txt
```
---

## Download Dataset:
Register and download HAM10000 from the ISIC Archive.
Place images under ./data/HAM10000/.

---

## Train Models & Run Ensemble Evaluation

All training, evaluation, and ensemble logic are implemented in a single Jupyter notebook:

`skin-lesion-classification-ensemble-ham10000.ipynb`

You can open this notebook in [Google Colab](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/blob/main/notebooks/skin-lesion-ensemble-classification.ipynb) or run it locally to:

- Train individual models (ResNet50, DenseNet121, ViT, EfficientNetB3, ConvNeXt, etc.)
- Perform ensemble evaluation
- Generate plots (e.g., confusion matrix, ROC-AUC)
- Visualize Grad-CAMs
- Export LaTeX tables for paper

> All experiments were performed using 7 backbone models: CNN, ResNet50, DenseNet121, ViT, EfficientNetB3, MobileNetV3, and ConvNeXt.

---

---

## Example Outputs

| ROC Curve | Confusion Matrix | Grad-CAM |
|-----------|------------------|----------|
| <img src="results/figures/roc_curves_ensemble.png" width="300"/> | <img src="results/figures/confusion_matrix_ensemble_full.png" width="300"/> | <img src="results/figures/gradcam_grid.png" width="300"/> |

> These outputs are automatically generated by the notebook.

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
	•	ISIC Archive for providing the HAM10000 dataset.
	•	Open-source deep learning libraries: PyTorch, scikit-learn, and Matplotlib.

---

## Contact

If you have any questions, encounter issues, or wish to collaborate, please feel free to contact me:  
[mdnaimhassansaykat@gmail.com](mailto:mdnaimhassansaykat@gmail.com)

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

### Known Issues
- **MobileNetV3 checkpoint** fails to load (shape mismatches across multiple blocks) — likely trained with a different architecture variant. Marked as TODO (skip in current eval).
- **ResNet50 / ViT checkpoints** load but perform poorly on this split, so excluded from the default ensemble until retrained/verified.

---

### Reproducibility

To reproduce validation metrics:

```bash
bash scripts/eval_all.sh
```
