# Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/blob/main/notebooks/skin-lesion-ensemble-classification.ipynb)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17390952.svg)](https://doi.org/10.5281/zenodo.17390952)
[![Star](https://img.shields.io/github/stars/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000?style=social)](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/stargazers)
[![Watch](https://img.shields.io/github/watchers/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000?style=social)](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/watchers)

> **Abstract:**  
> This project presents a generalizable ensemble deep learning framework for multiclass skin lesion classification across HAM10000 and ISIC 2019 datasets.  
> By combining seven deep architectures (CNN, ResNet50, DenseNet121, EfficientNetB3, ConvNeXt, MobileNetV3, and ViT), the ensemble achieves robust internal and external validation performance with improved reliability and interpretability (Grad-CAMs, calibration metrics, and confidence intervals).

This repository contains code, trained model checkpoints, evaluation outputs, and the preprint paper for our study:
“Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019”

## Key Features
- **Unified Ensemble Framework:** integrates multiple deep models for robust lesion classification.  
- **Cross-Dataset Validation:** evaluated on both **HAM10000** and **ISIC 2019** for generalizability.  
- **Explainable AI (XAI):** Grad-CAM visualizations for lesion region interpretability.  
- **Calibration Metrics:** includes ECE and reliability diagrams.  
- **Reproducible Pipeline:** fully scripted for training, evaluation, and ensemble inference.  
- **LaTeX Export:** auto-generates results and tables for research publication.

---

## Table of Contents

- [Data & Weights](#data--weights)
- [Paper (Preprint)](#paper-preprint)
- [Repository Structure](#repository-structure)
- [How to Reproduce](#how-to-reproduce)
- [Results](#results)
- [Citation](#citation)
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

## Results
- Ensemble consistently outperformed individual models in Accuracy, Weighted F1, and ROC-AUC.
- Calibration analysis using Expected Calibration Error (ECE) showed strong reliability.
- Grad-CAM visualizations confirmed clinically relevant lesion focus.

---

## Performance Summary (HAM10000)

| Model | Accuracy | F1 (Weighted) | ROC-AUC | Dataset Split |
|:------|:----------:|:--------------:|:--------:|:--------------|
| ResNet50 | 92.1% | 91.7% | 97.2% | Internal |
| ViT-B16 | 91.8% | 91.3% | 96.8% | Internal |
| Ensemble (7 models) | **94.5%** | **94.1%** | **98.3%** | Internal |
| Ensemble (ISIC 2019) | 92.7% | 92.2% | 97.9% | External |

---

## Environment & Hardware
- **OS:** macOS / Linux / Windows  
- **Python:** 3.9 or later  
- **PyTorch:** 2.0+ with CUDA (optional)  
- **Recommended GPU:** ≥8 GB VRAM (e.g., RTX 3060 or higher)  
- **CPU mode** supported (for testing or feature extraction)

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

## Contributing
Contributions, pull requests, and dataset integration (e.g., ISIC 2020) are welcome!  
Please open an issue to discuss proposed changes before submitting a PR.

---

## Contact
For questions or collaboration inquiries:  
[mdnaimhassansaykat@gmail.com](mailto:mdnaimhassansaykat@gmail.com)

---

## Institutional Support
This work was conducted as part of the **MSc in Artificial Intelligence** program at *Université Paris-Saclay*.  
We acknowledge the use of publicly available data from the **ISIC Archive**.

---

## Demo Preview
<img src="assets/demo.gif" width="800"/>

---

## Related Work
- Tschandl et al., *HAM10000 Dataset: A Large Collection of Multi-Source Dermatoscopic Images*, Sci. Data 2018.  
- Brinker et al., *Skin Cancer Classification Using Deep Ensembles*, JAMA Dermatology, 2020.  
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
