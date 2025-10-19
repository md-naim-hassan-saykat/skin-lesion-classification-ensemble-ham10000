# Ensemble Deep Learning for Robust Skin Lesion Classification (HAM10000)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/blob/main/notebooks/skin-lesion-ensemble-classification.ipynb)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains code, trained model checkpoints, evaluation outputs, and the preprint paper for our study:
“Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019”

---

## Data & Weights

- **Dataset**  
  HAM10000 (ISIC Archive): https://www.isic-archive.com/

- **Trained Weights**  
  Due to GitHub's file size limits, trained model weights are hosted externally:  
  - [Zenodo DOI](https://doi.org/10.5281/zenodo.17390952)

---

## Paper (Preprint)

**Zenodo DOI**: [10.5281/zenodo.17390952](https://doi.org/10.5281/zenodo.17390952)  
**Title**: *Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019*  
**Status**: Preprint on Zenodo (not yet peer-reviewed or journal-published)

---

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

1. **Clone the repository**:
   ```bash
   git clone https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000.git
   cd skin-lesion-classification-ensemble-ham10000
   ```
   
---

## Install dependencies:
pip install -r requirements.txt

---

## Download dataset:
Register and download HAM10000 from the ISIC Archive.
Place images under ./data/HAM10000/.

---

## Train models:
python train.py --model resnet50 --epochs 50
python train.py --model densenet121 --epochs 50
# ... repeat for other backbones

---

## Run ensemble evaluation:
python ensemble.py --models resnet50 densenet121 vit efficientnet_b3

---

## 	Reproduce tables/figures:
Use the Jupyter notebook in ./notebooks/ to regenerate plots and LaTeX-ready tables.

---

## Results
	•	Ensemble consistently outperformed individual models in Accuracy, Weighted F1, and ROC-AUC.
	•	Calibration analysis using Expected Calibration Error (ECE) showed strong reliability.
	•	Grad-CAM visualizations confirmed clinically relevant lesion focus.

## See full results in the paper:
skin_lesion_classification_ensemble_ham10000.pdf

---

## Citation
@misc{saykat2025zenodo,
  author       = {Hassan, Md Naim},
  title        = {Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019},
  year         = 2025,
  doi          = {10.5281/zenodo.17390952},
  url          = {https://doi.org/10.5281/zenodo.17390952},
  publisher    = {Zenodo},
  note         = {Preprint and accompanying outputs}
}

---

## License
This project is released under the MIT License.
See the LICENSE file for details.

---

## Acknowledgements
	•	ISIC Archive for providing the HAM10000 dataset.
	•	Open-source deep learning libraries: PyTorch, scikit-learn, and Matplotlib.

 ---

 ## Author

 **Md Naim Hassan Saykat**  
*MSc in Artificial Intelligence, Université Paris-Saclay*  

[LinkedIn](https://www.linkedin.com/in/md-naim-hassan-saykat/)  
[GitHub](https://github.com/md-naim-hassan-saykat)  
[Academic Email](mailto:md-naim-hassan.saykat@universite-paris-saclay.fr)  
[Personal Email](mailto:mdnaimhassansaykat@gmail.com)
