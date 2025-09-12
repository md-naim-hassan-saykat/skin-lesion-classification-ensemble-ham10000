# Ensemble Deep Learning for Robust Skin Lesion Classification (HAM10000)

This repository contains the code, results, and supporting material for the paper:  
**"Ensemble Deep Learning for Robust Skin Lesion Classification: A Comprehensive Study on HAM10000"** (Submitted to *Expert Systems with Applications*, 2025).

---

skin-lesion-classification-ensemble-ham10000/
│
├── main.tex
├── references.bib
├── README.md
├── .gitignore
├── notebooks/
│   └── skin-lesion-classification-ensemble-ham10000.ipynb
├── results/
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curves_ensemble.png
│   │   ├── gradcam_grid.png
│   │   └── ...
│   ├── tables/
│   │   ├── master_results_with_95CI.tex
│   │   └── delta_ensemble_vs_models.tex
│   └── metrics_with_95CI.csv
└── paper/
    └── skin_lesion_classification_ensemble_ham10000.pdf

---

---

## 📊 Dataset
- **HAM10000 dataset** (10,015 dermoscopic images, 7 diagnostic categories):  
  🔗 [ISIC Archive](https://www.isic-archive.com/)

---

## 🏗️ Models
We evaluated **7 architectures**:
- CNN (baseline)
- ResNet-50
- DenseNet-121
- EfficientNet-B3
- ConvNeXt-Tiny
- MobileNetV3
- Vision Transformer (ViT)

Final predictions were aggregated via **probability-level ensembling**.

---

## Results
- ROC–AUC (macro) > **0.90** for all individual models.
- Ensemble outperformed every single backbone (Accuracy, Weighted F1, ROC–AUC).
- Statistical significance validated with:
  - Bootstrap CIs  
  - McNemar’s test  

See figures in `output/`:
- `confusion_matrix.png`  
- `roc_curves_ensemble.png`  
- `calibration_ece.png`  
- `gradcam_grid.png`  

---

## Reproduce
Clone the repository:
```bash
git clone https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000.git
cd skin-lesion-classification-ensemble-ham10000

---

## Weights & Checkpoints

Due to GitHub’s file size limits, trained weights are not stored here.
They will be shared via:
	•	GitHub Release v1.0
	•	Zenodo (recommended for citation)
	•	Hugging Face Hub (optional, if added)

## Citation
@article{saykat2025ensemble,
  title   = {Ensemble Deep Learning for Robust Skin Lesion Classification: A Comprehensive Study on HAM10000},
  author  = {Saykat, Md Naim Hassan},
  journal = {Expert Systems with Applications},
  year    = {2025},
  note    = {Submitted}
}

---

## Contact

Md Naim Hassan Saykat
University email: md-naim-hassan.saykat@universite-paris-saclay.fr
Personal email: mdnaimhassansaykat@gmail.com
