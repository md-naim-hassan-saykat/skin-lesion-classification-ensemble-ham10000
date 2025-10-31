# Installation Guide

This guide explains how to set up and reproduce experiments for the **Generalizable Ensemble Deep Learning for Skin Lesion Classification** project.
The framework uses **PyTorch**, **scikit-learn**, and **Grad-CAM**, ensuring full reproducibility across macOS, Linux, and Google Colab.

---

## 1. Clone the Repository

```bash
git clone https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000.git
cd skin-lesion-classification-ensemble-ham10000
```

---

## 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Verify Setup

```bash
make lint
make test
```
or manually:
```bash
pytest -q
pytest --cov=src
```

---

## 4. Optional: Colab Setup
To reproduce results interactively, open the notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/blob/main/notebooks/skin-lesion-ensemble-classification.ipynb
)

---

## 5. Data Preparation
Download HAM10000 and merge parts 1 & 2 into data/HAM10000/.
```bash
mkdir -p data/HAM10000
cp ~/Downloads/HAM10000_images_part_*/*.jpg data/HAM10000/
```
