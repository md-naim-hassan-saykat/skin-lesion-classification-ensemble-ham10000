# Training Pipeline

This document outlines the end-to-end **training pipeline** for the *Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019* project.  
It describes dataset preparation, preprocessing, training configuration, and model checkpointing.

---

## 1. Overview

The training pipeline is designed to ensure **reproducibility, scalability, and flexibility**.  
It supports multiple CNN backbones (e.g., ResNet, EfficientNet, ConvNeXt) and can be easily extended to new architectures or datasets.

All training configurations, dataset paths, and hyperparameters are defined via YAML files located in the `config/` directory.

---

## 2. Dataset Preparation

### 2.1 Source Dataset
We use the **HAM10000** dataset — a benchmark dataset for skin lesion analysis containing dermatoscopic images across **seven diagnostic categories**:
1. Melanocytic nevi (NV)  
2. Melanoma (MEL)  
3. Benign keratosis-like lesions (BKL)  
4. Basal cell carcinoma (BCC)  
5. Actinic keratoses (AKIEC)  
6. Vascular lesions (VASC)  
7. Dermatofibroma (DF)

Optional external datasets such as **ISIC 2019** can be added for cross-validation.

---

### 2.2 Train/Validation Split

The dataset is split into training and validation subsets using the script:

```bash
python scripts/prepare_split.py
