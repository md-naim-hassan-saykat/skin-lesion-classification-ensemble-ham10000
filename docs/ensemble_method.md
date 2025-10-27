# Ensemble Method

This document describes the ensemble strategy used in the **Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019** project.  
The goal of the ensemble is to combine multiple deep learning models to improve classification robustness, calibration, and overall performance.

---

## 1. Overview

Individual CNN architectures (e.g., ResNet, EfficientNet, ConvNeXt, DenseNet) are trained independently on the same training and validation splits.  
Each model generates probabilistic outputs (softmax scores) for the seven lesion categories.  
These probabilities are aggregated to produce the final ensemble prediction.

---

## 2. Motivation

Ensembling leverages the **complementary strengths** of diverse models:
- **ResNet** captures hierarchical texture features.  
- **EfficientNet** focuses on efficient scaling and balanced capacity.  
- **ConvNeXt** captures contextual relationships with modern convolutional design.

By averaging predictions, the ensemble mitigates overfitting and model bias.

---

## 3. Ensemble Strategy

### 3.1 Averaging-based Fusion
Each model outputs a probability vector \( p_i \) of size 7 (one per class).  
The ensemble computes:
\[
p_{ensemble} = \frac{1}{N} \sum_{i=1}^{N} p_i
\]
where \( N \) is the number of models in the ensemble.

### 3.2 Weighted Fusion (Optional)
In advanced setups, models may contribute unequally:
\[
p_{ensemble} = \sum_{i=1}^{N} w_i \cdot p_i
\]
where \( w_i \) is the normalized performance weight of model \( i \) based on validation AUC.

---

## 4. Implementation

The ensemble logic is implemented in [`src/ensemble.py`](../src/ensemble.py).

Example usage:
```bash
python src/ensemble.py \
  --csvs results/resnet/preds.csv \
         results/efficientnet/preds.csv \
         results/convnext/preds.csv \
  --out outputs/ensemble_metrics.json
```
