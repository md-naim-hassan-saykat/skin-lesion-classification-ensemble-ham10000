# Ensemble Method

This document outlines the ensemble strategy used in the **Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019** project.
The ensemble integrates multiple deep learning models to enhance **classification robustness**, **calibration**, and **generalization** across datasets.

---

## 1. Overview

Multiple CNN architectures (e.g., **ResNet**, **EfficientNet**, **ConvNeXt**, **DenseNet**) are trained independently on identical training and validation splits.
Each model outputs class probability distributions (softmax scores) for the **seven skin lesion categories**.
These probabilities are combined using a late-fusion ensemble method to obtain the final consensus prediction.

---

## 2. Motivation

Individual CNNs capture complementary visual cues — combining them reduces individual biases and overfitting.

- **ResNet** extracts strong hierarchical and texture-based features.
- **EfficientNet** achieves parameter-efficient scaling with balanced capacity.
- **ConvNeXt** incorporates modern convolutional designs that capture contextual semantics.
- **DenseNet** enhances feature reuse through dense connectivity.

By aggregating their outputs, the ensemble achieves **higher stability and accuracy** across diverse skin lesion types and imaging conditions.

---

## 3. Ensemble Fusion Strategies

### 3.1 Simple Averaging (Default)
For each model \( i \), the network produces a probability vector \( p_i \in \mathbb{R}^7 \).
The ensemble computes the mean probability per class:

\[
p_{\text{ensemble}} = \frac{1}{N} \sum_{i=1}^{N} p_i
\]

where \( N \) is the total number of models in the ensemble.
The final prediction is given by \( \arg\max(p_{\text{ensemble}}) \).

### 3.2 Weighted Averaging (Optional)
In cases where certain models perform better (e.g., higher validation AUC), a weighted average can be applied:

\[
p_{\text{ensemble}} = \sum_{i=1}^{N} w_i \cdot p_i
\]

where \( w_i \) are normalized weights such that \( \sum_i w_i = 1 \).
This allows better-performing models to contribute more to the final decision.

---

## 4. Implementation

The ensemble computation and metric evaluation are implemented in
[`src/ensemble.py`](../src/ensemble.py).

Each individual model produces a CSV file containing predicted probabilities for each class.
The ensemble script averages (or optionally weights) these predictions and computes key evaluation metrics such as **accuracy**, **F1-score**, and **AUC**.

Example usage:
```bash
python src/ensemble.py \
  --csvs outputs/resnet_val_preds.csv \
         outputs/efficientnet_val_preds.csv \
         outputs/convnext_val_preds.csv \
  --out outputs/ensemble_metrics.json
```
