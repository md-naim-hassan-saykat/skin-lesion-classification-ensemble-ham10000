# Ensemble Method

This document outlines the ensemble strategy implemented in the **Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019** project.
The ensemble integrates multiple deep learning architectures to improve **classification robustness**, **calibration**, and **generalization** across datasets.

---

## 1. Overview

Multiple backbone architectures (e.g., **ResNet**, **EfficientNet**, **ConvNeXt**, **DenseNet**, **ViT**) are trained independently on identical training and validation splits.
Each model outputs softmax probabilities for the **seven skin lesion categories**.
The ensemble combines these outputs using late-fusion techniques to obtain a consensus prediction.

This strategy leverages the complementary strengths of diverse feature extractors — improving reliability and minimizing variance under domain shift.

---

## 2. Motivation

Single CNNs often capture biased or limited visual cues depending on their architectural design. By aggregating multiple models:

- **ResNet** contributes hierarchical texture features.  
- **DenseNet** encourages feature reuse through dense connectivity.  
- **EfficientNet** achieves optimal scaling of width, depth, and resolution.  
- **ConvNeXt** captures long-range spatial patterns with modern convolutional blocks.  
- **ViT** learns global attention-based representations.

By fusing their outputs, the ensemble achieves **higher accuracy, stability, and calibration** across datasets (HAM10000 → ISIC 2019).

---

## 3. Ensemble Fusion Strategies

### 3.1 Simple Averaging (Default)
For each model \( i \), the network produces a class-probability vector \( p_i \in \mathbb{R}^7 \).
The ensemble computes the mean probability per class:

\[
p_{\text{ensemble}} = \frac{1}{N} \sum_{i=1}^{N} p_i
\]

where \( N \) is the total number of models.  
The final label is obtained as \( \arg\max(p_{\text{ensemble}}) \).

### 3.2 Weighted Averaging (Optional)
To emphasize higher-performing models (based on validation AUC or F1):

\[
p_{\text{ensemble}} = \sum_{i=1}^{N} w_i \cdot p_i
\]

where \( \sum_i w_i = 1 \).  
This allows models with better calibration or generalization to contribute more.

---

## 4. Implementation

The ensemble logic is implemented in [`src/ensemble.py`](../src/ensemble.py).

Each trained model saves a CSV file of predicted probabilities.
The ensemble script averages (or weights) these CSVs and computes the consolidated **Accuracy**, **F1**, **ROC-AUC**, and **ECE (Expected Calibration Error)**.

### Example Usage

```bash
cd ~/skin-lesion-classification-ensemble-ham10000
source .venv/bin/activate
export PYTHONPATH="$PWD"

python src/ensemble.py \
  --csvs outputs/resnet_val_preds.csv \
         outputs/convnext_val_preds.csv \
         outputs/efficientnet_val_preds.csv \
  --out outputs/ensemble_metrics.json
```
