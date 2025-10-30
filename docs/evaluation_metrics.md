# Evaluation Metrics

This document details the evaluation metrics used in the **Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019** project.
The evaluation emphasizes **accuracy**, **class balance**, **calibration**, and **generalization** across internal and external datasets.

---

## 1. Overview

All trained models and ensemble results are evaluated using a unified metric suite.  
Predictions are compared with ground truth labels to measure performance consistency.

Implementation:
- [`src/evaluate.py`](../src/evaluate.py) — checkpoint evaluation and report generation  
- [`src/utils.py`](../src/utils.py) — metric computation and logging  
- [`src/ensemble.py`](../src/ensemble.py) — post-fusion metric aggregation  

Results are visualized in the notebook using ROC curves, confusion matrices, and Grad-CAM overlays.

---

## 2. Classification Metrics

### 2.1 Accuracy
\[
\text{Accuracy} = \frac{\text{TP + TN}}{\text{Total Samples}}
\]

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)
```

### 2.2 F1 Score (Weighted)
Balances precision and recall, robust for class imbalance.

```python
from sklearn.metrics import f1_score
f1_score(y_true, y_pred, average="weighted")
```

### 2.3 ROC-AUC
Evaluates the area under the Receiver Operating Characteristic curve — capturing ranking quality for multi-class predictions.

```python
from sklearn.metrics import roc_auc_score
roc_auc_score(y_true, y_prob, multi_class="ovr")
```

---

## 3. Calibration Metrics
Expected Calibration Error (ECE)

Quantifies how well predicted probabilities reflect true correctness likelihoods:
[
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} , \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
]

Implemented in src/utils.py and plotted in the notebook as reliability diagrams.

---

## 4. Visual Diagnostics
Generated automatically in the notebook:
	•	ROC Curves — per class and averaged
	•	Confusion Matrices — normalized
	•	Grad-CAM Visualizations — lesion focus interpretability
	•	Calibration Plots — ECE reliability curves

---

## 5. Summary
All metrics are logged in results/metrics/*.json and summarized in LaTeX-ready tables within the notebook for reproducibility.

---
