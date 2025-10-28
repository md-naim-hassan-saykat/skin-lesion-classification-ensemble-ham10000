# Evaluation Metrics

This document outlines the evaluation strategy and performance metrics used in the **Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019** project.
The evaluation emphasizes **accuracy**, **robustness**, and **generalization** across internal (HAM10000) and external (ISIC 2019) datasets.

---

## 1. Overview

Both **individual models** and **ensemble configurations** are evaluated using a unified set of quantitative metrics.
Each model outputs per-class probabilities, and predictions are compared with the true class labels to assess overall and class-wise performance.

Evaluation logic is implemented in:
- [`src/evaluate.py`](../src/evaluate.py) — for checkpoint-level evaluation
- [`src/utils.py`](../src/utils.py) — for metric computation and logging

All evaluation steps run automatically after training or when performing ensemble fusion via [`src/ensemble.py`](../src/ensemble.py).

---

## 2. Classification Metrics

### 2.1 Accuracy
Measures the overall proportion of correctly classified samples:
\[
\text{Accuracy} = \frac{\text{TP + TN}}{\text{Total Samples}}
\]

In Python:
```python
from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)
```
