# Evaluation Metrics

This document describes the evaluation strategy and performance metrics used in the **Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019** project.  
The evaluation focuses on assessing **accuracy, robustness, and generalization** across both internal and external datasets.

---

## 1. Overview

The models (individual and ensemble) are evaluated on multiple criteria to ensure balanced performance across all lesion categories.  
Metrics are computed using both **class-based** and **probability-based** approaches.

All evaluation scripts are located in `src/evaluate.py` and executed automatically after training or ensemble fusion.

---

## 2. Classification Metrics

### 2.1 Accuracy
The simplest measure — the fraction of correctly predicted samples:
\[
\text{Accuracy} = \frac{\text{TP + TN}}{\text{Total Samples}}
\]

In Python:
```python
from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)
```
