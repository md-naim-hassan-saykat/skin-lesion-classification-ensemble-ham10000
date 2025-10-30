---

### 2. `docs/evaluation_metrics.md` (Refined)
**Enhancements:** explicitly include calibration, per-class recall/F1, and reference to notebook visualizations.

```markdown
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
