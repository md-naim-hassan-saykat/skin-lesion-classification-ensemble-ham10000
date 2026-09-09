# Evaluation Metrics

This document describes the evaluation metrics implemented in the
**Generalizable Ensemble Deep Learning for Skin Lesion Classification:
Internal and External Validation on HAM10000 and ISIC 2019** project.

The evaluation framework considers overall classification performance,
class imbalance, probability discrimination, calibration, and external
generalization.

---

## 1. Overview

Individual model checkpoints are evaluated through
[`src/evaluate.py`](../src/evaluate.py), while the equal-weight seven-model
ensemble is evaluated through [`src/ensemble.py`](../src/ensemble.py).

Shared metric computation is implemented in
[`src/utils.py`](../src/utils.py).

The canonical seven-class ordering is:

```text
akiec
bcc
bkl
df
mel
nv
vasc
```

The implemented metric suites differ slightly between HAM10000 and ISIC 2019
to match the reported evaluation protocol.

### HAM10000

The implemented HAM10000 metric suite reports:

- Accuracy
- Weighted F1-score
- Macro one-vs-rest ROC-AUC
- Micro one-vs-rest ROC-AUC
- Expected Calibration Error (ECE)

### ISIC 2019

The implemented ISIC 2019 metric suite reports:

- Accuracy
- Weighted F1-score
- Macro one-vs-rest ROC-AUC
- Weighted one-vs-rest ROC-AUC

---

## 2. Accuracy

For multiclass classification, accuracy is the proportion of evaluated samples
assigned to the correct class:

$begin:math:display$
\\mathrm\{Accuracy\}
\=
\\frac\{1\}\{n\}
\\sum\_\{i\=1\}\^\{n\}
\\mathbf\{1\}
\\left\(
\\hat\{y\}\_i \= y\_i
\\right\)\,
$end:math:display$

where:

- $begin:math:text$n$end:math:text$ is the number of evaluated samples;
- $begin:math:text$y\_i$end:math:text$ is the ground-truth class of sample $begin:math:text$i$end:math:text$;
- $begin:math:text$\\hat\{y\}\_i$end:math:text$ is the predicted class; and
- $begin:math:text$\\mathbf\{1\}\(\\cdot\)$end:math:text$ is the indicator function.

The repository computes accuracy using:

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_true, y_pred)
```

Accuracy provides an intuitive measure of overall correctness but can be
influenced by class imbalance. It is therefore interpreted together with
weighted F1-score and ROC-AUC.

---

## 3. Weighted F1-Score

For class $begin:math:text$c$end:math:text$, the F1-score is the harmonic mean of precision and recall:

$begin:math:display$
F1\_c
\=
2
\\frac\{
\\mathrm\{Precision\}\_c
\\mathrm\{Recall\}\_c
\}\{
\\mathrm\{Precision\}\_c
\+
\\mathrm\{Recall\}\_c
\}\.
$end:math:display$

The weighted multiclass F1-score is:

$begin:math:display$
F1\_\{\\mathrm\{weighted\}\}
\=
\\sum\_\{c\=1\}\^\{C\}
\\frac\{n\_c\}\{n\}
F1\_c\,
$end:math:display$

where:

- $begin:math:text$C$end:math:text$ is the number of classes;
- $begin:math:text$n\_c$end:math:text$ is the number of ground-truth samples belonging to class $begin:math:text$c$end:math:text$; and
- $begin:math:text$n$end:math:text$ is the total number of evaluated samples.

The implementation uses:

```python
from sklearn.metrics import f1_score

f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)
```

Weighting by class support provides a single precision-recall summary while
accounting for the observed class distribution.

Because weighted F1-score gives greater influence to more prevalent classes,
it should not be interpreted as a class-balanced measure by itself.

---

## 4. ROC-AUC

The area under the Receiver Operating Characteristic curve (ROC-AUC) evaluates
the ranking ability of predicted class probabilities across classification
thresholds.

For multiclass evaluation, the repository uses one-vs-rest (OvR) ROC-AUC.

### 4.1 Macro ROC-AUC

Macro ROC-AUC computes the one-vs-rest AUC independently for each class and
then gives every class equal weight:

$begin:math:display$
\\mathrm\{AUC\}\_\{\\mathrm\{macro\}\}
\=
\\frac\{1\}\{C\}
\\sum\_\{c\=1\}\^\{C\}
\\mathrm\{AUC\}\_c\.
$end:math:display$

Macro ROC-AUC is reported for both HAM10000 and ISIC 2019.

Because each class contributes equally, macro ROC-AUC is particularly useful
for examining discrimination in an imbalanced multiclass setting.

### 4.2 Micro ROC-AUC

For HAM10000, the repository additionally reports micro ROC-AUC.

The multiclass ground-truth labels are first converted to a one-vs-rest binary
representation. The class-sample decisions are then aggregated before the AUC
is calculated.

Micro ROC-AUC therefore summarizes discrimination across all individual
class-sample decisions collectively.

This metric is part of the implemented HAM10000 evaluation suite.

### 4.3 Weighted ROC-AUC

For ISIC 2019, the repository additionally reports weighted one-vs-rest
ROC-AUC:

$begin:math:display$
\\mathrm\{AUC\}\_\{\\mathrm\{weighted\}\}
\=
\\sum\_\{c\=1\}\^\{C\}
\\frac\{n\_c\}\{n\}
\\mathrm\{AUC\}\_c\.
$end:math:display$

Each class-specific AUC is weighted according to its ground-truth support.

This metric is part of the implemented ISIC 2019 evaluation suite.

---

## 5. Expected Calibration Error

Expected Calibration Error (ECE) measures the discrepancy between predictive
confidence and empirical correctness.

The repository implements maximum-confidence multiclass ECE using equal-width
confidence bins.

For sample $begin:math:text$i$end:math:text$, predictive confidence is:

$begin:math:display$
\\mathrm\{conf\}\_i
\=
\\max\_c p\_\{i\,c\}\,
$end:math:display$

and the predicted class is:

$begin:math:display$
\\hat\{y\}\_i
\=
\\operatorname\*\{arg\\\,max\}\_c p\_\{i\,c\}\.
$end:math:display$

The predictions are partitioned into $begin:math:text$M$end:math:text$ confidence bins
$begin:math:text$B\_1\, \\ldots\, B\_M$end:math:text$.

For bin $begin:math:text$B\_m$end:math:text$, empirical accuracy and mean confidence are compared. ECE is
then calculated as:

$begin:math:display$
\\mathrm\{ECE\}
\=
\\sum\_\{m\=1\}\^\{M\}
\\frac\{\|B\_m\|\}\{n\}
\\left\|
\\mathrm\{acc\}\(B\_m\)
\-
\\mathrm\{conf\}\(B\_m\)
\\right\|\.
$end:math:display$

The repository uses 15 equal-width bins by default:

```text
ece_bins = 15
```

ECE is included in the implemented HAM10000 metric suite.

A lower ECE indicates closer agreement between predictive confidence and
empirical correctness under this binning definition. ECE should be interpreted
alongside classification and discrimination metrics rather than as a standalone
measure of model quality.

---

## 6. Probability and Class-Order Requirements

Probability-based metrics require a probability matrix of shape:

```text
(N, 7)
```

where $begin:math:text$N$end:math:text$ is the number of evaluated samples and the seven columns correspond
to the canonical class order:

```python
["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
```

Incorrect class ordering can invalidate ROC-AUC, calibration, and ensemble
calculations even when the probability matrix has the expected dimensions.

The evaluation pipeline therefore validates the canonical ImageFolder class
ordering.

If a historical checkpoint used a different output ordering, its probability
columns must be mapped to the canonical order using the corresponding audited
output permutation before metrics are interpreted.

---

## 7. Dataset-Specific Metric Computation

The shared implementation in [`src/utils.py`](../src/utils.py) computes the
following core metrics for both datasets:

```text
accuracy
weighted_f1
macro_roc_auc_ovr
```

For HAM10000, it additionally computes:

```text
micro_roc_auc_ovr
ece
```

For ISIC 2019, it additionally computes:

```text
weighted_roc_auc_ovr
```

If a ROC-AUC value cannot be computed because the required class representation
is absent from the evaluated labels, the implementation records that AUC value
as unavailable rather than silently substituting another definition.

---

## 8. Diagnostic and Interpretability Outputs

The repository contains curated figures used to complement the numerical
evaluation.

Depending on the analysis artifact, these include:

- ROC curves;
- confusion matrices;
- model-comparison figures;
- calibration analyses; and
- Grad-CAM++ interpretability visualizations.

Curated manuscript and reproducibility figures are maintained under:

```text
results/figures/
```

Curated numerical tables are maintained under:

```text
results/tables/
```

These artifacts are distinct from temporary outputs generated during local
evaluation runs.

---

## 9. Statistical Analyses

The released result artifacts may additionally contain statistical analyses
used to quantify uncertainty and compare models.

These include, where applicable:

- bootstrap confidence intervals;
- McNemar comparisons;
- paired bootstrap model comparisons;
- point-difference summaries; and
- calibration summaries.

Such analyses complement the core point-estimate metrics and should be
considered when interpreting differences between individual models and the
ensemble.

---

## 10. Internal and External Evaluation

The standardized HAM10000 evaluation cohort contains 2,003 images and
represents the retrospective harmonized internal evaluation used for the final
analysis.

It should not be characterized as a universally untouched test split that was
identically held out during development of every archived model.

ISIC 2019 is used for external validation to assess generalization under a
dataset shift relative to HAM10000.

Internal and external results therefore provide complementary evidence and
should not be interpreted as interchangeable estimates of performance.

---

## 11. Interpretation

No single metric is sufficient to characterize the performance of an
imbalanced multiclass medical-image classifier.

The evaluation framework therefore considers:

- **Accuracy** for overall classification correctness;
- **Weighted F1-score** for a support-weighted precision-recall summary;
- **Macro ROC-AUC** for class-balanced discrimination;
- **Micro ROC-AUC** for the specified HAM10000 discrimination summary;
- **Weighted ROC-AUC** for the specified ISIC 2019 discrimination summary; and
- **ECE**, where reported, for probability calibration.

Model comparisons should be based on the complete metric profile, external
validation, and available uncertainty or statistical-comparison analyses rather
than on a single point estimate.
