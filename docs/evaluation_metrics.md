# Evaluation Metrics

This document describes the evaluation metrics implemented in the
**Generalizable Ensemble Deep Learning for Dermoscopic Skin-Lesion
Classification: Internal Evaluation on HAM10000 and External Evaluation on
ISIC 2019** project.

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

$$
\mathrm{Accuracy}
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf{1}
\left(
\hat{y}_i = y_i
\right),
$$

where:

- $n$ is the number of evaluated samples;
- $y_i$ is the ground-truth class of sample $i$;
- $\hat{y}_i$ is the predicted class; and
- $\mathbf{1}(\cdot)$ is the indicator function.

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

For class $c$, the F1-score is the harmonic mean of precision and recall:

$$
F1_c
=
2
\frac{
\mathrm{Precision}_c
\mathrm{Recall}_c
}{
\mathrm{Precision}_c
+
\mathrm{Recall}_c
}.
$$

The weighted multiclass F1-score is:

$$
F1_{\mathrm{weighted}}
=
\sum_{c=1}^{C}
\frac{n_c}{n}
F1_c,
$$

where:

- $C$ is the number of classes;
- $n_c$ is the number of ground-truth samples belonging to class $c$; and
- $n$ is the total number of evaluated samples.

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

$$
\mathrm{AUC}_{\mathrm{macro}}
=
\frac{1}{C}
\sum_{c=1}^{C}
\mathrm{AUC}_c.
$$

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

$$
\mathrm{AUC}_{\mathrm{weighted}}
=
\sum_{c=1}^{C}
\frac{n_c}{n}
\mathrm{AUC}_c.
$$

Each class-specific AUC is weighted according to its ground-truth support.

This metric is part of the implemented ISIC 2019 evaluation suite.

---

## 5. Expected Calibration Error

Expected Calibration Error (ECE) measures the discrepancy between predictive
confidence and empirical correctness.

The repository implements maximum-confidence multiclass ECE using equal-width
confidence bins.

For sample $i$, predictive confidence is:

$$
\mathrm{conf}_i
=
\max_c p_{i,c},
$$

and the predicted class is:

$$
\hat{y}_i
=
\operatorname*{arg\,max}_c p_{i,c}.
$$

The predictions are partitioned into $M$ confidence bins
$B_1, \ldots, B_M$.

For bin $B_m$, empirical accuracy and mean confidence are compared. ECE is
then calculated as:

$$
\mathrm{ECE}
=
\sum_{m=1}^{M}
\frac{|B_m|}{n}
\left|
\mathrm{acc}(B_m)
-
\mathrm{conf}(B_m)
\right|.
$$

The repository uses 15 equal-width, right-closed bins $(lo, hi]$ (the first
bin also includes 0), which is the convention of the reported values and of
the reliability table in Supplementary Table S7:

```text
ece_bins = 15
```

Bins that contain no predictions are skipped. Because maximum confidence in a
seven-class problem is never below $1/7$, the two lowest bins are always empty; on
the HAM10000 cohort, 11 of the 15 bins are populated for the ensemble.

ECE is included in the implemented HAM10000 metric suite.

### 5.1 Direction of miscalibration

ECE measures the size of the miscalibration but not its direction. The
direction is the sign of accuracy minus mean confidence: positive values mean
the model is **under-confident**, negative values **over-confident**.

On HAM10000, five of the seven models are over-confident. The equal-weight
ensemble is **under-confident**: its accuracy (0.9386) exceeds its mean
confidence (0.8232) in every populated bin. Averaging the probabilities of
models that disagree spreads probability mass across classes; the seven models
predict the same class for only 54.9% of the cohort.

### 5.2 Post-hoc temperature scaling

Temperature scaling divides the logits by a single scalar $T$ before the
softmax. For probability outputs, the logits are recovered as
$L = \log p$, which is exact up to an additive constant. $T$ is fitted by
minimizing multiclass negative log-likelihood. $T < 1$ sharpens the
distribution (correcting under-confidence) and $T > 1$ softens it.

To avoid fitting and scoring on the same images, $T$ is estimated out of fold
with 5-fold stratified cross-fitting (seed 42): it is fitted on four folds and
applied to the fifth, and the calibrated predictions are pooled.

A single positive temperature is a monotone transformation of the scores, so
the predicted class of every image, and therefore accuracy and weighted F1,
cannot change.

| HAM10000 ensemble | ECE | Accuracy | Weighted F1 | $T$ |
|---|---:|---:|---:|---:|
| Uncalibrated | 11.54% | 0.9386 | 0.9374 | - |
| Temperature scaled (out of fold) | 1.26% | 0.9386 | 0.9374 | 0.482 |

Calibrating each model separately and then averaging gives 13.17%, worse than
the uncalibrated ensemble; calibration is effective when applied to the
ensemble output.

Implementation: `scripts/revision_analysis/calibration_analysis.py`.
Results: `results/tables/calibration_temperature_scaling.csv` and
`results/tables/HAM10000_ensemble_reliability.csv`.

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

where $N$ is the number of evaluated samples and the seven columns correspond
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
- Grad-CAM interpretability visualizations.

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

### 9.1 Bootstrap confidence intervals

Point estimates are computed once on the complete 2,003-image cohort. Each
metric is then bootstrapped separately with 1,000 resamples of the cohort,
drawn with replacement using `numpy.random.default_rng`, and the 2.5th and
97.5th percentiles form the 95% interval. The generator is seeded per metric
from a base seed of 42: accuracy 42, weighted F1 43, macro ROC-AUC 44, and
micro ROC-AUC 45. A resample on which a metric is undefined is excluded.

These are the intervals in Supplementary Table S1 and in
`results/tables/HAM10000_master_results_with_95CI.csv`, reproduced exactly by
`scripts/revision_analysis/bootstrap_ci.py`. Running all four metrics on one
shared random stream gives intervals that differ in the third or fourth
decimal place.

### 9.2 McNemar's test

Paired correctness of the ensemble and each individual model is compared with
McNemar's chi-square test with continuity correction,

$$
\chi^2 = \frac{(|b - c| - 1)^2}{b + c},
$$

where $b$ counts images the ensemble classifies correctly and the model does
not, and $c$ the converse. The test establishes whether the two differ; the
direction is read from $b$ and $c$. On HAM10000, $b > c$ for six models and
$c > b$ for ViT (47 versus 88), so that significant difference favors ViT.

### 9.3 Paired bootstrap differences

Differences (ensemble minus model) in accuracy, weighted F1 and macro ROC-AUC
are bootstrapped with 1,000 paired resamples, using deterministic model- and
metric-specific seeds derived from a base seed of 42. Intervals that exclude
zero indicate a difference in the stated direction.

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

ISIC 2019 is used for external evaluation under dataset shift relative to
HAM10000.

**ISIC 2019 contains HAM10000.** The ISIC 2019 challenge training set is
assembled from HAM10000, BCN_20000 and MSK. Identified by image identifier,
10,011 of the 25,331 ISIC 2019 images used here (39.5%) are HAM10000 images
that every archived checkpoint saw during training. Metrics on the full
25,331-image set therefore overstate cross-dataset generalization.

The external evaluation is reported in two parts:

- the full set, for continuity with earlier results; and
- the 15,320 BCN_20000 and MSK images that no model saw, which is the
  genuinely external estimate.

On the unseen remainder, accuracy is 12.1 to 17.4 points lower than on the
full set for every model. Implementation:
`scripts/revision_analysis/external_decontamination.py`. Results:
`results/tables/ISIC2019_partitioned_by_source.csv` and
`results/tables/ISIC2019_unseen_per_class_F1.csv`.

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
- **ECE**, where reported, for probability calibration, together with the
  direction of miscalibration.

Model comparisons should be based on the complete metric profile, external
validation, and available uncertainty or statistical-comparison analyses rather
than on a single point estimate.
