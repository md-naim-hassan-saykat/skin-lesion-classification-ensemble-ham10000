# Ensemble Method

This document describes the ensemble strategy implemented in the
**Generalizable Ensemble Deep Learning for Skin Lesion Classification:
Internal and External Validation on HAM10000 and ISIC 2019** project.

The final ensemble combines the probability outputs of seven independently
trained deep learning models using equal-weight probability-level late fusion.

---

## 1. Overview

The study considers seven model architectures:

1. Conventional CNN
2. ResNet-50
3. DenseNet-121
4. EfficientNet-B3
5. ConvNeXt-Tiny
6. MobileNetV3-Large
7. ViT-B/16

Each model produces a probability distribution over the seven canonical skin
lesion classes:

```text
akiec
bcc
bkl
df
mel
nv
vasc
```

Before ensemble fusion, all model predictions must correspond to the same
samples in the same order and must use the same canonical class ordering.

The final ensemble is implemented in
[`src/ensemble.py`](../src/ensemble.py).

---

## 2. Rationale

Architecturally diverse models can learn different representations of the same
dermoscopic image. Convolutional architectures capture local and hierarchical
visual features in different ways, while the Vision Transformer provides a
complementary attention-based representation.

Combining their probability outputs reduces dependence on the prediction of any
single architecture and provides a reproducible consensus prediction.

The ensemble should not be assumed to outperform every constituent model on
every metric. Individual-model and ensemble performance should therefore be
interpreted from the reported evaluation results rather than from the ensemble
construction alone.

---

## 3. Equal-Weight Probability Fusion

For model $begin:math:text$i$end:math:text$, let

$begin:math:display$
\\mathbf\{p\}\_i \=
\\left\(
p\_\{i\,1\}\,
p\_\{i\,2\}\,
\\ldots\,
p\_\{i\,7\}
\\right\)
$end:math:display$

denote its predicted probability vector over the seven canonical classes.

For the seven-model ensemble, the final probability vector is

$begin:math:display$
\\mathbf\{p\}\_\{\\mathrm\{ens\}\}
\=
\\frac\{1\}\{7\}
\\sum\_\{i\=1\}\^\{7\}
\\mathbf\{p\}\_i\.
$end:math:display$

Each model therefore contributes equal weight:

$begin:math:display$
w\_i \= \\frac\{1\}\{7\}\.
$end:math:display$

The final predicted class is

$begin:math:display$
\\hat\{y\}
\=
\\operatorname\*\{arg\\\,max\}\_\{c\}
p\_\{\\mathrm\{ens\}\,c\}\.
$end:math:display$

The final repository implementation does not use validation-performance-based
or manually assigned model weights.

---

## 4. Prediction Alignment

Probability averaging is valid only when all seven prediction files represent
the same samples in exactly the same order.

Each prediction CSV is expected to contain:

```text
sample_index
y_true
p_akiec
p_bcc
p_bkl
p_df
p_mel
p_nv
p_vasc
```

Before averaging, [`src/ensemble.py`](../src/ensemble.py) verifies that:

- exactly seven prediction CSV files are supplied;
- all files contain the same number of samples;
- `sample_index` values are aligned;
- ground-truth labels are identical across files; and
- probability outputs follow the canonical seven-class representation.

The ensemble calculation terminates with an error if these alignment
requirements are violated.

---

## 5. Canonical Class Ordering

The canonical class order used throughout the repository is:

```python
["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
```

Historical checkpoint outputs must be mapped to this order before their
probabilities are combined.

If an archived checkpoint used a different output ordering, the corresponding
output permutation must be established from the historical checkpoint metadata
or evaluation records. Such mappings must not be guessed.

See [`src/config.yaml`](../src/config.yaml) for the documented
output-permutation configuration.

---

## 6. Implementation

Individual checkpoint evaluation is performed by:

```text
src/evaluate.py
```

The ensemble calculation is performed by:

```text
src/ensemble.py
```

The complete seven-model HAM10000 evaluation workflow can be run with:

```bash
bash scripts/eval_all.sh
```

This workflow evaluates the seven fixed model checkpoints, writes individual
prediction CSV files, verifies their availability, and passes them to the
ensemble implementation.

---

## 7. Direct Ensemble Usage

If seven aligned prediction CSV files have already been generated, the ensemble
can be evaluated directly:

```bash
python src/ensemble.py \
  --csvs outputs/harmonized_ham10000/cnn_predictions.csv \
         outputs/harmonized_ham10000/resnet50_predictions.csv \
         outputs/harmonized_ham10000/densenet121_predictions.csv \
         outputs/harmonized_ham10000/efficientnet_b3_predictions.csv \
         outputs/harmonized_ham10000/convnext_tiny_predictions.csv \
         outputs/harmonized_ham10000/mobilenet_v3_large_predictions.csv \
         outputs/harmonized_ham10000/vit_b_16_predictions.csv \
  --dataset ham10000 \
  --out outputs/harmonized_ham10000/ensemble_metrics.json
```

The ensemble implementation writes:

```text
outputs/harmonized_ham10000/ensemble_metrics.json
outputs/harmonized_ham10000/ensemble_predictions.csv
```

---

## 8. Evaluation Metrics

For HAM10000, the implemented ensemble evaluation reports:

- Accuracy
- Weighted F1-score
- Macro one-vs-rest ROC-AUC
- Micro one-vs-rest ROC-AUC
- Expected Calibration Error (ECE)

For ISIC 2019, the implemented metric set reports:

- Accuracy
- Weighted F1-score
- Macro one-vs-rest ROC-AUC
- Weighted one-vs-rest ROC-AUC

See [`evaluation_metrics.md`](evaluation_metrics.md) for metric definitions and
implementation details.

---

## 9. Reproducibility Considerations

The standardized 2,003-image HAM10000 evaluation cohort used in the final
analysis is a retrospective harmonized evaluation cohort. It should not be
described as a universally untouched test split that was identically held out
during development of every archived model.

Reproducing manuscript-equivalent predictions additionally requires the exact
configuration associated with each archived checkpoint, including, where
applicable:

- checkpoint architecture;
- image preprocessing;
- canonical class mapping;
- checkpoint-specific output permutation; and
- evaluation-cohort reconstruction.

The repository therefore distinguishes between executable repository defaults
and exact historical reconstruction of the evaluation protocol used to obtain
the manuscript-reported results.
