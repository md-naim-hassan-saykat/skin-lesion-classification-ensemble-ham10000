# Training Pipeline

This document describes the repository training and evaluation workflow for the
**Generalizable Ensemble Deep Learning for Skin Lesion Classification:
Internal and External Validation on HAM10000 and ISIC 2019** project.

The repository provides executable training support for seven model
architectures together with canonical seven-class evaluation and equal-weight
probability-level ensemble inference.

---

## 1. Scope

The repository supports the following model architectures:

1. Conventional CNN
2. ResNet-50
3. DenseNet-121
4. EfficientNet-B3
5. ConvNeXt-Tiny
6. MobileNetV3-Large
7. ViT-B/16

The principal implementation components are:

```text
src/config.yaml
src/data.py
src/models.py
src/train.py
src/evaluate.py
src/ensemble.py
src/utils.py
```

The current training implementation provides a reproducible executable
quick-start workflow for new experiments and methodological inspection.

It should not automatically be interpreted as an exact reconstruction of every
historical training run used to produce the archived manuscript checkpoints.
Exact historical reproduction requires the corresponding model-specific
experimental records.

---

## 2. Canonical Classes

The repository uses seven canonical lesion classes:

```text
akiec
bcc
bkl
df
mel
nv
vasc
```

The canonical order is:

```python
["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
```

Training and evaluation data organized with `torchvision.datasets.ImageFolder`
must preserve this class representation.

The training implementation validates the discovered class ordering before
training proceeds.

---

## 3. Training Data Layout

The default training data root is configured in:

```text
src/config.yaml
```

The current default is:

```text
data/HAM10000/trainval/
```

The data loader expects separate `train` and `val` directories beneath this
root.

The expected structure is:

```text
data/HAM10000/trainval/
├── train/
│   ├── akiec/
│   ├── bcc/
│   ├── bkl/
│   ├── df/
│   ├── mel/
│   ├── nv/
│   └── vasc/
└── val/
    ├── akiec/
    ├── bcc/
    ├── bkl/
    ├── df/
    ├── mel/
    ├── nv/
    └── vasc/
```

The raw HAM10000 image collection can first be prepared with:

```bash
bash scripts/prepare_ham10000.sh
```

That script prepares the complete 10,015-image HAM10000 collection.

Preparing the raw image collection does not itself construct the `train/val`
ImageFolder split required by `src/train.py`.

---

## 4. Repository Preprocessing

The executable quick-start preprocessing pipeline is implemented in:

```text
src/data.py
```

### 4.1 Training Transform

For new repository training runs, the training transform performs:

1. Resize to the configured image size
2. Random horizontal flip
3. Random rotation up to 10 degrees
4. Color jitter
5. Conversion to a PyTorch tensor
6. ImageNet normalization

The normalization parameters are:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

### 4.2 Validation Transform

Validation preprocessing performs:

1. Resize to the configured image size
2. Conversion to a PyTorch tensor
3. ImageNet normalization

Random augmentation is not applied to the validation loader.

### 4.3 Historical Preprocessing

These repository transforms are executable defaults for new or quick-start
training runs.

They should not be interpreted as evidence that every historical archived
checkpoint used exactly the same preprocessing or augmentation pipeline.

For manuscript-equivalent reconstruction, the preprocessing associated with
each historical checkpoint should be used whenever it differs from the current
repository defaults.

---

## 5. Training Workflow

At a high level, [`src/train.py`](../src/train.py) performs the following
workflow:

```text
1. Load src/config.yaml.
2. Set the configured random seed.
3. Resolve the selected model architecture.
4. Resolve the configured image size.
5. Construct training and validation data loaders.
6. Validate the canonical class ordering.
7. Initialize the selected model.
8. Compute class weights from the training dataset.
9. Initialize weighted cross-entropy loss.
10. Initialize the AdamW optimizer.
11. Initialize cosine-annealing learning-rate scheduling.
12. Train for one epoch.
13. Evaluate on the validation loader.
14. Track validation weighted F1-score.
15. Save the best-performing checkpoint.
16. Apply early stopping according to the configured patience.
17. Save the final checkpoint and training history.
```

---

## 6. Class-Imbalance Handling

HAM10000 has an imbalanced class distribution.

For the executable training workflow, class weights are calculated from the
training dataset.

For class $begin:math:text$c$end:math:text$, the initial weight is inversely related to its observed
training frequency:

$begin:math:display$
w\_c \\propto \\frac\{1\}\{n\_c\}\,
$end:math:display$

where $begin:math:text$n\_c$end:math:text$ is the number of training samples belonging to class $begin:math:text$c$end:math:text$.

The weights are normalized before being supplied to:

```python
torch.nn.CrossEntropyLoss
```

This gives less frequent classes greater contribution to the training loss than
they would receive under unweighted cross-entropy.

---

## 7. Repository Training Configuration

The quick-start defaults are maintained in:

```text
src/config.yaml
```

The current training configuration includes:

```text
Random seed:          42
Number of classes:    7
Maximum epochs:       50
Batch size:           32
Learning rate:        1e-4
Weight decay:         1e-4
Early-stop patience:  7
Pretrained weights:   enabled
```

The current executable training implementation uses:

```text
Optimizer:             AdamW
Learning-rate schedule: Cosine annealing
Loss:                  Weighted cross-entropy
Checkpoint criterion:  Validation weighted F1-score
```

These settings describe the current repository quick-start implementation.

They should not be presented as proof that all seven historical archived
checkpoints were trained using exactly the same optimizer, augmentation,
preprocessing, or other hyperparameters.

---

## 8. Train an Individual Architecture

Training requires an explicit model identifier.

For example, to train ResNet-50:

```bash
python src/train.py \
  --config src/config.yaml \
  --model resnet50
```

Supported model identifiers are:

```text
cnn
resnet50
densenet121
efficientnet_b3
convnext_tiny
mobilenet_v3_large
vit_b_16
```

To specify a different output directory:

```bash
python src/train.py \
  --config src/config.yaml \
  --model resnet50 \
  --outdir outputs/training
```

The training implementation automatically selects an available execution
device according to its supported device-selection logic.

---

## 9. Training Outputs

For a selected architecture, the training workflow writes model and training
artifacts to the configured output directory.

Typical outputs include:

```text
<model>_best.pth
<model>_last.pth
best_val_metrics.json
train_history.json
```

The best checkpoint is selected according to validation weighted F1-score.

The final checkpoint records the model state at the end of the training
workflow.

Generated experimental outputs should remain separate from the curated
manuscript artifacts maintained under `results/`.

---

## 10. Evaluate an Individual Checkpoint

Individual checkpoints can be evaluated using:

```text
src/evaluate.py
```

For example:

```bash
python src/evaluate.py \
  --model resnet50 \
  --checkpoint checkpoints/resnet50_ham10000.pth \
  --data_dir data/evaluation/HAM10000_standardized \
  --dataset ham10000 \
  --out outputs/resnet50_metrics.json \
  --save_csv outputs/resnet50_predictions.csv
```

The evaluator:

- constructs the requested architecture;
- loads the supplied checkpoint;
- validates the canonical seven-class ImageFolder ordering;
- performs inference without gradient computation;
- converts model logits to probabilities;
- applies a supplied output permutation when required;
- computes the dataset-specific evaluation metrics; and
- optionally writes per-sample probability predictions.

---

## 11. Checkpoint Compatibility

Archived checkpoints are loaded using strict state-dictionary matching.

The evaluation implementation does not silently discard unmatched learned
parameters.

If the selected model architecture does not match the supplied checkpoint,
evaluation terminates rather than reporting results from a partially loaded
model.

This protects against accidental evaluation of an incompatible checkpoint.

---

## 12. Output-Class Permutations

The canonical output order is:

```python
["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
```

If a historical checkpoint produced logits in another class order, its
probabilities must be mapped to the canonical order before metric computation
or ensemble fusion.

`src/evaluate.py` supports an explicit permutation through:

```text
--output_permutation
```

For example, the identity permutation is:

```bash
--output_permutation 0,1,2,3,4,5,6
```

Historical output permutations must be established from archived evaluation
records or checkpoint metadata and must not be guessed.

The repository's documented permutation configuration is maintained in:

```text
src/config.yaml
```

---

## 13. Batch HAM10000 Evaluation

The recommended entry point for evaluating the complete seven-model HAM10000
workflow is:

```bash
bash scripts/eval_all.sh
```

By default, this workflow expects the standardized evaluation cohort at:

```text
data/evaluation/HAM10000_standardized/
```

and the archived checkpoints under:

```text
checkpoints/
```

Generated outputs are written by default beneath:

```text
outputs/harmonized_ham10000/
```

The batch workflow requires one fixed checkpoint for each of the seven model
architectures.

---

## 14. Ensemble Evaluation

The final ensemble uses the equal-weight arithmetic mean of the seven aligned
probability outputs:

$begin:math:display$
\\mathbf\{p\}\_\{\\mathrm\{ens\}\}
\=
\\frac\{1\}\{7\}
\\sum\_\{i\=1\}\^\{7\}
\\mathbf\{p\}\_i\.
$end:math:display$

Each model therefore contributes weight:

$begin:math:display$
w\_i \= \\frac\{1\}\{7\}\.
$end:math:display$

The ensemble implementation requires exactly seven aligned prediction CSV
files.

For the complete workflow, use:

```bash
bash scripts/eval_all.sh
```

See [`ensemble_method.md`](ensemble_method.md) for the ensemble definition,
class-order requirements, and prediction-alignment checks.

---

## 15. Evaluation Metrics

The implemented metric suite depends on the evaluation dataset.

### HAM10000

The HAM10000 evaluation reports:

```text
Accuracy
Weighted F1-score
Macro one-vs-rest ROC-AUC
Micro one-vs-rest ROC-AUC
Expected Calibration Error
```

### ISIC 2019

The ISIC 2019 evaluation reports:

```text
Accuracy
Weighted F1-score
Macro one-vs-rest ROC-AUC
Weighted one-vs-rest ROC-AUC
```

See [`evaluation_metrics.md`](evaluation_metrics.md) for definitions and
implementation details.

---

## 16. Internal and External Evaluation

The repository distinguishes between:

- HAM10000 internal/harmonized evaluation; and
- ISIC 2019 external evaluation.

The standardized 2,003-image HAM10000 cohort used in the final analysis is a
retrospective harmonized evaluation cohort.

It should not be described as a universally untouched test split that was
identically held out during development of every archived model.

ISIC 2019 provides external validation evidence under dataset shift relative to
HAM10000.

---

## 17. Generated and Curated Results

Newly generated experimental outputs belong under:

```text
outputs/
```

Curated manuscript and reproducibility artifacts are maintained separately
under:

```text
results/
```

For example:

```text
results/
├── figures/
└── tables/
```

This separation prevents temporary training or evaluation outputs from being
confused with the curated research results distributed with the repository.

---

## 18. Reproducibility Considerations

The repository fixes the configured random seed for its executable workflow,
but setting a random seed alone does not guarantee bitwise-identical execution
across all hardware, software versions, devices, or parallel execution
environments.

The current training pipeline should therefore be understood as a reproducible
reference implementation rather than a guarantee of bitwise determinism across
all systems.

For manuscript-equivalent reconstruction, verify the historical configuration
associated with each archived model, including:

- dataset and split construction;
- random seed and sampling procedure;
- model architecture;
- initialization or pretrained weights;
- optimizer and learning-rate schedule;
- training duration and stopping criteria;
- loss function and class weighting;
- image preprocessing and augmentation;
- checkpoint parameters;
- canonical class ordering; and
- checkpoint-specific output permutation.

The current repository defaults should not be assumed to reproduce every
historical training run without the corresponding archived experimental
information.

---

## 19. Related Documentation

Additional details are available in:

- [`installation.md`](installation.md) — environment and dataset setup
- [`evaluation_metrics.md`](evaluation_metrics.md) — implemented metrics
- [`ensemble_method.md`](ensemble_method.md) — seven-model ensemble methodology
- [`../README.md`](../README.md) — repository overview and primary usage
