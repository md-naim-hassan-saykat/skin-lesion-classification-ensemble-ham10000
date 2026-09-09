# Installation Guide

This guide describes how to configure a local environment for the
**Generalizable Ensemble Deep Learning for Skin Lesion Classification:
Internal and External Validation on HAM10000 and ISIC 2019** repository.

The recommended environment uses Python 3.11 and PyTorch. The repository
provides executable training, evaluation, ensemble, testing, and
reproducibility workflows for HAM10000 and ISIC 2019.

---

## 1. System Requirements

Recommended:

- Python 3.11
- Git
- Python `venv`
- `make` for repository convenience commands
- `unzip` for HAM10000 archive preparation

GPU acceleration is not required for repository validation or unit testing.
Model training and large-scale inference can benefit substantially from
hardware acceleration such as a CUDA-capable GPU.

Apple Silicon systems may also use PyTorch's MPS backend when supported by the
installed PyTorch version and hardware.

---

## 2. Clone the Repository

Clone the repository and enter the project directory:

```bash
git clone https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000.git
cd skin-lesion-classification-ensemble-ham10000
```

---

## 3. Create a Virtual Environment

Python 3.11 is recommended for consistency with the repository configuration.

On macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Confirm the active Python version:

```bash
python --version
```

---

## 4. Install Project Dependencies

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install the main project dependencies:

```bash
python -m pip install -r requirements.txt
```

The main dependency set includes the packages required for model execution,
scientific computing, image processing, evaluation, visualization, and
interpretability.

For development, testing, linting, and repository validation, additionally
install:

```bash
python -m pip install -r requirements-ci.txt
```

Check the installed environment for dependency conflicts:

```bash
python -m pip check
```

---

## 5. Enable Pre-Commit Checks

For a development checkout, install the repository's pre-commit hooks:

```bash
pre-commit install
```

Run all configured checks once:

```bash
pre-commit run --all-files
```

The repository uses Black and Ruff together with additional checks for
configuration files, syntax, whitespace, line endings, merge conflicts, and
other repository-quality requirements.

---

## 6. Verify the Installation

The recommended full repository verification command is:

```bash
make verify
```

This runs the configured pre-commit checks followed by the test suite.

The main components can also be run separately:

```bash
make check
make test
```

or directly:

```bash
pre-commit run --all-files
pytest -q
```

A successful verification confirms that the software environment and repository
tests are functioning correctly.

It does not by itself establish exact reproduction of the
manuscript-reported results. Exact historical reproduction additionally
requires the corresponding datasets, archived checkpoints, preprocessing
configuration, class-order mapping, and evaluation-cohort reconstruction.

---

## 7. HAM10000 Data Preparation

The repository does not automatically download the HAM10000 dataset.

Obtain HAM10000 from an authorized public distribution and make the following
archives available locally:

```text
HAM10000_images_part_1.zip
HAM10000_images_part_2.zip
```

The corresponding metadata file may also be provided:

```text
HAM10000_metadata.csv
```

The repository includes a preparation script:

```bash
bash scripts/prepare_ham10000.sh
```

The script searches supported local source locations for both HAM10000 image
archives, extracts the images, consolidates them into the repository data
directory, optionally copies the metadata file when available, and verifies the
resulting image count.

A source directory can be specified explicitly:

```bash
HAM_SOURCE_DIR=/path/to/HAM10000 \
bash scripts/prepare_ham10000.sh
```

After successful preparation, the complete raw HAM10000 image collection is
stored under:

```text
data/HAM10000/images/
```

The script verifies that the prepared collection contains exactly 10,015
HAM10000 images.

---

## 8. Standardized HAM10000 Evaluation Cohort

The complete 10,015-image HAM10000 collection is distinct from the standardized
2,003-image retrospective evaluation cohort used in the final harmonized
analysis.

By default, the seven-model batch evaluation workflow expects this evaluation
cohort at:

```text
data/evaluation/HAM10000_standardized/
```

with the canonical ImageFolder structure:

```text
data/evaluation/HAM10000_standardized/
├── akiec/
├── bcc/
├── bkl/
├── df/
├── mel/
├── nv/
└── vasc/
```

The canonical class order is:

```python
["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
```

A different evaluation directory can be supplied through the
`HAM_EVAL_DIR` environment variable.

For example:

```bash
HAM_EVAL_DIR=/path/to/HAM10000_standardized \
bash scripts/eval_all.sh
```

---

## 9. Model Checkpoints

The seven-model evaluation workflow requires one fixed checkpoint for each
architecture:

1. Conventional CNN
2. ResNet-50
3. DenseNet-121
4. EfficientNet-B3
5. ConvNeXt-Tiny
6. MobileNetV3-Large
7. ViT-B/16

By default, the batch evaluation script searches for checkpoints under:

```text
checkpoints/
```

A different checkpoint directory can be supplied with:

```bash
CHECKPOINT_DIR=/path/to/checkpoints \
bash scripts/eval_all.sh
```

Datasets and large model checkpoints are not treated as ordinary repository
source files.

For manuscript-equivalent evaluation, each checkpoint must correspond to the
intended architecture and historical evaluation configuration.

---

## 10. Run the Seven-Model HAM10000 Evaluation

After the standardized evaluation cohort and required checkpoints are
available, run:

```bash
bash scripts/eval_all.sh
```

The workflow evaluates the seven individual models, generates aligned
probability predictions, and constructs the equal-weight seven-model ensemble.

The default generated output directory is:

```text
outputs/harmonized_ham10000/
```

The main workflow settings can be overridden through environment variables:

```bash
HAM_EVAL_DIR=/path/to/HAM10000_standardized \
CHECKPOINT_DIR=/path/to/checkpoints \
OUT_DIR=/path/to/output \
PYTHON=/path/to/python \
bash scripts/eval_all.sh
```

See [`ensemble_method.md`](ensemble_method.md) for details of the ensemble
calculation and prediction-alignment requirements.

---

## 11. Training

The repository also provides an executable training pipeline through:

```text
src/train.py
```

Training requires an explicit model identifier.

For example:

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

The current repository training configuration is intended as an executable
quick-start and inspection workflow. It should not automatically be interpreted
as the exact historical configuration used to train every archived checkpoint.

See [`training_pipeline.md`](training_pipeline.md) for additional details.

---

## 12. Optional Notebook and Google Colab

The repository includes an exploratory notebook under:

```text
notebooks/
```

When the corresponding notebook is compatible with the current repository
state, it may be opened in Google Colab for interactive exploration.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/blob/main/notebooks/skin_lesion_ensemble_classification.ipynb)

The notebook should be treated as an interactive research artifact rather than
as the authoritative reproducibility entry point.

For repository validation and scripted evaluation, prefer the maintained
command-line workflows documented above.

---

## 13. Generated and Curated Outputs

Temporary or newly generated experimental outputs should be written under:

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

This distinction prevents newly generated experimental files from being
confused with the curated results distributed with the repository.

---

## 14. Reproducibility Considerations

The repository provides executable source code, configuration, tests,
documentation, and curated research artifacts to support transparent inspection
and reproducibility.

However, exact reproduction of manuscript-reported predictions requires the
historical configuration associated with the corresponding archived
checkpoints, including, where applicable:

- dataset and evaluation-cohort construction;
- checkpoint architecture and parameters;
- image preprocessing;
- sample ordering;
- canonical class ordering;
- checkpoint-specific output permutation; and
- ensemble membership and weighting.

The standardized 2,003-image HAM10000 evaluation cohort is a retrospective
harmonized evaluation cohort. It should not be described as a universally
untouched test split that was identically held out during development of every
historical model.

Repository defaults therefore provide an executable reference configuration but
should not be assumed to reproduce every historical training or evaluation run
without the corresponding archived experimental information.

---

## 15. Related Documentation

Additional methodological and reproducibility details are available in:

- [`training_pipeline.md`](training_pipeline.md) — training and checkpoint workflow
- [`evaluation_metrics.md`](evaluation_metrics.md) — implemented evaluation metrics
- [`ensemble_method.md`](ensemble_method.md) — seven-model probability ensemble
- [`../README.md`](../README.md) — repository overview and usage
