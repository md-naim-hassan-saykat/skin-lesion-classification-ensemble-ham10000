# Generalizable Ensemble Deep Learning for Dermoscopic Skin-Lesion Classification

**Internal Evaluation on HAM10000 and External Evaluation on ISIC 2019**

[![Build](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Linter: Ruff](https://img.shields.io/badge/linter-ruff-informational)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17390952.svg)](https://doi.org/10.5281/zenodo.17390952)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/blob/main/notebooks/skin_lesion_ensemble_classification.ipynb)

---

A reproducible evaluation framework for dermoscopic skin-lesion classification using seven convolutional and transformer-based architectures and an equal-weight probability ensemble, with retrospective internal evaluation on HAM10000 and external cross-dataset evaluation on ISIC 2019.

The study evaluates predictive performance, cross-dataset generalization, probabilistic calibration, paired statistical differences, and qualitative Grad-CAM attribution within a harmonized seven-class evaluation framework.

> **Research-use notice:** This repository is intended for research and reproducibility purposes. The models and outputs have not been clinically validated and must not be used as a substitute for professional medical diagnosis or clinical decision-making.

---

## Highlights

- **Seven architectures:** custom CNN, ResNet-50, DenseNet-121, EfficientNet-B3, ConvNeXt-Tiny, MobileNetV3-Large, and ViT-B/16.
- **Seven-model ensemble:** equal-weight probability averaging across all seven models (`1/7` per model), with no validation-based optimization of ensemble weights.
- **HAM10000 evaluation:** standardized retrospective evaluation cohort of **2,003 images** across seven diagnostic classes.
- **External evaluation:** harmonized **ISIC 2019 cohort of 25,331 images** mapped to the same seven-class label space.
- **Strongest internal model:** ViT-B/16, with **0.9591 accuracy**, **0.9585 weighted F1**, **0.9959 macro ROC-AUC**, and **0.9978 micro ROC-AUC**.
- **Strongest external model:** ConvNeXt-Tiny, with **0.6963 accuracy**, **0.6755 weighted F1**, **0.9021 macro ROC-AUC**, and **0.9053 weighted ROC-AUC**.
- **Ensemble performance:** competitive internally and externally, but it does **not** uniformly outperform the strongest individual architecture.
- **Calibration:** ViT achieved the lowest HAM10000 ECE (**1.37%**), whereas the equal-weight ensemble had an ECE of **11.54%**.
- **Statistical analysis:** McNemar's test and paired bootstrap comparisons quantify differences between the ensemble and individual models.
- **Interpretability:** Grad-CAM provides qualitative comparisons of spatial attribution patterns; no claim of clinically validated localization is made.
- **Reproducibility:** code, trained model checkpoints, prediction outputs, figures, tables, and evaluation artifacts are released through GitHub and Zenodo.

---

## Table of Contents

- [Study Overview](#study-overview)
- [Datasets](#datasets)
- [Diagnostic Classes](#diagnostic-classes)
- [Models](#models)
- [Evaluation Protocol](#evaluation-protocol)
- [Ensemble Strategy](#ensemble-strategy)
- [Results](#results)
  - [HAM10000 Internal Evaluation](#ham10000-internal-evaluation)
  - [ISIC 2019 External Evaluation](#isic-2019-external-evaluation)
  - [Calibration](#calibration)
  - [Statistical Comparisons](#statistical-comparisons)
  - [External Per-Class Performance](#external-per-class-performance)
- [Explainability](#explainability)
- [Reproducibility](#reproducibility)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Running the Repository](#running-the-repository)
- [Notebook](#notebook)
- [Repository Structure](#repository-structure)
- [Limitations](#limitations)
- [Ethics and Data Use](#ethics-and-data-use)
- [Reproducibility Assets](#reproducibility-assets)
- [Citation](#citation)
- [Development and Contributions](#development-and-contributions)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Study Overview

This project investigates the performance and generalizability of deep learning models for seven-class dermoscopic skin-lesion classification.

Seven fixed archived HAM10000-trained model checkpoints are evaluated using architecture-specific preprocessing and a common canonical output ordering. Their predictions are also combined using an equal-weight probability ensemble.

The evaluation has two complementary components:

1. **HAM10000:** retrospective harmonized comparison on a standardized 2,003-image evaluation cohort.
2. **ISIC 2019:** external cross-dataset evaluation on a harmonized 25,331-image cohort.

The framework evaluates multiple dimensions of model behavior rather than relying on accuracy alone:

- accuracy;
- weighted F1 score;
- macro ROC-AUC;
- micro ROC-AUC on HAM10000;
- weighted ROC-AUC on ISIC 2019;
- bootstrap confidence intervals;
- paired McNemar comparisons;
- paired bootstrap performance differences;
- Expected Calibration Error (ECE);
- confusion matrices;
- classwise ROC analysis;
- qualitative Grad-CAM attribution.

A central finding is that **model ranking changes under cross-dataset evaluation**. ViT provides the strongest internal point estimates, whereas ConvNeXt-Tiny provides the strongest external point estimates.

---

## Datasets

### HAM10000

The **HAM10000** dataset contains 10,015 dermoscopic images representing seven common pigmented skin-lesion categories.

HAM10000 is used for model training and retrospective internal evaluation.

The final harmonized internal analysis uses a standardized evaluation cohort containing:

**n = 2,003 images**

Because the archived checkpoints were originally generated under differing historical training and data-partitioning pipelines, this cohort should be interpreted as a **standardized retrospective evaluation cohort**, not as a universally untouched test set shared by every model.

Dataset:

- HAM10000 / Human Against Machine with 10,000 training images
- ISIC Archive / associated public dataset distribution

### ISIC 2019

ISIC 2019 is used for external cross-dataset evaluation.

For compatibility with the HAM10000 seven-class label space:

- `AK` and `SCC` are mapped to `akiec`;
- only samples corresponding to the seven canonical diagnostic classes are retained.

The resulting harmonized external evaluation cohort contains:

**n = 25,331 images**

This external evaluation is intended to assess model behavior under dataset shift rather than to represent prospective clinical validation.

---

## Diagnostic Classes

All downstream analyses use the following canonical seven-class ordering:

| Code | Diagnostic category |
|---|---|
| `akiec` | Actinic keratoses and intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

Historical classifier outputs are mapped to this canonical ordering before metric calculation and ensembling.

This harmonization is particularly important for archived models whose original classifier ordering differed from the standardized evaluation ordering.

---

## Models

Seven architectures spanning conventional CNNs, modern convolutional networks, efficient architectures, and transformers are evaluated.

| Model | Architecture family |
|---|---|
| CNN | Custom convolutional neural network |
| ResNet-50 | Residual CNN |
| DenseNet-121 | Densely connected CNN |
| EfficientNet-B3 | Compound-scaled CNN |
| ConvNeXt-Tiny | Modern convolutional architecture |
| MobileNetV3-Large | Efficient/mobile CNN |
| ViT-B/16 | Vision Transformer |

### Custom CNN

The baseline CNN uses four convolutional blocks based on:

`Conv → BatchNorm → ReLU → MaxPool`

followed by a fully connected seven-class classifier.

### Transfer-learning architectures

The remaining architectures use their corresponding archived training configurations and architecture-specific preprocessing pipelines.

---

## Evaluation Protocol

The study uses **fixed archived model checkpoints**.

The historical checkpoints were not all generated from an identical original training split. Accordingly, the purpose of the current evaluation framework is to provide a harmonized retrospective comparison using:

- fixed checkpoint versions;
- fixed evaluation datasets;
- architecture-specific preprocessing;
- canonical seven-class output harmonization;
- common downstream metric calculation;
- reproducible prediction and analysis artifacts.

For HAM10000, all final quantitative analyses are derived from the same standardized 2,003-image evaluation cohort and harmonized prediction outputs.

These outputs are used for:

- performance metrics;
- bootstrap confidence intervals;
- Expected Calibration Error;
- McNemar comparisons;
- paired bootstrap differences;
- confusion matrices;
- ROC analyses.

For ISIC 2019, the same fixed checkpoint versions are evaluated on the same harmonized external dataset using their corresponding preprocessing pipelines and the common seven-class output ordering.

### Primary metrics

HAM10000:

- Accuracy
- Weighted F1
- Macro one-vs-rest ROC-AUC
- Micro one-vs-rest ROC-AUC

ISIC 2019:

- Accuracy
- Weighted F1
- Macro one-vs-rest ROC-AUC
- Weighted one-vs-rest ROC-AUC

### Uncertainty analysis

For the HAM10000 supplementary uncertainty analysis:

- point estimates are calculated on the complete 2,003-image cohort;
- **1,000 bootstrap resamples** are used;
- **95% bootstrap percentile confidence intervals** are reported;
- the base random seed is **42**.

### Calibration

Expected Calibration Error is evaluated on HAM10000 using:

- maximum-confidence multiclass calibration;
- **15 confidence bins**.

Lower ECE indicates closer agreement between predictive confidence and observed accuracy.

---

## Ensemble Strategy

The final ensemble combines all seven evaluated models:

1. CNN
2. ResNet-50
3. DenseNet-121
4. EfficientNet-B3
5. ConvNeXt-Tiny
6. MobileNetV3-Large
7. ViT-B/16

For each image, each model produces a seven-class probability vector.

The ensemble prediction is:

```text
p_ensemble = (p_CNN
            + p_ResNet50
            + p_DenseNet121
            + p_EfficientNetB3
            + p_ConvNeXtTiny
            + p_MobileNetV3L
            + p_ViT) / 7
```

Equivalently,

```text
p_ensemble = (1/7) × Σ p_m
```

No validation-based optimization of ensemble weights is used in the reported final analysis.

---

# Results

## HAM10000 Internal Evaluation

Performance on the standardized **2,003-image HAM10000 evaluation cohort**:

| Model | Accuracy | Weighted F1 | Macro ROC-AUC | Micro ROC-AUC |
|---|---:|---:|---:|---:|
| **ViT** | **0.9591** | **0.9585** | **0.9959** | **0.9978** |
| **Ensemble** | **0.9386** | **0.9374** | **0.9951** | **0.9971** |
| ConvNeXt-Tiny | 0.9126 | 0.9124 | 0.9882 | 0.9937 |
| EfficientNet-B3 | 0.8997 | 0.8984 | 0.9818 | 0.9905 |
| DenseNet-121 | 0.8787 | 0.8742 | 0.9786 | 0.9892 |
| MobileNetV3-L | 0.8587 | 0.8551 | 0.9720 | 0.9857 |
| CNN | 0.7898 | 0.7763 | 0.9388 | 0.9712 |
| ResNet-50 | 0.6850 | 0.7142 | 0.9319 | 0.9386 |

**ViT achieved the highest point estimate across all four reported HAM10000 performance metrics.**

The equal-weight ensemble achieved the second-highest point estimates across all four metrics. Therefore, the final results do **not** support a claim that the ensemble uniformly outperforms the strongest individual architecture.

### 95% bootstrap confidence intervals

| Model | Accuracy (95% CI) | Weighted F1 (95% CI) | Macro AUC (95% CI) | Micro AUC (95% CI) |
|---|---|---|---|---|
| ViT | 0.9591 [0.9506, 0.9666] | 0.9585 [0.9484, 0.9676] | 0.9959 [0.9927, 0.9978] | 0.9978 [0.9968, 0.9987] |
| Ensemble | 0.9386 [0.9281, 0.9491] | 0.9374 [0.9263, 0.9483] | 0.9951 [0.9937, 0.9963] | 0.9971 [0.9963, 0.9978] |
| ConvNeXt-Tiny | 0.9126 [0.9006, 0.9246] | 0.9124 [0.9005, 0.9253] | 0.9882 [0.9834, 0.9916] | 0.9937 [0.9918, 0.9951] |
| EfficientNet-B3 | 0.8997 [0.8862, 0.9121] | 0.8984 [0.8849, 0.9104] | 0.9818 [0.9775, 0.9855] | 0.9905 [0.9884, 0.9924] |
| DenseNet-121 | 0.8787 [0.8642, 0.8932] | 0.8742 [0.8598, 0.8889] | 0.9786 [0.9733, 0.9832] | 0.9892 [0.9869, 0.9911] |
| MobileNetV3-L | 0.8587 [0.8432, 0.8742] | 0.8551 [0.8385, 0.8719] | 0.9720 [0.9670, 0.9766] | 0.9857 [0.9833, 0.9881] |
| CNN | 0.7898 [0.7718, 0.8083] | 0.7763 [0.7569, 0.7949] | 0.9388 [0.9301, 0.9470] | 0.9712 [0.9672, 0.9747] |
| ResNet-50 | 0.6850 [0.6625, 0.7050] | 0.7142 [0.6962, 0.7320] | 0.9319 [0.9226, 0.9399] | 0.9386 [0.9321, 0.9446] |

---

## ISIC 2019 External Evaluation

The same fixed HAM10000-trained models are evaluated on the harmonized **25,331-image ISIC 2019 external evaluation set**.

| Model | Accuracy | Weighted F1 | Macro ROC-AUC | Weighted ROC-AUC |
|---|---:|---:|---:|---:|
| CNN | 0.5816 | 0.5331 | 0.7594 | 0.7744 |
| ResNet-50 | 0.4969 | 0.5198 | 0.8116 | 0.8117 |
| DenseNet-121 | 0.6842 | 0.6538 | 0.8883 | 0.8873 |
| EfficientNet-B3 | 0.6344 | 0.5774 | 0.8730 | 0.8770 |
| **ConvNeXt-Tiny** | **0.6963** | **0.6755** | **0.9021** | **0.9053** |
| MobileNetV3-L | 0.6430 | 0.6017 | 0.8592 | 0.8570 |
| ViT | 0.6651 | 0.6239 | 0.8995 | 0.8962 |
| Ensemble | 0.6830 | 0.6446 | 0.8981 | 0.9011 |

**ConvNeXt-Tiny achieved the highest point estimate across all four external evaluation metrics.**

The change in ranking relative to HAM10000 illustrates substantial architecture-dependent sensitivity to cross-dataset domain shift.

ViT, despite providing the strongest internal results, does not retain the top external ranking.

---

## Calibration

Expected Calibration Error on the standardized HAM10000 evaluation cohort:

| Model | ECE (%) |
|---|---:|
| **ViT** | **1.37** |
| CNN | 3.13 |
| EfficientNet-B3 | 4.46 |
| ConvNeXt-Tiny | 5.47 |
| MobileNetV3-L | 6.66 |
| DenseNet-121 | 6.96 |
| ResNet-50 | 9.33 |
| Ensemble | 11.54 |

ViT achieved the lowest ECE (**1.37%**).

The ensemble exhibited the highest ECE (**11.54%**), demonstrating that strong classification and discrimination performance does not necessarily imply well-calibrated predictive probabilities.

Explicit ensemble-level calibration should therefore be investigated before probability estimates are considered for downstream clinical interpretation.

---

## Statistical Comparisons

### McNemar's test

Paired classification outcomes between the equal-weight ensemble and each individual model were compared on HAM10000.

| Comparison | b | c | p-value |
|---|---:|---:|---:|
| Ensemble vs CNN | 322 | 24 | 2.176 × 10⁻⁵⁷ |
| Ensemble vs ResNet-50 | 540 | 32 | 9.810 × 10⁻¹⁰⁰ |
| Ensemble vs DenseNet-121 | 137 | 17 | 8.869 × 10⁻²² |
| Ensemble vs EfficientNet-B3 | 100 | 22 | 3.141 × 10⁻¹² |
| Ensemble vs ConvNeXt-Tiny | 84 | 32 | 2.188 × 10⁻⁶ |
| Ensemble vs MobileNetV3-L | 175 | 15 | 8.781 × 10⁻³¹ |
| Ensemble vs ViT | 47 | 88 | 5.760 × 10⁻⁴ |

Here:

- `b > c` favors the ensemble;
- `c > b` favors the individual model.

The ensemble therefore shows favorable paired classification differences relative to CNN, ResNet-50, DenseNet-121, EfficientNet-B3, ConvNeXt-Tiny, and MobileNetV3-L.

The comparison with **ViT favors ViT**, not the ensemble.

### Paired bootstrap: Ensemble minus ViT

| Metric | Δ Estimate | 95% CI |
|---|---:|---|
| Accuracy | -0.0205 | [-0.0320, -0.0090] |
| Weighted F1 | -0.0211 | [-0.0327, -0.0092] |
| Macro ROC-AUC | -0.0008 | [-0.0029, 0.0020] |

Negative values favor ViT.

The accuracy and weighted F1 confidence intervals remain below zero, whereas the small macro ROC-AUC difference includes zero.

---

## External Per-Class Performance

Per-class F1 scores on the harmonized ISIC 2019 external evaluation set:

| Model | AKIEC | BCC | BKL | DF | MEL | NV | VASC |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.13 | 0.32 | 0.32 | 0.00 | 0.32 | 0.76 | 0.56 |
| ResNet-50 | 0.24 | 0.37 | 0.31 | 0.15 | 0.27 | 0.73 | 0.26 |
| DenseNet-121 | 0.34 | **0.46** | 0.49 | 0.43 | 0.52 | 0.82 | 0.71 |
| EfficientNet-B3 | 0.33 | 0.38 | 0.40 | 0.23 | 0.37 | 0.77 | 0.74 |
| **ConvNeXt-Tiny** | **0.37** | 0.44 | **0.52** | **0.47** | **0.53** | **0.85** | **0.75** |
| MobileNetV3-L | 0.23 | 0.38 | 0.43 | 0.34 | 0.44 | 0.80 | 0.68 |
| ViT | 0.19 | 0.43 | 0.46 | 0.39 | 0.48 | 0.81 | 0.74 |

ConvNeXt-Tiny achieved the highest reported F1 for **AKIEC, BKL, DF, MEL, NV, and VASC**, while DenseNet-121 achieved the highest F1 for **BCC**.

### Ensemble classwise ROC-AUC on ISIC 2019

| Class | ROC-AUC |
|---|---:|
| AKIEC | 0.834 |
| BCC | 0.898 |
| BKL | 0.878 |
| DF | 0.920 |
| MEL | 0.843 |
| NV | 0.933 |
| VASC | 0.980 |

The ensemble's highest classwise external AUC was observed for **VASC (0.980)**, followed by **NV (0.933)** and **DF (0.920)**.

---

## Explainability

Grad-CAM is used to qualitatively examine spatial attribution patterns associated with model predictions.

For the ensemble:

1. Grad-CAM maps are generated independently from the seven component models.
2. Each model uses its corresponding preprocessing pipeline and canonical class mapping.
3. Individual attribution maps are independently normalized.
4. Maps are resized to a common spatial resolution.
5. The seven normalized maps are combined using equal-weight averaging.

The attribution analyses use the same fixed archived checkpoints as the quantitative evaluation.

Representative visualizations include:

- per-class HAM10000 ensemble Grad-CAM examples;
- same-image melanoma comparisons across CNN, ResNet-50, DenseNet-121, ViT, and the ensemble.

These visualizations demonstrate differences in spatial attribution patterns across architectures.

> **Important:** Grad-CAM results are qualitative interpretability aids. They were not evaluated against dermatologist annotations or localization ground truth and should not be interpreted as evidence of causal attribution, clinically validated lesion localization, or superior clinical explainability.

---

## Reproducibility

The repository is designed to support independent verification of the reported analyses.

The final evaluation framework is based on:

- fixed archived model checkpoints;
- documented architecture-specific preprocessing;
- standardized HAM10000 evaluation cohort;
- harmonized ISIC 2019 external evaluation set;
- canonical seven-class output mapping;
- harmonized prediction outputs;
- standardized downstream metric calculation;
- released statistical and visualization artifacts.

### Important reproducibility note

The seven archived checkpoints were produced under **different historical training and data-partitioning pipelines**.

Therefore, reproducibility in this project means reproducing the evaluation of the fixed archived models using the documented datasets, preprocessing procedures, class mappings, prediction outputs, and downstream analysis.

It does **not** imply that all seven architectures were originally trained prospectively using the same train/validation/test split.

This distinction is essential when interpreting the internal HAM10000 comparison.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000.git
cd skin-lesion-classification-ensemble-ham10000
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For development and CI dependencies:

```bash
pip install -r requirements-ci.txt
```

---

## Data Preparation

### HAM10000

Download HAM10000 from its public distribution or the ISIC resources.

Common distributions contain:

```text
HAM10000_images_part_1.zip
HAM10000_images_part_2.zip
HAM10000_metadata.csv
```

After downloading, extract both image archives into an appropriate local data directory.

Example:

```text
data/
└── HAM10000/
    ├── HAM10000_metadata.csv
    └── images/
        ├── ISIC_0024306.jpg
        ├── ISIC_0024307.jpg
        └── ...
```

The complete HAM10000 dataset contains **10,015 dermoscopic images**.

### ISIC 2019

For external evaluation, obtain the public ISIC 2019 images and ground-truth labels.

The external evaluation used in the study retains samples mapping to:

```text
akiec
bcc
bkl
df
mel
nv
vasc
```

with ISIC 2019 `AK` and `SCC` mapped to the unified `akiec` category.

After harmonization, the external evaluation cohort contains **25,331 images**.

> Dataset files are not redistributed through this repository. Users are responsible for obtaining the datasets from their official/public sources and complying with their respective licenses and terms of use.

---

## Running the Repository

The repository contains training, evaluation, ensembling, and analysis utilities for reproducing the project workflow.

Set the repository root on `PYTHONPATH` when required:

```bash
export PYTHONPATH="$PWD"
```

The exact commands depend on whether you are:

- training a model;
- evaluating a fixed checkpoint;
- generating prediction outputs;
- calculating ensemble predictions;
- reproducing statistical analyses;
- generating ROC curves or confusion matrices;
- generating Grad-CAM attribution maps.

For evaluation of archived checkpoints, ensure that:

1. the expected checkpoint files are available;
2. dataset paths are configured correctly;
3. architecture-specific preprocessing is preserved;
4. model outputs are converted to the canonical seven-class ordering before ensembling or metric calculation.

The final manuscript results should be reproduced from the **fixed checkpoint and harmonized evaluation artifacts** associated with the reported analysis rather than by substituting alternative checkpoints or post-hoc tuned predictions.

---

## Notebook

The main research notebook is available at:

[`notebooks/skin_lesion_ensemble_classification.ipynb`](notebooks/skin_lesion_ensemble_classification.ipynb)

It can be used in:

- a local Python environment;
- Kaggle;
- Google Colab.

### Local

Install dependencies:

```bash
pip install -r requirements.txt
```

Then configure dataset and checkpoint paths for your local environment.

### Kaggle

Kaggle provides GPU-enabled notebook environments and convenient dataset mounting.

Attach the required HAM10000 and/or ISIC 2019 datasets and update paths where necessary before running the notebook.

### Google Colab

Install additional packages if required:

```bash
!pip install timm einops grad-cam
```

If using Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then configure the dataset and checkpoint paths for the Colab environment.

> Paths are environment-specific. Reproducing the final reported results requires the same fixed checkpoints, evaluation-cohort definitions, preprocessing procedures, and class mappings used in the released evaluation artifacts.

---

## Repository Structure

A simplified overview of the project structure is shown below:

```text
skin-lesion-classification-ensemble-ham10000/
│
├── .github/                      # GitHub workflows and repository configuration
├── notebooks/                    # Research and demonstration notebooks
├── outputs/                      # Evaluation and reproducibility outputs
├── results/                      # Tables, figures, and analysis results
│   ├── figures/
│   └── tables/
├── scripts/                      # Evaluation and automation scripts
├── src/                          # Core source code
│   ├── train.py
│   ├── evaluate.py
│   ├── ensemble.py
│   ├── utils.py
│   └── config.yaml
├── tests/                        # Automated tests
│
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── LICENSE
├── Makefile
├── requirements.txt
├── requirements-ci.txt
└── README.md
```

The exact repository contents may evolve as reproducibility artifacts are consolidated.

---

## Limitations

Several limitations should be considered when interpreting the reported results.

### 1. Retrospective internal harmonization

The archived model checkpoints were produced under differing historical training and data-partitioning pipelines.

The standardized 2,003-image HAM10000 cohort therefore represents a **retrospective harmonized comparison cohort**, not a common untouched test split for every model.

### 2. External domain shift

Performance changes substantially for several architectures on ISIC 2019.

Differences between HAM10000 and ISIC 2019 may include:

- acquisition conditions;
- class distributions;
- lesion appearance;
- institutional sources;
- dataset-specific characteristics.

The current experiments do not isolate the causal contribution of individual sources of domain shift.

### 3. Calibration

The ensemble achieved strong discrimination metrics but had an ECE of **11.54%** on HAM10000.

Independent calibration assessment and explicit calibration procedures would be required before probabilistic outputs could be considered for clinical decision support.

### 4. Interpretability

Grad-CAM analysis was performed qualitatively on representative HAM10000 examples.

External ISIC 2019 Grad-CAM analysis was not performed as part of the reported study.

### 5. No dermatologist validation

Predictions and attribution maps were not evaluated by dermatologists.

The results therefore do not establish clinical diagnostic utility or clinically meaningful localization.

### 6. Additional external validation

Evaluation on additional independent institutions, demographic subgroups, and prospective clinical datasets is required before conclusions about real-world clinical generalizability can be made.

---

## Ethics and Data Use

This study uses only publicly available, de-identified dermoscopic datasets:

- HAM10000
- ISIC 2019

No new human participants, identifiable patient information, or animal subjects were involved.

Accordingly, institutional ethics approval and informed consent were not required for this analysis.

Nevertheless, dataset composition, class imbalance, demographic representation, fairness, calibration, and cross-domain generalization remain important considerations for future clinical translation.

This repository contains no newly collected identifiable patient information.

---

## Reproducibility Assets

Due to repository storage constraints, larger reproducibility assets are archived through Zenodo.

The released resources include, where applicable:

- trained model checkpoints;
- prediction outputs;
- evaluation metrics;
- statistical analysis artifacts;
- ROC analyses;
- confusion matrices;
- Grad-CAM visualizations;
- tables and figures;
- supporting reproducibility files.

**Zenodo DOI:**  
[10.5281/zenodo.17390952](https://doi.org/10.5281/zenodo.17390952)

The GitHub repository and Zenodo archive together provide the code and artifacts required for independent verification and further methodological investigation.

---

## Paper / Preprint

**Title:**  
*Generalizable Ensemble Deep Learning for Dermoscopic Skin-Lesion Classification: Internal Evaluation on HAM10000 and External Evaluation on ISIC 2019*

**Author:**  
Md Naim Hassan Saykat

**Affiliation:**  
Department of Computer Science, Université Paris-Saclay, France

**Associated archive:**  
[Zenodo DOI: 10.5281/zenodo.17390952](https://doi.org/10.5281/zenodo.17390952)

---

## Citation

If you use this repository, evaluation framework, models, or associated artifacts, please cite the accompanying work and Zenodo archive.

```bibtex
@misc{saykat2025dermoscopic,
  author    = {Saykat, Md Naim Hassan},
  title     = {Generalizable Ensemble Deep Learning for Dermoscopic Skin-Lesion Classification: Internal Evaluation on HAM10000 and External Evaluation on ISIC 2019},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17390952},
  url       = {https://doi.org/10.5281/zenodo.17390952},
  note      = {Preprint, code, models, and accompanying reproducibility artifacts}
}
```

A machine-readable citation file is also provided in [`CITATION.cff`](CITATION.cff).

---

## Development and Contributions

Contributions that improve reproducibility, documentation, evaluation, or code quality are welcome.

For development:

```bash
git clone https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000.git
cd skin-lesion-classification-ensemble-ham10000

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt -r requirements-ci.txt

ruff check src
black --check src
make test
```

Before submitting changes, please review:

- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)

For bugs, reproducibility problems, or feature requests, please open a GitHub issue.

---

## Environment

The project is designed for modern Python/PyTorch environments.

| Component | Supported / used environment |
|---|---|
| Python | 3.9+ |
| PyTorch | 2.x |
| Torchvision | compatible with installed PyTorch version |
| OS | Linux / macOS |
| Accelerator | CUDA / MPS where supported |
| CPU | supported for evaluation, but slower |

For exact dependency versions, refer to:

- [`requirements.txt`](requirements.txt)
- [`requirements-ci.txt`](requirements-ci.txt)

---

## Future Work

Important directions for extending this work include:

- prospectively training all architectures using one predefined train/validation/test protocol;
- evaluating additional independent clinical datasets;
- investigating domain adaptation;
- evaluating ensemble-level calibration;
- extending attribution analysis to external datasets;
- investigating self-supervised and semi-supervised learning;
- incorporating multimodal clinical information;
- performing subgroup and fairness analyses;
- conducting dermatologist reader studies;
- evaluating predictions and attribution maps against clinically meaningful reference standards.

---

## Keywords

Dermoscopic Image Analysis · Skin-Lesion Classification · HAM10000 · ISIC 2019 · Deep Learning · Ensemble Learning · Vision Transformer · ConvNeXt · External Evaluation · Domain Shift · Model Calibration · Grad-CAM · Explainable AI · Medical Imaging · Reproducibility

---

## License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

Dataset licenses and terms remain governed by the respective dataset providers.

---

## Acknowledgements

This research was conducted independently as part of the Master's programme in Artificial Intelligence at **Université Paris-Saclay**.

The author's academic studies were supported by the **French Government Scholarship (BGF — Bourses du Gouvernement Français)**. The scholarship did not specifically fund the design, implementation, or reporting of this research.

Computational experiments were conducted using publicly accessible **Kaggle cloud infrastructure**.

The author also acknowledges:

- the creators and maintainers of HAM10000;
- the International Skin Imaging Collaboration (ISIC);
- the developers and maintainers of PyTorch, torchvision, scikit-learn, NumPy, pandas, Matplotlib, and the broader open-source scientific Python ecosystem.

---

## Contact

**Md Naim Hassan Saykat**  
Department of Computer Science  
Université Paris-Saclay  
France

Academic correspondence:  
`md-naim-hassan.saykat@universite-paris-saclay.fr`

Project / collaboration correspondence:  
`mdnaimhassansaykat@gmail.com`

GitHub:  
[md-naim-hassan-saykat](https://github.com/md-naim-hassan-saykat)

---

## Need Help?

If you encounter a reproducibility, installation, or evaluation issue, please open an issue through the repository's issue tracker:

[Open a GitHub issue](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/issues/new/choose)

When reporting reproducibility issues, please include:

- operating system;
- Python version;
- PyTorch version;
- dataset being evaluated;
- model/checkpoint name;
- relevant command;
- complete error message or traceback.

---

## Responsible Use

This repository provides experimental research software for dermoscopic image classification.

It is **not a medical device**, has **not undergone prospective clinical validation**, and should **not be used independently for diagnosis, treatment decisions, patient triage, or other clinical decision-making**.

The reported results should be interpreted in the context of the methodological limitations described above, particularly retrospective internal harmonization, dataset shift, calibration differences, and the absence of dermatologist validation.
