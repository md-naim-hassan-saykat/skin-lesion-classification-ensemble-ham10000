# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Harmonized retrospective evaluation framework for seven dermoscopic
  skin-lesion classification architectures:
  - custom CNN;
  - ResNet-50;
  - DenseNet-121;
  - EfficientNet-B3;
  - ConvNeXt-Tiny;
  - MobileNetV3-Large;
  - ViT-B/16.
- Equal-weight seven-model probability ensemble using arithmetic averaging of
  class-probability vectors.
- Standardized HAM10000 retrospective evaluation cohort of 2,003 images.
- Harmonized ISIC 2019 external evaluation cohort of 25,331 images using the
  common seven-class label space.
- Canonical class-output harmonization for:
  `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, and `vasc`.
- HAM10000 evaluation metrics:
  - accuracy;
  - weighted F1;
  - macro ROC-AUC;
  - micro ROC-AUC;
  - Expected Calibration Error.
- ISIC 2019 external evaluation metrics:
  - accuracy;
  - weighted F1;
  - macro ROC-AUC;
  - weighted ROC-AUC.
- Bootstrap uncertainty analysis using 1,000 resamples and 95% percentile
  confidence intervals.
- Paired McNemar comparisons between the ensemble and individual models.
- Paired bootstrap comparisons of ensemble-versus-model performance
  differences.
- External ISIC 2019 per-class F1 analysis.
- Confusion-matrix and ROC-curve reproducibility figures.
- Qualitative Grad-CAM attribution analysis, including class-specific ensemble
  visualizations.
- Machine-readable manuscript and supplementary result tables under
  `results/tables/`.
- Curated manuscript and reproducibility figures under `results/figures/`.
- Dedicated documentation for:
  - installation;
  - training;
  - evaluation metrics;
  - ensemble methodology.
- Repository metadata through `CITATION.cff` and `metadata.yaml`.
- Contributor guidance, Code of Conduct, security policy, issue templates, and
  pull-request template.
- Automated code-quality checks using Black, Ruff, and pre-commit.
- Automated test suite using pytest.
- GitHub Actions continuous integration for repository quality checks and
  tests.
- Makefile targets for setup, validation, testing, evaluation, and cleanup.
- HAM10000 data-preparation utility with raw-dataset image-count validation.
- Seven-model batch evaluation workflow through `scripts/eval_all.sh`.

### Changed

- Revised the project documentation to distinguish the standardized
  retrospective HAM10000 evaluation cohort from a universally untouched
  common test set.
- Revised interpretation of the reported results to reflect that ViT-B/16,
  rather than the ensemble, provides the strongest HAM10000 point estimates.
- Documented ConvNeXt-Tiny as the strongest model by the reported ISIC 2019
  external point estimates.
- Clarified that the equal-weight ensemble does not uniformly outperform the
  strongest individual architecture.
- Clarified that archived checkpoints originated from differing historical
  training and data-partitioning pipelines.
- Separated historical manuscript-reproduction requirements from current
  repository quick-start training defaults.
- Refined `src/config.yaml` to distinguish reported evaluation settings,
  checkpoint compatibility metadata, preprocessing defaults, and quick-start
  training configuration.
- Standardized the recommended development environment around Python 3.11.
- Updated and constrained runtime and CI dependency specifications.
- Aligned Ruff configuration and the pre-commit Ruff version.
- Expanded the README to document datasets, evaluation methodology, statistical
  analyses, calibration, interpretability, limitations, reproducibility
  artifacts, and responsible use.
- Refined Git attributes so curated figures are stored directly in Git while
  large checkpoint and numerical-array formats remain suitable for Git LFS.
- Consolidated pytest configuration into `pytest.ini`.

### Removed

- Redundant local pytest configuration.
- Outdated documentation implying that the reported ensemble consists of only
  DenseNet-121, EfficientNet-B3, and ConvNeXt-Tiny.
- Ambiguous claims that the ensemble is uniformly the strongest-performing
  approach.

---

## [1.0.0] - 2025-10-19

### Added

- Initial public research-software release.
- Source code for dermoscopic skin-lesion classification experiments.
- Model training, evaluation, and probability-level ensemble utilities.
- Initial HAM10000 evaluation workflow.
- Research notebook for skin-lesion ensemble classification.
- Repository automation and continuous-integration infrastructure.
- Reproducibility documentation and project metadata.
- MIT License.
- Zenodo archival record and DOI:
  `10.5281/zenodo.17390952`.

[Unreleased]: https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/compare/v1.0.0...HEAD
[1.0.0]: https://doi.org/10.5281/zenodo.17390952
