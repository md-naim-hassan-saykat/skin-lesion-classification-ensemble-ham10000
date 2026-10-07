# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [2.1.0] - 2026-10-06

Released together with Zenodo archive version 2.1.0
([10.5281/zenodo.23187641](https://doi.org/10.5281/zenodo.23187641)).

### Archive

- The Zenodo archive now contains the revision analyses (ISIC 2019 partition by
  source, per-class F1 on the unseen images, temperature scaling, reliability
  table, computational cost), the analysis scripts and the research notebook.
- The master confidence-interval table in the archive now equals
  Supplementary Table S1, and the archive README was corrected.
- The archive no longer redistributes the HAM10000 and ISIC 2019 images.
  Earlier archive versions are now licensed CC BY-NC 4.0 and MIT to match the
  dataset licences.

### Fixed

- `scripts/eval_all.sh` and `src/evaluate.py` evaluated every checkpoint with
  one ImageNet transform at 224 x 224 and with identity class ordering, which
  reproduces the two defects corrected in the revised manuscript. Each
  checkpoint is now evaluated with its audited input size, normalization and
  output permutation (`archived_checkpoints` in `src/config.yaml`,
  `src/archived.py`).
- Filled the ResNet-50 and ViT-B/16 output permutations, previously `null`,
  with the audited value `6,5,0,2,3,1,4` (training class indices in HAM10000
  metadata order).
- `CustomCNN` now matches the archived CNN checkpoint (flattened 256 x 14 x 14
  features, 512-unit hidden layer, dropout 0.5), so the checkpoint loads with
  strict matching.
- The archived ViT-B/16 checkpoint is a Hugging Face
  `ViTForImageClassification` state dict; it is now built with
  `build_archived_vit()` instead of the torchvision ViT, and `transformers` is
  added to `requirements.txt`.
- Checkpoint search patterns in `scripts/eval_all.sh` now match the archived
  file names `mobilenetv3_ham10000.pth` and `vit_ham10000_best_model.pth`.
- `results/tables/HAM10000_master_results_with_95CI.csv` now matches
  Supplementary Table S1 and the README. The previous file used one shared
  bootstrap stream; the published intervals use per-metric seeds 42 to 45.
- Added the ensemble row to `results/tables/ISIC2019_per_class_F1.csv` and
  corrected the README statement that ConvNeXt-Tiny had the highest DF and VASC
  F1 (the ensemble does).
- ECE bins are now right-closed, `(lo, hi]`, matching the reported values and
  Supplementary Table S7. Reported ECE values are unchanged.
- `metadata.yaml` listed weighted cross-entropy, cosine annealing and
  Grad-CAM++ as the study settings. The archived models used unweighted
  cross-entropy and Adam with a constant learning rate (ConvNeXt-Tiny: AdamW
  with `ReduceLROnPlateau`), and attribution used Grad-CAM.
- Updated the project title in `CITATION.cff`, `metadata.yaml`, the
  documentation and `CONTRIBUTING.md`, and replaced remaining Grad-CAM++
  references.
- Repaired broken mathematical markup in the README and documentation.
- `.gitignore` re-ignored `results/tables/*.csv` after allowing them, so new
  curated tables could not be added without `git add -f`.

### Added

- Partition of ISIC 2019 by source: 10,011 of 25,331 images are HAM10000
  training images; results on the 15,320 unseen images
  (`ISIC2019_partitioned_by_source.csv`, `ISIC2019_contamination_summary.csv`,
  `ISIC2019_unseen_per_class_F1.csv`).
- Post-hoc temperature scaling of the ensemble and each model, with the
  ensemble reliability table (`calibration_temperature_scaling.csv`,
  `HAM10000_ensemble_reliability.csv`).
- Computational cost of each model and the ensemble on an NVIDIA T4
  (`computational_cost.csv`, `computational_overhead.json`).
- Archived training-protocol and test-membership audits
  (`HAM10000_protocol_audit.csv`,
  `HAM10000_historical_test_membership_audit.csv`), and the archived training
  protocol of each checkpoint in `docs/training_pipeline.md`.
- Analysis scripts under `scripts/revision_analysis/`: `bootstrap_ci.py`,
  `calibration_analysis.py`, `external_decontamination.py`,
  `verify_label_order.py`, `benchmark_overhead.py`, `kaggle_preflight.py`,
  `per_class_f1.py`.
- `tests/test_archived.py`, pinning the audited permutations and
  preprocessing.
- README sections on computational cost and on the corrections made to the
  original submission.
- Research notebook: a Revision Analyses section (checkpoint pre-flight,
  computational cost, ISIC 2019 partition by source, per-class F1 on the unseen
  remainder, temperature scaling and reliability table); the Overview and ISIC
  section state the ISIC 2019 overlap; the master 95% CI table is now taken from
  Supplementary Table S1 instead of a second bootstrap with different intervals.

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
