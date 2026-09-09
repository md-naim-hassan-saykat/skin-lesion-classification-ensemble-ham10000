# Pull Request

## Summary

Provide a concise description of the purpose of this pull request.

Explain:

- what is being changed;
- why the change is needed; and
- which part of the repository is affected.

If the PR addresses an existing issue, reference it here.

Example:

```text
Closes #123
```

---

## Type of Change

Select all that apply:

- [ ] Bug fix
- [ ] New feature or functionality
- [ ] Model or training-pipeline update
- [ ] Evaluation or metrics update
- [ ] Ensemble-method update
- [ ] Data-processing update
- [ ] Interpretability / Grad-CAM++ update
- [ ] Statistical-analysis update
- [ ] Code refactoring or optimization
- [ ] Tests added or updated
- [ ] Documentation update
- [ ] CI / development tooling update
- [ ] Configuration or dependency update
- [ ] Reproducibility artifact update
- [ ] Other

If **Other** is selected, briefly describe the change:

```text
Description:
```

---

## Changes Made

Describe the principal changes introduced by this PR.

Where useful, identify the affected files or components.

Example:

```text
- Updated src/evaluate.py to ...
- Added tests for ...
- Updated docs/evaluation_metrics.md to ...
```

Keep this section focused on changes that are relevant to reviewing the PR.

---

## Motivation and Context

Explain why this change is necessary.

For research-related changes, describe whether the PR affects:

- executable repository behavior;
- manuscript-related methodology;
- reported or curated results;
- reproducibility;
- external validation; or
- documentation only.

If the change intentionally alters previous behavior or results, explain why.

---

## Testing and Validation

Describe how the changes were tested.

The standard repository validation command is:

```bash
make verify
```

This runs the configured pre-commit checks and test suite.

Individual checks may also be run with:

```bash
make check
make test
pre-commit run --all-files
pytest -q
```

For changes affecting evaluation, additional validation may include:

```bash
bash scripts/eval_all.sh
```

Do not run dataset-dependent or checkpoint-dependent commands solely to complete
this template if the required data or model artifacts are not available.

### Validation Results

Indicate the checks performed:

- [ ] `make verify` passes
- [ ] Pre-commit checks pass
- [ ] Tests pass
- [ ] Relevant evaluation workflow was tested
- [ ] Relevant training workflow was tested
- [ ] Documentation-only change; runtime evaluation not required
- [ ] Additional manual validation was performed
- [ ] Not applicable

Provide additional details when needed:

```text
Validation notes:
```

---

## Research and Evaluation Impact

Complete this section if the PR affects training, preprocessing, checkpoints,
evaluation, metrics, ensemble inference, statistical analysis, or reported
research results.

### Affected Dataset

- [ ] HAM10000
- [ ] ISIC 2019
- [ ] Both
- [ ] Other
- [ ] Not applicable

### Affected Model or Component

- [ ] Conventional CNN
- [ ] ResNet-50
- [ ] DenseNet-121
- [ ] EfficientNet-B3
- [ ] ConvNeXt-Tiny
- [ ] MobileNetV3-Large
- [ ] ViT-B/16
- [ ] Seven-model ensemble
- [ ] Evaluation / metrics
- [ ] Calibration
- [ ] Interpretability
- [ ] Other
- [ ] Not applicable

### Does This PR Change Reported Numerical Results?

- [ ] No
- [ ] Yes
- [ ] Not applicable

If **Yes**, describe the affected metrics, tables, figures, or other artifacts
and explain why the results changed.

```text
Result changes:
```

Do not describe a numerical change as a regression solely because it differs
from a previously reported value. Determine whether the difference is expected
from the methodological or implementation change introduced by the PR.

---

## Reproducibility Considerations

For changes that affect experimental behavior, confirm the relevant items:

- [ ] Dataset or cohort construction remains documented
- [ ] Canonical class ordering remains correct
- [ ] Checkpoint compatibility has been considered
- [ ] Output permutations have been considered where applicable
- [ ] Preprocessing changes are documented
- [ ] Training-configuration changes are documented
- [ ] Evaluation-metric changes are documented
- [ ] Ensemble membership and weighting remain documented
- [ ] Randomness or seed-related changes are documented
- [ ] Not applicable

The canonical lesion-class order used by the repository is:

```text
akiec
bcc
bkl
df
mel
nv
vasc
```

If this PR changes any assumption used by archived checkpoints or manuscript
results, explain the compatibility implications.

---

## Generated and Curated Artifacts

If this PR modifies results, figures, tables, prediction files, or other
research artifacts, describe them here.

Newly generated experimental outputs normally belong under:

```text
outputs/
```

Curated manuscript and reproducibility artifacts belong under:

```text
results/
```

Confirm where applicable:

- [ ] Generated outputs are not being mistaken for curated results
- [ ] Updated figures correspond to the updated numerical results
- [ ] Updated tables correspond to the updated numerical results
- [ ] Obsolete artifacts were removed or clearly superseded
- [ ] Not applicable

---

## Dependencies and Configuration

Does this PR modify any of the following?

- [ ] `requirements.txt`
- [ ] `requirements-ci.txt`
- [ ] `pyproject.toml`
- [ ] `src/config.yaml`
- [ ] `.pre-commit-config.yaml`
- [ ] GitHub Actions / CI configuration
- [ ] Other environment or configuration files
- [ ] No dependency or configuration changes

If dependencies or configuration changed, explain why and indicate whether the
environment was validated successfully.

```text
Dependency/configuration notes:
```

---

## Documentation

List any documentation updated as part of this PR.

Examples include:

```text
README.md
docs/installation.md
docs/training_pipeline.md
docs/evaluation_metrics.md
docs/ensemble_method.md
```

If user-facing behavior changed but documentation was not updated, explain why.

```text
Documentation notes:
```

---

## Screenshots, Figures, or Logs

If useful for review, include relevant:

- terminal output;
- error messages;
- evaluation summaries;
- figures;
- confusion matrices;
- ROC curves; or
- Grad-CAM++ visualizations.

Include only the information needed to review the change.

Do not include credentials, API keys, private data, patient information, or
other sensitive information.

---

## Data and Model Artifacts

Do not commit or attach datasets, checkpoints, or other third-party artifacts
unless their licenses and distribution terms explicitly permit redistribution.

Confirm where applicable:

- [ ] No restricted dataset files are included
- [ ] No unauthorized model/checkpoint files are included
- [ ] No credentials, secrets, or private information are included
- [ ] Not applicable

---

## Final Checklist

Before submitting the pull request, confirm that:

- [ ] The PR has a clear and descriptive title
- [ ] The change is limited to the intended scope
- [ ] Code follows the repository's Black and Ruff configuration
- [ ] Relevant tests and validation checks pass
- [ ] New or changed behavior is covered by tests where practical
- [ ] Documentation is updated where necessary
- [ ] Research-result changes are documented where applicable
- [ ] Generated and curated artifacts are clearly distinguished
- [ ] No unnecessary large files are included
- [ ] No credentials, private data, or sensitive information are included
- [ ] The branch is based on the current `main` branch
- [ ] Merge conflicts have been resolved
