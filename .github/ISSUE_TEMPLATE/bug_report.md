---
name: Bug report
about: Report a reproducible software, evaluation, or repository workflow issue
title: "[BUG] "
labels: bug
---

## Bug Description

Provide a clear and concise description of the problem.

Explain what went wrong and, where relevant, which part of the repository is
affected.

Examples include:

- installation or dependency errors;
- training failures;
- checkpoint-loading errors;
- evaluation failures;
- ensemble or prediction-alignment errors;
- dataset-preparation problems;
- unexpected script behavior; or
- repository configuration issues.

> Please do not include private data, credentials, API keys, patient
> information, or other sensitive information in a public issue.

---

## Affected Component

Select the component most closely related to the issue:

- [ ] Installation / environment setup
- [ ] Data preparation
- [ ] Training
- [ ] Individual-model evaluation
- [ ] Ensemble evaluation
- [ ] Metrics / statistical analysis
- [ ] Interpretability / Grad-CAM++
- [ ] Documentation
- [ ] Tests / CI / pre-commit
- [ ] Other

If applicable, provide the relevant file, script, or command:

```text
Example: src/evaluate.py
```

---

## Steps to Reproduce

Provide the smallest reproducible sequence of steps that demonstrates the
problem.

For example:

1. Clone the repository:

   ```bash
   git clone https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000.git
   cd skin-lesion-classification-ensemble-ham10000
   ```

2. Create and activate a Python 3.11 virtual environment:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

   For development or repository-validation issues, also install:

   ```bash
   python -m pip install -r requirements-ci.txt
   ```

4. Run the command that produces the problem. For example:

   ```bash
   bash scripts/eval_all.sh
   ```

5. Describe what happens and include the relevant error message below.

If your issue does not involve the batch evaluation workflow, replace the
example command with the exact command needed to reproduce the problem.

---

## Expected Behavior

Describe what you expected to happen.

---

## Actual Behavior

Describe what actually happened.

Include the exact error message or unexpected output when possible.

---

## Minimal Error Output

Paste the smallest relevant terminal output or stack trace.

```text
Paste error output here
```

Please avoid posting unnecessarily long logs. Attach additional diagnostic
information only when it is needed to reproduce or understand the problem.

---

## Environment

Please provide the following information:

- **Operating system:** e.g., macOS 15, Ubuntu 24.04, Windows 11
- **Python version:** output of `python --version`
- **PyTorch version:** output of `python -c "import torch; print(torch.__version__)"`
- **torchvision version:** output of `python -c "import torchvision; print(torchvision.__version__)"`
- **Repository branch:** e.g., `main`
- **Commit hash:** output of `git rev-parse --short HEAD`
- **Execution device:** CPU / CUDA / MPS
- **Hardware (optional):** e.g., Apple M2 Pro, NVIDIA RTX 4090

If the issue may involve dependency conflicts, also provide:

```bash
python -m pip check
```

---

## Dataset and Checkpoint Context

Complete this section only when relevant to the issue.

- **Dataset:** HAM10000 / ISIC 2019 / other
- **Dataset path or layout:** describe without exposing private information
- **Model:** CNN / ResNet-50 / DenseNet-121 / EfficientNet-B3 / ConvNeXt-Tiny / MobileNetV3-Large / ViT-B/16 / ensemble
- **Checkpoint:** filename or identifier, if applicable
- **Evaluation cohort:** full dataset / standardized cohort / other
- **Output permutation used:** if applicable

Do not upload datasets or checkpoints unless their licenses and distribution
terms explicitly permit redistribution.

---

## Repository Validation

If possible, run:

```bash
make verify
```

Indicate whether it succeeds:

- [ ] All checks pass
- [ ] Pre-commit checks fail
- [ ] Tests fail
- [ ] Unable to run

If validation fails, include the relevant failure output in the error section
above.

---

## Additional Context

Add any other information that may help reproduce or diagnose the issue.

This may include:

- screenshots;
- configuration changes;
- relevant output files;
- whether the issue occurs consistently;
- whether the issue began after a particular commit or environment change; or
- a minimal example demonstrating the behavior.

---

## Research-Result Questions

If the issue concerns a discrepancy in reported metrics, reproduced predictions,
calibration results, or other research outputs, please provide enough
information to distinguish a software defect from a difference in experimental
configuration.

Where applicable, include:

- dataset or evaluation-cohort construction;
- model checkpoint;
- preprocessing configuration;
- canonical class ordering;
- output permutation;
- evaluation command; and
- relevant generated metrics.

Repository defaults are executable reference settings and may not reconstruct
every historical experiment without the corresponding archived experimental
configuration.
