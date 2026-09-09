# Contributing Guidelines

Thank you for your interest in contributing to Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019.

Contributions that improve the correctness, reproducibility, documentation, evaluation, or maintainability of the project are welcome. These may include bug fixes, documentation improvements, tests, model or evaluation enhancements, and reproducibility improvements.

Before contributing, please review the repository’s Code of Conduct.

---

## How to Contribute

1. Fork the repository to your GitHub account.
2. Clone your fork locally:

git clone https://github.com/<your-username>/skin-lesion-classification-ensemble-ham10000.git
cd skin-lesion-classification-ensemble-ham10000

3. Create a branch for your contribution:

git checkout -b feat/my-feature

    Use a descriptive branch name such as fix/data-loader, docs/evaluation, or test/ensemble.
4. Make your changes while keeping the scope of the contribution focused.
5. Run the quality checks and tests described below before opening a pull request.

---

## Local Development Setup

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

On Windows:

python -m venv .venv
.venv\Scripts\activate

Upgrade pip and install the project and development dependencies:

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-ci.txt

Install and enable pre-commit:

python -m pip install pre-commit
pre-commit install

---

## Code Quality

The repository uses:

* Black for Python code formatting.
* Ruff for linting and import organization.
* pytest for automated tests.
* pre-commit for formatting, linting, syntax validation, and repository-hygiene checks.

Black and Ruff are configured with a line length of 100 characters.

Before submitting a pull request, run:

pre-commit run --all-files
pytest -q

All applicable checks should pass.

If you prefer the repository’s Makefile commands, you may also use:

make lint
make test

Do not manually reformat files in a way that conflicts with Black or Ruff. The project-specific Ruff configuration is maintained in pyproject.toml, and the pre-commit hooks are defined in .pre-commit-config.yaml.

---

## Testing

Changes that affect program behavior should include appropriate tests whenever practical.

When adding or modifying functionality:

* Add or update tests under tests/.
* Keep tests deterministic where possible.
* Avoid tests that require downloading the full HAM10000 or ISIC 2019 datasets unless explicitly necessary.
* Do not require GPU hardware for basic unit or smoke tests.
* Ensure existing tests continue to pass.

Run the complete test suite with:

pytest -q

---

## Research and Reproducibility Contributions

Contributions involving models, training procedures, evaluation methods, calibration, interpretability, or reported results should be reproducible and clearly documented.

Where applicable, please provide:

* A clear description of the methodological change.
* Relevant hyperparameters and experimental settings.
* Dataset and split information.
* Evaluation metrics and comparison criteria.
* Random seeds or other controls needed for reproducibility.
* Tests or validation demonstrating that the implementation behaves as intended.

Changes to reported experimental results should include sufficient supporting evidence and should not overwrite established reference results without explanation.

---

## Data, Models, and Generated Artifacts

Do not commit large datasets, model checkpoints, credentials, API keys, or unnecessary generated files to the repository.

In particular, contributors should not commit local copies of HAM10000, ISIC 2019, trained model weights, caches, or temporary experiment outputs.

Curated reproducibility artifacts under designated repository directories may be committed when they are intentionally part of the project and comply with the repository’s file-tracking policy.

If a contribution requires a large external artifact, describe how it can be obtained or reproduced rather than committing it directly to Git.

---

## Commit Guidelines

Keep commits focused and use concise, descriptive commit messages.

Examples:

fix: correct class-weight computation
feat: add calibration evaluation
test: extend ensemble smoke tests
docs: clarify external validation workflow
chore: update development tooling

Avoid combining unrelated changes into a single commit.

---

## Pull Request Process

Before opening a pull request:

1. Synchronize your branch with the latest upstream main when necessary.
2. Run:

pre-commit run --all-files
pytest -q

3. Push your branch to your fork.
4. Open a pull request against the repository’s main branch.

In the pull request description, clearly explain:

* The motivation for the contribution.
* The changes that were made.
* Any methodological or behavioral implications.
* The tests or validation performed.
* Any limitations, assumptions, or follow-up work that reviewers should know about.

Automated CI checks should pass before a contribution is merged.

For changes affecting research results, please distinguish clearly between reproduced reference results and newly generated experimental results.

---

## Reporting Bugs and Requesting Features

For reproducible bugs, unexpected behavior, or feature proposals, please use the repository’s issue tracker.

When reporting a bug, include enough information to reproduce the problem, including the relevant command, environment, error message, and minimal reproduction steps where possible.

Security-related issues should not be reported through a public issue. Please follow the instructions in the repository’s Security Policy.

---

## Need Help?

For questions about contributing or the development setup, please open an issue.

For matters that should be handled privately, contact the maintainer at mdnaimhassansaykat@gmail.com.

---

Thank you for helping improve the reliability, reproducibility, and usefulness of this project for the medical AI research community.
