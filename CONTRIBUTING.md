# Contributing Guidelines

Thank you for your interest in contributing to **Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019**!

We welcome contributions of all kinds — from **bug fixes and documentation improvements** to **new model experiments, training pipelines, or evaluation methods**.
Your input helps improve the reproducibility, quality, and research impact of this project.

---

## How to Contribute

1. **Fork** the repository to your GitHub account.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/skin-lesion-classification-ensemble-ham10000.git
   cd skin-lesion-classification-ensemble-ham10000
   ```
3.	Create a new branch for your feature or fix:
	```bash
    git checkout -b feat/my-feature
    ```
4.	Make your changes and run tests locally before committing.

---

## Local Development Setup
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-ci.txt  # optional: for linting and testing tools
ruff check src --fix
black src

# Enable pre-commit hooks (auto-formatting & lint checks)
pre-commit install
```
You can now run:
```bash
make lint
make test
```

---

## Code Style

	•	Formatting: enforced via Black
	•	Linting: checked with Ruff
	•	Testing: handled by pytest

All pre-commit hooks are configured automatically.
Please ensure no lint or test failures before opening a PR.

## Pull Request Process
1.	Ensure your branch is up to date with main:
```bash
git pull --rebase origin main
```
2.	Push your branch and open a Pull Request.
3.	Clearly describe:

	    •	The motivation for your change
	    •	A summary of modifications
	    •	Tests performed (if applicable)
4.	The CI pipeline will run automatically — all tests must pass before merging.

---

## Need Help?

If you encounter any setup or contribution issues, please [open an issue](https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000/issues/new/choose)
or contact the maintainer at [mdnaimhassansaykat@gmail.com](mailto:mdnaimhassansaykat@gmail.com).

Thank you for helping make this project robust, reproducible, and impactful for the medical AI research community!
