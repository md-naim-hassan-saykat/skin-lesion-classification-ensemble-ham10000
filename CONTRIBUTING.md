# Contributing Guidelines

Thank you for your interest in contributing to **Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019**!  
We welcome pull requests for bug fixes, documentation improvements, and research extensions.

---

## Local Development Setup

```bash
# Clone the repo
git clone https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000.git
cd skin-lesion-classification-ensemble-ham10000
```

# Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

# Install dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-ci.txt  # optional: for linting and tests
pre-commit install
```
