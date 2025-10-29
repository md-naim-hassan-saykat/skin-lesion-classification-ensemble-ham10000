# Contributing Guidelines

_For issues or feature discussions, please open an [Issue](../../issues)._

Thank you for your interest in contributing to **Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019**!
We welcome pull requests for bug fixes, documentation improvements, and research extensions.

---

## Local Development Setup

### Clone the repository
```bash
git clone https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000.git
cd skin-lesion-classification-ensemble-ham10000
```

### Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-ci.txt  
pre-commit install
```

### Code Style & Quality
```bash
make format  
make lint     
make test     
make all
```

### Testing
```bash
pytest -q
pytest --cov=src
```

### Pull Request Process
```bash
# Create a new feature branch
git checkout -b feature/your-feature-name

# Stage and commit your changes
git add -A
git commit -m "Add: improved ensemble averaging logic"

# Push your branch to GitHub
git push --set-upstream origin feature/your-feature-name
```

## Contribution Ideas

You can contribute by:
- Adding new CNN or transformer backbones (e.g., ViT, Swin)
- Improving ensemble fusion or weighting strategies
- Enhancing documentation or examples
- Extending dataset support (ISIC 2020, PAD-UFES)
- Benchmarking cross-domain generalization

## Code of Conduct

Please maintain a **respectful and inclusive** environment for all contributors.
All discussions, issues, and PRs must follow [GitHub’s Community Guidelines](https://docs.github.com/en/site-policy/github-terms/github-community-guidelines).

## Attribution

All contributions will be acknowledged in the project’s contributor list and referenced in the release notes.
Thank you for helping advance open, reproducible research in dermatological AI!
