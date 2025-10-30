## Summary
Briefly describe the purpose and motivation of this Pull Request (PR).  
Explain what problem it solves or what feature or enhancement it introduces.

---

## Changes Made
Select all that apply:

- [ ] Core logic / model updates  
- [ ] Code refactoring or optimization  
- [ ] Tests added or updated  
- [ ] Documentation improvements  
- [ ] CI / configuration updates  
- [ ] Bug fix  
- [ ] New feature or functionality  
- [ ] Other (please specify): ___

---

## Testing Performed
Describe how this PR was tested and validated.

Example:
```bash
cd ~/skin-lesion-classification-ensemble-ham10000
source .venv/bin/activate
export PYTHONPATH="$PWD"
pytest -q
python src/evaluate.py --help
bash scripts/eval_all.sh
```

---

## If applicable, include:

	•	All unit and integration tests passed
	•	Manual verification on HAM10000 / ISIC 2019 datasets
	•	Reproducible metrics consistent with prior results

---

## Screenshots or Logs (Optional)
Attach relevant output, figures, or Grad-CAM visualizations if this PR affects evaluation or visualization code.

---

## Checklist
Before submitting, confirm that:

	•	Code follows the repository’s style guidelines (Black, Ruff)
	•	All tests pass locally
	•	Documentation is up to date
	•	No merge conflicts with the main branch
	•	PR title follows the conventional format (e.g., feat:, fix:, docs:)

---
