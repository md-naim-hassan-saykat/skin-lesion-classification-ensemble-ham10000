---
name: Bug report
about: Report a reproducible issue or error to help improve the project
title: "[BUG] Short description"
labels: bug
---

## Description
A clear and concise description of the issue or unexpected behavior.

---

## Steps to Reproduce
Please provide detailed steps to reproduce the issue:

1. Clone the repository:
	```bash
   	git clone https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000.git
   	cd skin-lesion-classification-ensemble-ham10000
	```

2.	Activate the environment:
	```bash
 	python -m venv .venv && source .venv/bin/activate
	```

3.	Run the evaluation:
	```bash
	bash scripts/eval_all.sh
	```

4.	Observe the error output: …

Expected Behavior

Describe what you expected to happen instead of the observed behavior.

---

## Environment Details

Please complete the following information:

	•	OS: (e.g., macOS 15.5 / Ubuntu 22.04)
	•	Python version: (e.g., 3.11.4)
	•	PyTorch version: (e.g., 2.2.2)
	•	Branch or commit hash: (e.g., main or abcd123)
	•	Hardware (optional): (e.g., M2 Pro / RTX 4090)

  ---

  ## Logs / Screenshots
  If applicable, include relevant terminal output, stack traces, or screenshots for better context.
