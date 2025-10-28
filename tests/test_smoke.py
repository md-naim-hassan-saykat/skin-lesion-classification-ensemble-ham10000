"""Basic smoke tests for the Skin Lesion Ensemble project.

These tests ensure that:
1. Core source files import successfully.
2. The CLI interface (`evaluate.py --help`) executes without error.
"""

import subprocess
import sys
import pytest


def test_imports():
    """Verify that key source modules import without errors."""
    import src.evaluate  # noqa: F401
    import src.ensemble  # noqa: F401


@pytest.mark.timeout(10)
def test_eval_help():
    """Ensure `src/evaluate.py --help` runs successfully."""
    result = subprocess.run(
        [sys.executable, "src/evaluate.py", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    # The command should complete successfully
    assert result.returncode == 0, f"Non-zero exit: {result.stderr}"
    assert "usage" in result.stdout.lower(), "Missing usage text in help output"
