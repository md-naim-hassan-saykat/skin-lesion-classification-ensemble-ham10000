import subprocess
import sys

import pytest


def test_imports():
    """Verify that key source modules import without errors."""
    import src.ensemble  # noqa: F401
    import src.evaluate  # noqa: F401


@pytest.mark.timeout(10)
def test_eval_help():
    result = subprocess.run(
        [sys.executable, "src/evaluate.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
