def test_imports():
    import src.evaluate  # noqa: F401
    import src.ensemble  # noqa: F401

def test_eval_help():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "src/evaluate.py", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
