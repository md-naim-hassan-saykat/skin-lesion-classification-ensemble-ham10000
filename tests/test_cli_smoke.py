import subprocess, sys
def test_eval_help():
    r = subprocess.run([sys.executable, "src/evaluate.py", "--help"], capture_output=True)
    assert r.returncode == 0
