import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ)
ENV["PYTHONPATH"] = str(ROOT)

def run(*args):
    p = subprocess.run(
        [sys.executable, "-m", "rvi_multimatrix", *args],
        cwd=ROOT,
        env=ENV,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if p.returncode != 0:
        raise AssertionError(
            f"command failed {args}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
        )
    return p.stdout

assert "0.4.0" in run("--version")
assert "semantic" in run("matrix", "list")
assert "SEED" in run("operator", "list")
assert len(run("kernel", "hash").strip()) == 64
assert "valid: True" in run("kernel", "validate")
assert "params" in run("model", "info")
assert "logits" in run("model", "smoke", "--seq-len", "8", "--batch-size", "1")
assert "Complete CLI" in (ROOT / "README.md").read_text(encoding="utf-8")
assert (ROOT / "docs" / "CLI_MANUAL.md").exists()

with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "lab"
    run("init", str(out))
    assert (out / "workspace.json").exists()
    assert (out / "configs" / "config.json").exists()

print("CLI PASS")
