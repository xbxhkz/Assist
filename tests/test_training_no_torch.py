"""Locks the core architectural invariant: importing the Py3.14 app must never
pull in the training stack (torch/peft/bitsandbytes/transformers). Those run
ONLY inside the separate CUDA venv via training_sidecar/train.py.

Runs `import app` in a CLEAN subprocess (not under the pytest conftest stubs)
so it reflects how production actually loads the app. A stray `import torch`
anywhere in the app's import graph makes the child fail (the libs aren't
installed in this interpreter), so this test fails loudly on regression."""
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_app_import_does_not_load_training_stack():
    code = (
        "import app, sys\n"
        "bad = [m for m in ('torch', 'peft', 'bitsandbytes', 'transformers') if m in sys.modules]\n"
        "assert not bad, 'app imported the training stack: ' + ','.join(bad)\n"
        "print('NO_TRAINING_STACK_OK')\n"
    )
    p = subprocess.run([sys.executable, "-c", code], cwd=_REPO_ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert p.returncode == 0, (
        "clean `import app` failed or pulled in the training stack:\n"
        f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    assert "NO_TRAINING_STACK_OK" in p.stdout
