import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_index_html_wires_the_editor():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="rail-workflows"' in html
    assert 'id="workflows-modal"' in html
    assert 'src="/static/js/workflows.js"' in html
    assert 'id="wf-canvas"' in html and 'id="wf-wires"' in html
    assert 'id="wf-triggers"' in html
    assert 'id="wf-triggers-btn"' in html


@pytest.mark.skipif(not shutil.which("node"), reason="node binary not on PATH")
def test_workflows_js_is_syntactically_valid():
    # Copy to a .mjs temp file so node parses it as an ES module (a bare .js is
    # treated as CommonJS and would reject `import`). --check validates syntax
    # only — it does not resolve the relative imports, so no DOM is executed.
    src = (ROOT / "static" / "js" / "workflows.js").read_text(encoding="utf-8")
    fd, tmp = tempfile.mkstemp(suffix=".mjs")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(src)
        subprocess.run(["node", "--check", tmp], check=True, capture_output=True, text=True)
    finally:
        os.unlink(tmp)
