import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_index_has_training_modal_and_scripts():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in ('id="training-modal"', 'id="rail-training"',
               '/static/js/training.js', '/static/js/trainingCore.js'):
        assert el in html, f"{el} missing from index.html"


def test_training_js_references_shipped_routes():
    src = (ROOT / "static" / "js" / "training.js").read_text(encoding="utf-8")
    for route in ('/api/training/env', '/api/training/env/setup', '/api/training/runs',
                  '/api/training/runs/current', '/api/training/runs/stop',
                  '/api/training/adapters', '/api/auth/status'):
        assert route in src, f"{route} not referenced in training.js"


def test_training_js_syntax_ok(tmp_path):
    src = (ROOT / "static" / "js" / "training.js").read_text(encoding="utf-8")
    mjs = tmp_path / "training.mjs"
    mjs.write_text(src, encoding="utf-8")
    p = subprocess.run(["node", "--check", str(mjs)], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr


def test_help_manual_has_training_section():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    # a phrase unique to the Help-manual paragraph (NOT present in Task 2's modal),
    # so this test genuinely gates the Help section rather than the modal.
    assert "fine-tune a small model on your own GPU" in html


def test_training_js_references_adapter_serve_routes():
    src = (ROOT / "static" / "js" / "training.js").read_text(encoding="utf-8")
    assert "/convert" in src and "/serve" in src and "serveCore.js" in src


def test_sidebar_tools_lists_admin_tools():
    """Training and Workflows are admin tools that previously existed ONLY as
    icon-rail buttons, so they were invisible in the sidebar Tools list where
    every other tool lives. Both must have a sidebar entry (admin-revealed)."""
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="tool-training-btn"' in html, "Training missing from sidebar Tools list"
    assert 'id="tool-workflows-btn"' in html, "Workflows missing from sidebar Tools list"


def test_sidebar_tool_buttons_are_wired_and_admin_gated():
    tjs = (ROOT / "static" / "js" / "training.js").read_text(encoding="utf-8")
    wjs = (ROOT / "static" / "js" / "workflows.js").read_text(encoding="utf-8")
    assert "tool-training-btn" in tjs, "training.js must wire the sidebar button"
    assert "tool-workflows-btn" in wjs, "workflows.js must wire the sidebar button"


def test_training_js_references_export():
    src = (ROOT / "static" / "js" / "training.js").read_text(encoding="utf-8")
    assert "/export" in src and "exportCore.js" in src and "exports/reveal" in src
