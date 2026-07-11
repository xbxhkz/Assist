import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
def _read(p): return (ROOT / p).read_text(encoding="utf-8")


def test_index_has_input_control_toggle():
    html = _read("static/index.html")
    for el in ('id="input-control-toggle"', 'id="input-control-indicator"'):
        assert el in html, f"{el} missing from index.html"
    assert 'src="/static/js/inputControl.js"' in html


def test_inputcontrol_js_posts_setting():
    js = _read("static/js/inputControl.js")
    assert "input_control_enabled" in js
    assert "/api/auth/settings" in js
