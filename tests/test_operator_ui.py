import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
def _read(p): return (ROOT / p).read_text(encoding="utf-8")


def test_index_has_operator_entry_and_modal():
    html = _read("static/index.html")
    for el in ('id="tool-operator-btn"', 'id="operator-modal"', 'id="operator-transcript"',
               'id="operator-goal"', 'id="operator-active-indicator"'):
        assert el in html, f"{el} missing"
    assert 'src="/static/js/operator.js"' in html


def test_operator_js_wires_endpoints():
    js = _read("static/js/operator.js")
    for ep in ("/api/operator/start", "/api/operator/status",
               "/api/operator/decision", "/api/operator/stop"):
        assert ep in js, f"{ep} not wired"
