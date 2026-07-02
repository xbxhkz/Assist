"""Text guards that the Local Models UI is wired (elements + endpoint calls)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_index_has_localmodels_modal_and_opener():
    html = _read("static/index.html")
    assert 'id="localmodels-modal"' in html
    assert 'id="tool-localmodels-btn"' in html
    assert 'src="/static/js/localModels.js"' in html or "js/localModels.js" in html


def test_localmodels_js_calls_all_endpoints():
    js = _read("static/js/localModels.js")
    for ep in ("/api/localmodels/models", "/api/localmodels/status",
               "/api/localmodels/serve", "/api/localmodels/stop"):
        assert ep in js, f"{ep} not called in localModels.js"
