"""Text guards that the Image Models UI is wired (elements + endpoint calls)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_index_has_imagemodels_card_elements():
    html = _read("static/index.html")
    for el in ('id="imagemodels-status"', 'id="imagemodels-list"',
               'id="imagemodels-browse-btn"', 'id="imagemodels-serve-btn"'):
        assert el in html, f"{el} missing from index.html"


def test_imagemodels_js_calls_all_endpoints_and_renders_list():
    js = _read("static/js/imageModels.js")
    for ep in ("/api/imagemodels/models", "/api/imagemodels/status",
               "/api/imagemodels/serve", "/api/imagemodels/stop"):
        assert ep in js, f"{ep} not called in imageModels.js"
    assert "imagemodels-list" in js, "model list not rendered"
