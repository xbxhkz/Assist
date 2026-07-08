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


def test_localmodels_ui_has_device_toggle():
    html = _read("static/index.html")
    assert 'name="localmodels-device"' in html
    js = _read("static/js/localModels.js")
    assert "localmodels-device" in js


def test_localmodels_js_guards_ram_against_image_model():
    """Serving an LLM while an image model runs pages a small-RAM machine —
    the card must check the image subsystem and offer to stop it."""
    js = _read("static/js/localModels.js")
    assert "/api/imagemodels/status" in js
    assert "/api/imagemodels/stop" in js
