"""Text guards for the Phase 3c hardware-aware UI."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_index_has_hardware_and_recommendations_elements():
    html = _read("static/index.html")
    assert 'id="localmodels-hardware"' in html
    assert 'id="localmodels-recommendations"' in html


def test_js_calls_hardware_recommendations_delete_and_uses_fit():
    js = _read("static/js/localModels.js")
    for ep in ("/api/localmodels/hardware", "/api/localmodels/recommendations",
               "/api/localmodels/delete"):
        assert ep in js, f"{ep} not called in localModels.js"
    assert ".fit" in js or "fit.verdict" in js  # badges consume the fit annotation
    assert "keydown" in js or "keypress" in js   # Enter-to-search wiring
