"""Guard: the Settings → Vision model dropdown lists downloaded local models,
not only the currently-running endpoint (so the vision model is selectable
even when a different chat model is served)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_vision_settings_pulls_local_models():
    js = (ROOT / "static" / "js" / "settings.js").read_text(encoding="utf-8")
    vis = js.split("initVisionSettings", 1)[1].split("function ", 1)[0]
    assert "/api/localmodels/models" in vis, \
        "vision model dropdown must include downloaded local models"
