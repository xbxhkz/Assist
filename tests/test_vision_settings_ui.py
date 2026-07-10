"""Guard: the Settings → Vision model dropdown lists downloaded local models,
not only the currently-running endpoint (so the vision model is selectable
even when a different chat model is served)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_vision_settings_pulls_local_models():
    js = (ROOT / "static" / "js" / "settings.js").read_text(encoding="utf-8")
    tail = js[js.find("initVisionSettings"):]
    # Bound the section at the NEXT top-level async function (nested plain
    # `function` helpers inside initVisionSettings must not end the slice).
    nxt = tail.find("async function", len("async function initVisionSettings"))
    section = tail[:nxt] if nxt > 0 else tail
    assert "/api/localmodels/models" in section, \
        "vision model dropdown must include downloaded local models"
