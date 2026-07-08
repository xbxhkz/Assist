"""image_size setting: declared, per-user, and offered in the Settings UI.

Local diffusion pays ~4x for 1024x1024 vs 512x512 on small GPUs; the size
must be a user choice, not a hardcoded default in do_generate_image.
"""
import pathlib

import src.settings as settings

ROOT = pathlib.Path(settings.__file__).resolve().parents[1]


def test_image_size_has_default():
    assert settings.DEFAULT_SETTINGS.get("image_size") == "1024x1024"


def test_image_size_is_per_user():
    assert "image_size" in settings._PER_USER_KEYS


def test_generate_image_reads_size_setting():
    import inspect
    import src.ai_interaction as ai
    src_text = inspect.getsource(ai.do_generate_image)
    assert 'get_user_setting("image_size"' in src_text, \
        "do_generate_image must consult the per-user image_size setting"


def test_settings_ui_offers_size_select():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="set-imgSizeSelect"' in html
    js = (ROOT / "static" / "js" / "settings.js").read_text(encoding="utf-8")
    assert "image_size" in js
