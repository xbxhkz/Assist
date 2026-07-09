import src.settings as settings


def test_default_is_false():
    assert settings.DEFAULT_SETTINGS.get("screen_access_enabled") is False


def test_reset_helper_forces_false(tmp_path, monkeypatch):
    p = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "SETTINGS_FILE", str(p))
    settings.save_settings({"screen_access_enabled": True, "keep": 1})
    settings.reset_screen_access()
    saved = settings.load_settings()
    assert saved["screen_access_enabled"] is False
    assert saved["keep"] == 1  # other settings preserved
