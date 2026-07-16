import src.settings as settings


def test_camera_access_defaults_off():
    assert settings.DEFAULT_SETTINGS["camera_access_enabled"] is False
    assert settings.DEFAULT_SETTINGS["webcam_describe_default"] is False


def test_reset_camera_access_forces_off(monkeypatch):
    store = {"camera_access_enabled": True}
    monkeypatch.setattr(settings, "load_settings", lambda: dict(store))
    saved = {}
    monkeypatch.setattr(settings, "save_settings", lambda s: saved.update(s))
    settings.reset_camera_access()
    assert saved.get("camera_access_enabled") is False
