import src.settings as settings


def test_default_input_control_is_false():
    assert settings.DEFAULT_SETTINGS.get("input_control_enabled") is False


def test_reset_input_control_forces_off(monkeypatch):
    store = {"input_control_enabled": True}
    monkeypatch.setattr(settings, "load_settings", lambda: dict(store))
    saved = {}
    monkeypatch.setattr(settings, "save_settings", lambda s: saved.update(s))
    settings.reset_input_control()
    assert saved.get("input_control_enabled") is False
