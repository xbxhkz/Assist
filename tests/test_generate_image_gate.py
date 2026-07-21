import src.agent_loop as al


def _patch(monkeypatch, enabled, image_model):
    def fake_get_setting(key, default=None):
        if key == "image_gen_enabled":
            return enabled
        if key == "image_model":
            return image_model
        return default
    monkeypatch.setattr(al, "get_setting", fake_get_setting)


def test_hidden_when_disabled_and_no_default(monkeypatch):
    _patch(monkeypatch, False, "")
    assert al._generate_image_hidden() is True


def test_available_when_default_image_model_set(monkeypatch):
    _patch(monkeypatch, False, "flux.gguf")
    assert al._generate_image_hidden() is False


def test_available_when_enabled(monkeypatch):
    _patch(monkeypatch, True, "")
    assert al._generate_image_hidden() is False
