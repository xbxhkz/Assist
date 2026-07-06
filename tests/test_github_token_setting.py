import src.settings as s


def test_github_token_default_empty(monkeypatch):
    monkeypatch.setattr(s, "load_settings", lambda: dict(s.DEFAULT_SETTINGS))
    assert s.get_github_token() == ""


def test_github_token_read_and_stripped(monkeypatch):
    monkeypatch.setattr(s, "load_settings", lambda: {**s.DEFAULT_SETTINGS, "github_token": "  ghp_x  "})
    assert s.get_github_token() == "ghp_x"
