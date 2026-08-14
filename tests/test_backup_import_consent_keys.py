"""Settings import must never grant a licence/consent acceptance.

`face_swap_license_accepted` is the guardrail the user has to grant personally
in Settings, after reading the InsightFace terms, before any model download can
happen (src/face_swap.py::_ensure_models_available). manage_settings already
refuses to write it (src/agent_tools/admin_tools.py::_CONSENT_KEYS), but
POST /api/import merged the caller's whole `settings` dict into the live
settings with no key filtering — and app_api can reach that route — so an agent
could grant the acceptance on the user's behalf by "restoring a backup".

The filter lives at the write boundary in the route itself (not only in a
caller blocklist) so every future caller of the import path inherits it.
"""
import asyncio
from unittest.mock import MagicMock

import routes.backup_routes as br


class _Req:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def _endpoint(monkeypatch):
    monkeypatch.setattr(br, "require_admin", lambda request: None)
    monkeypatch.setattr(br, "get_current_user", lambda request: "alice")

    mem = MagicMock()
    mem.load_all.return_value = []
    presets = MagicMock()
    presets.get_all.return_value = {}
    skills = MagicMock(spec=["load_all", "add_skill"])
    skills.load_all.return_value = []

    router = br.setup_backup_routes(mem, presets, skills)
    for r in router.routes:
        if r.path == "/api/import" and "POST" in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError("POST /api/import route not found")


def _run_import(monkeypatch, body, stored):
    """Run the import handler against an in-memory settings store."""
    endpoint = _endpoint(monkeypatch)
    with monkeypatch.context() as m:
        m.setattr(br, "load_settings", lambda: dict(stored))
        m.setattr(br, "save_settings", lambda s: stored.update(s))
        m.setattr(br, "load_features", lambda: {})
        m.setattr(br, "save_features", lambda f: None)
        return asyncio.run(endpoint(_Req(body)))


def test_import_cannot_grant_face_swap_licence(monkeypatch):
    """A body that explicitly sets the consent key must not turn the gate on."""
    stored = {"face_swap_license_accepted": False, "theme": "light"}
    result = _run_import(
        monkeypatch,
        {"settings": {"face_swap_license_accepted": True, "theme": "dark"}},
        stored,
    )

    assert result["ok"] is True
    # The consent key is untouched...
    assert stored["face_swap_license_accepted"] is False
    # ...while every other setting imports exactly as before.
    assert stored["theme"] == "dark"
    assert "settings" in result["imported"]
    assert "face_swap_license_accepted" in result.get("skipped", [])


def test_import_cannot_grant_licence_when_key_absent_from_store(monkeypatch):
    """Dropping must not re-add the key with an attacker-chosen value either."""
    stored = {"theme": "light"}
    _run_import(
        monkeypatch,
        {"settings": {"face_swap_license_accepted": True}},
        stored,
    )
    assert stored.get("face_swap_license_accepted") is not True


def test_import_of_ordinary_settings_is_unaffected(monkeypatch):
    """No consent key in the body -> no behaviour change, no `skipped` noise."""
    stored = {"theme": "light"}
    result = _run_import(
        monkeypatch,
        {"settings": {"theme": "dark", "temperature": 0.7}},
        stored,
    )

    assert result["ok"] is True
    assert stored == {"theme": "dark", "temperature": 0.7}
    assert "skipped" not in result


def test_consent_keys_match_admin_tools(monkeypatch):
    """Drift guard: the route's local copy must track the canonical set.

    backup_routes keeps its own literal rather than importing admin_tools —
    that import pulls the whole agent-tool registry (~700 modules) into a
    plain route module. This test is what keeps the two in sync.
    """
    from src.agent_tools.admin_tools import _CONSENT_KEYS as canonical

    assert br._CONSENT_KEYS == canonical
