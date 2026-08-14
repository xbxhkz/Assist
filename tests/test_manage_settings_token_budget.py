"""manage_settings' `set`-action key classification.

Two opposite regressions live here:
  - agent_input_token_budget must BE settable from chat (not flagged secret);
  - face_swap_license_accepted must NOT be settable from chat at all -- it is
    a license acceptance the user has to grant personally in Settings after
    reading the terms, and writing it from chat would satisfy
    src/face_swap.py's gate without the user ever seeing the license.
"""
import asyncio
import json

import src.settings as settings_mod
from src.agent_tools.admin_tools import do_manage_settings


def test_set_token_budget_is_not_refused_as_secret(monkeypatch):
    store = {}
    monkeypatch.setattr(settings_mod, "load_settings", lambda: dict(store))
    monkeypatch.setattr(settings_mod, "save_settings", lambda s: store.update(s))

    result = asyncio.run(do_manage_settings(json.dumps({
        "action": "set", "key": "agent_input_token_budget", "value": 8000,
    })))

    # The "token" substring used to flag this int setting as a credential and
    # refuse to set it (even though there's a deliberate "token budget" alias).
    assert "credential" not in result.get("response", "").lower(), result
    assert result.get("exit_code") == 0, result
    assert store.get("agent_input_token_budget") == 8000


def _settings_spy(monkeypatch, store):
    """Patch load/save_settings against `store`, recording every save."""
    saves = []

    def _save(s):
        saves.append(dict(s))
        store.clear()
        store.update(s)

    monkeypatch.setattr(settings_mod, "load_settings", lambda: dict(store))
    monkeypatch.setattr(settings_mod, "save_settings", _save)
    return saves


def test_set_face_swap_license_acceptance_is_refused(monkeypatch):
    """GUARDRAIL: the agent must never accept the InsightFace model license
    for the user. face_swap_license_accepted is a plain bool that passes
    every other `set` filter (not in _SECRET_KEYS, no 'token'/'_key'/'secret'
    suffix, not a dict/list), so without an explicit consent-key refusal the
    agent could satisfy src/face_swap.py's gate itself and trigger a
    several-hundred-MB non-commercial-licensed model download the user never
    saw terms for."""
    for value in (True, "true", "yes", "on", 1):
        store = {"face_swap_license_accepted": False}
        saves = _settings_spy(monkeypatch, store)

        result = asyncio.run(do_manage_settings(json.dumps({
            "action": "set", "key": "face_swap_license_accepted", "value": value,
        })))

        assert "can't set it on their behalf" in result.get("response", ""), (value, result)
        # The setting itself must be untouched -- and nothing written at all.
        assert store["face_swap_license_accepted"] is False, (value, store)
        assert saves == [], f"refusal still wrote settings for value={value!r}: {saves}"


def test_reset_face_swap_license_acceptance_is_still_allowed(monkeypatch):
    """Only `set` is refused: resetting the key back to its default (False)
    merely re-locks the gate, so it must keep working."""
    store = {"face_swap_license_accepted": True}
    _settings_spy(monkeypatch, store)

    result = asyncio.run(do_manage_settings(json.dumps({
        "action": "reset", "key": "face_swap_license_accepted",
    })))

    assert result.get("exit_code") == 0, result
    assert store["face_swap_license_accepted"] is False, store
