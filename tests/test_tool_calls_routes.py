"""GET /api/tool-calls must scope to the caller and validate its date filters.

Mirrors the direct-endpoint-call test style used by
tests/test_memory_routes_session_owner.py -- no HTTP layer needed, the route
function is called directly with a fake request.
"""
import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import routes.tool_calls_routes as tcr


def _route(router, path, method):
    for r in router.routes:
        if r.path == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(path)


def _request(user):
    return SimpleNamespace(state=SimpleNamespace(current_user=user))


def test_owner_scopes_to_caller(monkeypatch):
    captured = {}

    def fake_list_tool_calls(db, owner, **kwargs):
        captured["owner"] = owner
        return [], False

    monkeypatch.setattr(tcr, "list_tool_calls", fake_list_tool_calls)
    router = tcr.setup_tool_calls_routes()
    handler = _route(router, "/api/tool-calls", "GET")

    out = asyncio.run(handler(request=_request("alice")))

    assert captured["owner"] == "alice"
    assert out == {"tool_calls": [], "has_more": False}


def test_invalid_since_returns_400(monkeypatch):
    monkeypatch.setattr(tcr, "list_tool_calls", lambda db, owner, **kw: ([], False))
    router = tcr.setup_tool_calls_routes()
    handler = _route(router, "/api/tool-calls", "GET")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(handler(request=_request("alice"), since="not-a-date"))

    assert exc.value.status_code == 400


def test_invalid_until_returns_400(monkeypatch):
    monkeypatch.setattr(tcr, "list_tool_calls", lambda db, owner, **kw: ([], False))
    router = tcr.setup_tool_calls_routes()
    handler = _route(router, "/api/tool-calls", "GET")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(handler(request=_request("alice"), until="not-a-date"))

    assert exc.value.status_code == 400


def test_limit_and_offset_are_clamped(monkeypatch):
    captured = {}

    def fake_list_tool_calls(db, owner, **kwargs):
        captured.update(kwargs)
        return [], False

    monkeypatch.setattr(tcr, "list_tool_calls", fake_list_tool_calls)
    router = tcr.setup_tool_calls_routes()
    handler = _route(router, "/api/tool-calls", "GET")

    asyncio.run(handler(request=_request("alice"), limit=99999, offset=-5))

    assert captured["limit"] == 200
    assert captured["offset"] == 0


def test_filters_are_forwarded(monkeypatch):
    captured = {}

    def fake_list_tool_calls(db, owner, **kwargs):
        captured.update(kwargs)
        return [], False

    monkeypatch.setattr(tcr, "list_tool_calls", fake_list_tool_calls)
    router = tcr.setup_tool_calls_routes()
    handler = _route(router, "/api/tool-calls", "GET")

    asyncio.run(handler(request=_request("alice"), session_id="s1", tool_name="web_search"))

    assert captured["session_id"] == "s1"
    assert captured["tool_name"] == "web_search"


def test_since_offset_aware_iso_is_converted_to_naive_utc(monkeypatch):
    """ChatMessage.timestamp is naive UTC. An offset-aware 'since' must be
    converted to naive UTC before being handed to list_tool_calls, not
    compared as if the offset weren't there."""
    captured = {}

    def fake_list_tool_calls(db, owner, **kwargs):
        captured.update(kwargs)
        return [], False

    monkeypatch.setattr(tcr, "list_tool_calls", fake_list_tool_calls)
    router = tcr.setup_tool_calls_routes()
    handler = _route(router, "/api/tool-calls", "GET")

    asyncio.run(handler(request=_request("alice"), since="2026-03-01T12:00:00+05:00"))

    assert captured["since"] == datetime(2026, 3, 1, 7, 0, 0)
    assert captured["since"].tzinfo is None
