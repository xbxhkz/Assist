"""list_active_agents() joins agent_runs.list_active() against
SessionManager's in-memory session cache for owner-scoped display.
GET /api/agent-runs/active exposes it. See
docs/superpowers/specs/2026-08-11-mission-control-active-agents-design.md.
"""
import asyncio
from types import SimpleNamespace

import routes.agent_runs_routes as arr
from src import agent_runs


def _route(router, path, method):
    for r in router.routes:
        if r.path == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(path)


def _request(user):
    return SimpleNamespace(state=SimpleNamespace(current_user=user))


class _FakeSession:
    def __init__(self, owner, name):
        self.owner = owner
        self.name = name


class _FakeSessionManager:
    def __init__(self, sessions):
        self.sessions = sessions


def test_list_active_agents_filters_by_owner(monkeypatch):
    monkeypatch.setattr(agent_runs, "list_active", lambda: ["s1", "s2", "s3"])
    sm = _FakeSessionManager({
        "s1": _FakeSession(owner="alice", name="Alice A"),
        "s2": _FakeSession(owner="bob", name="Bob B"),
        "s3": _FakeSession(owner="alice", name="Alice C"),
    })

    out = arr.list_active_agents(sm, owner="alice")

    assert out == [
        {"session_id": "s1", "session_name": "Alice A"},
        {"session_id": "s3", "session_name": "Alice C"},
    ]


def test_list_active_agents_skips_missing_session(monkeypatch):
    monkeypatch.setattr(agent_runs, "list_active", lambda: ["gone", "s1"])
    sm = _FakeSessionManager({"s1": _FakeSession(owner="alice", name="Alice A")})

    out = arr.list_active_agents(sm, owner="alice")

    assert out == [{"session_id": "s1", "session_name": "Alice A"}]


def test_list_active_agents_none_owner_matches_none_owner_sessions(monkeypatch):
    monkeypatch.setattr(agent_runs, "list_active", lambda: ["s1", "s2"])
    sm = _FakeSessionManager({
        "s1": _FakeSession(owner=None, name="Shared"),
        "s2": _FakeSession(owner="alice", name="Alice A"),
    })

    out = arr.list_active_agents(sm, owner=None)

    assert out == [{"session_id": "s1", "session_name": "Shared"}]


def test_route_scopes_to_effective_user(monkeypatch):
    captured = {}

    def fake_list_active_agents(session_manager, owner):
        captured["owner"] = owner
        return [{"session_id": "s1", "session_name": "Alice A"}]

    monkeypatch.setattr(arr, "list_active_agents", fake_list_active_agents)
    monkeypatch.setattr(arr, "effective_user", lambda request: "alice")
    router = arr.setup_agent_runs_routes(session_manager=object())
    handler = _route(router, "/api/agent-runs/active", "GET")

    out = asyncio.run(handler(request=_request("alice")))

    assert captured["owner"] == "alice"
    assert out == {"active": [{"session_id": "s1", "session_name": "Alice A"}]}
