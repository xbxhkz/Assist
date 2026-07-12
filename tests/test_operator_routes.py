import asyncio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.operator_routes as orr


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(orr, "require_admin", lambda request: None)
    # both consents on
    monkeypatch.setattr(orr, "get_setting",
                        lambda k, d=None: True if k in ("screen_access_enabled", "input_control_enabled") else d)
    orr._reset_for_test()
    app = FastAPI()
    app.include_router(orr.setup_operator_routes())
    return TestClient(app)


def test_start_refuses_without_consent(client, monkeypatch):
    monkeypatch.setattr(orr, "get_setting", lambda k, d=None: False)
    r = client.post("/api/operator/start", json={"goal": "x"})
    assert r.status_code == 400 and "off" in r.json()["detail"].lower()


def test_status_idle_before_start(client):
    r = client.get("/api/operator/status")
    assert r.status_code == 200 and r.json()["status"] == "idle"


def test_start_requires_goal(client):
    r = client.post("/api/operator/start", json={"goal": ""})
    assert r.status_code == 400


def test_single_session(client, monkeypatch):
    # make the session runner block so the first stays "active"
    async def fake_runner(state):
        state["status"] = "running"
        await asyncio.Event().wait()  # never returns
    monkeypatch.setattr(orr, "_run_session", fake_runner)
    assert client.post("/api/operator/start", json={"goal": "a"}).status_code == 200
    assert client.post("/api/operator/start", json={"goal": "b"}).status_code == 409  # already running


def test_stop_flips_input_control_off(client, monkeypatch):
    flipped = {}
    monkeypatch.setattr(orr, "set_input_control", lambda on: flipped.setdefault("v", on))
    async def fake_runner(state):
        state["status"] = "running"
        await asyncio.Event().wait()
    monkeypatch.setattr(orr, "_run_session", fake_runner)
    client.post("/api/operator/start", json={"goal": "a"})
    r = client.post("/api/operator/stop")
    assert r.status_code == 200 and flipped["v"] is False


def test_stop_during_decide_ends_session_no_lingering_card(monkeypatch):
    import asyncio as _a
    from src.operator.actions import Action
    orr._reset_for_test()
    monkeypatch.setattr(orr, "set_input_control", lambda on: None)
    monkeypatch.setattr(orr, "get_setting",
                        lambda k, d=None: {"screen_access_enabled": True, "input_control_enabled": True,
                                           "operator_max_rounds": 30, "operator_max_seconds": 600}.get(k, d))
    release = {}

    async def fake_perceive(*a, **k):
        return {"windows": [], "elements": []}
    async def fake_decide(goal, history, percept, **k):
        await release["ev"].wait()  # simulate a long model call
        return Action(kind="act", tool="mouse", args={"action": "click", "x": 1, "y": 1})
    async def fake_execute(action, ctx, **k):
        return {"output": "ok"}
    monkeypatch.setattr(orr.primitives, "real_perceive", fake_perceive)
    monkeypatch.setattr(orr.primitives, "real_decide", fake_decide)
    monkeypatch.setattr(orr.primitives, "real_execute", fake_execute)

    async def drive():
        release["ev"] = _a.Event()
        state = orr._new_state("g", None)
        task = _a.create_task(orr._run_session(state))
        await _a.sleep(0)          # let it reach the blocked decide()
        state["_stop"] = True      # user hits panic Stop mid-decide
        state["_wake"].set()
        release["ev"].set()        # decide now returns a mutating action
        await _a.wait_for(task, timeout=5)  # MUST NOT hang
        return state
    state = _a.run(drive())
    assert state["status"] == "stopped"
    assert state["pending"] is None  # no fresh approval card after Stop


def test_resolve_operator_chat_prefers_served_model(monkeypatch):
    # The stale default_model must NOT be used when the endpoint serves a model:
    # the operator uses whatever the default endpoint actually reports.
    monkeypatch.setattr(orr, "get_setting",
                        lambda k, d=None: "ep1" if k == "default_endpoint_id" else d)

    def probe(ep_id, owner):
        assert ep_id == "ep1"
        return ("http://127.0.0.1:8100/v1/chat/completions", "served-agent-model", {"H": "1"})

    url, model, headers = orr._resolve_operator_chat("admin", probe=probe)
    assert model == "served-agent-model" and url.endswith("/chat/completions")


def test_resolve_operator_chat_falls_back_when_probe_has_no_model(monkeypatch):
    monkeypatch.setattr(orr, "get_setting",
                        lambda k, d=None: {"default_endpoint_id": "ep1",
                                           "default_model": "agent-x"}.get(k, d))

    def probe(ep_id, owner):
        return (None, None, None)  # endpoint reports no usable model

    def fake_resolve(spec, owner):
        return ("http://fallback/v1/chat/completions", spec, {})

    _, model, _ = orr._resolve_operator_chat("admin", probe=probe, resolve_model=fake_resolve)
    assert model == "agent-x"  # fell back to default_model


def test_resolve_operator_chat_falls_back_when_probe_raises(monkeypatch):
    monkeypatch.setattr(orr, "get_setting",
                        lambda k, d=None: {"default_endpoint_id": "ep1",
                                           "default_model": "agent-x"}.get(k, d))

    def probe(ep_id, owner):
        raise RuntimeError("endpoint down")

    def fake_resolve(spec, owner):
        return ("http://fallback", spec, {})

    _, model, _ = orr._resolve_operator_chat("admin", probe=probe, resolve_model=fake_resolve)
    assert model == "agent-x"
