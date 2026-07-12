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
