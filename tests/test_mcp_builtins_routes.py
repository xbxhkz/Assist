import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.mcp_routes as mr
from core.middleware import require_admin


class FakeMgr:
    def __init__(self):
        self.disconnected = []
        self.reconnected = []
    def get_server_status(self, sid):
        return {"status": "connected", "tool_count": 1, "error": None}
    def is_builtin(self, sid):
        return sid in {"image_gen", "memory", "rag", "email"}
    async def disconnect_server(self, sid):
        self.disconnected.append(sid)
    async def _reconnect_builtin(self, sid):
        self.reconnected.append(sid)
        return True


@pytest.fixture
def client(monkeypatch):
    fake = FakeMgr()
    # Setting store the endpoints read/write.
    store = {"disabled_builtin_mcp": []}
    monkeypatch.setattr(mr, "get_setting",
                        lambda k, d=None: store.get(k, d), raising=False)
    def _save(d):
        store.update(d)
    monkeypatch.setattr(mr, "set_setting", _save, raising=False)
    app = FastAPI()
    app.include_router(mr.setup_mcp_routes(fake))
    app.dependency_overrides = {}
    # require_admin is called inside handlers; stub it to pass.
    monkeypatch.setattr(mr, "require_admin", lambda request: None)
    return TestClient(app), fake, store


def test_list_builtins(client):
    c, _fake, _store = client
    r = c.get("/api/mcp/builtins")
    assert r.status_code == 200
    ids = {b["id"]: b for b in r.json()["builtins"]}
    assert set(ids) == {"image_gen", "memory", "rag", "email"}
    assert ids["memory"]["enabled"] is True
    assert ids["memory"]["status"] == "connected"


def test_toggle_off_disconnects_and_persists(client):
    c, fake, store = client
    r = c.post("/api/mcp/builtins/rag/toggle", json={"enabled": False})
    assert r.status_code == 200 and r.json()["enabled"] is False
    assert "rag" in store["disabled_builtin_mcp"]
    assert "rag" in fake.disconnected


def test_toggle_on_reconnects_and_persists(client):
    c, fake, store = client
    store["disabled_builtin_mcp"] = ["rag"]
    r = c.post("/api/mcp/builtins/rag/toggle", json={"enabled": True})
    assert r.status_code == 200 and r.json()["enabled"] is True
    assert "rag" not in store["disabled_builtin_mcp"]
    assert "rag" in fake.reconnected


def test_toggle_rejects_unknown_id(client):
    c, *_ = client
    assert c.post("/api/mcp/builtins/nope/toggle", json={"enabled": False}).status_code == 400
