from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.training_routes as tr


class FakeMgr:
    def env_status(self):
        return {"status": "not_installed"}
    def status(self):
        return {"status": "idle"}
    def list_adapters(self):
        return [{"run_id": "run-1", "complete": True, "base_model": "x", "path": "p"}]


def _client(monkeypatch, mgr):
    monkeypatch.setattr(tr, "get_training_manager", lambda: mgr)
    monkeypatch.setattr(tr, "require_admin", lambda: None)  # bypass admin gate for shape tests
    app = FastAPI()
    app.include_router(tr.setup_training_routes())
    return TestClient(app)


def test_adapters_endpoint(monkeypatch):
    c = _client(monkeypatch, FakeMgr())
    r = c.get("/api/training/adapters")
    assert r.status_code == 200 and r.json()["adapters"][0]["run_id"] == "run-1"


def test_run_status_endpoint(monkeypatch):
    c = _client(monkeypatch, FakeMgr())
    r = c.get("/api/training/runs/current")
    assert r.status_code == 200 and r.json()["status"] == "idle"


def test_start_run_validates_body(monkeypatch):
    mgr = FakeMgr()
    mgr.start = lambda cfg: {"started": True, "run_id": "run-x"}
    c = _client(monkeypatch, mgr)
    r = c.post("/api/training/runs", json={"base_model": "x/Qwen2.5-0.5B",
                                           "dataset_path": "d.jsonl", "steps": 2})
    assert r.status_code == 200 and r.json()["started"] is True


def test_start_run_surfaces_manager_error(monkeypatch):
    mgr = FakeMgr()
    mgr.start = lambda cfg: {"error": "bad config"}
    c = _client(monkeypatch, mgr)
    r = c.post("/api/training/runs", json={"base_model": "", "dataset_path": "", "steps": 2})
    assert r.status_code == 400
