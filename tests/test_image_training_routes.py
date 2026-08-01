from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.image_training_routes as itr


class FakeMgr:
    def env_status(self):
        return {"status": "not_installed"}

    def status(self):
        return {"status": "idle"}


def _client(monkeypatch, mgr):
    monkeypatch.setattr(itr, "get_image_training_manager", lambda: mgr)
    monkeypatch.setattr(itr, "require_admin", lambda: None)  # bypass admin gate for shape tests
    app = FastAPI()
    app.include_router(itr.setup_image_training_routes())
    return TestClient(app)


def test_env_endpoint(monkeypatch):
    c = _client(monkeypatch, FakeMgr())
    r = c.get("/api/image-training/env")
    assert r.status_code == 200 and r.json()["status"] == "not_installed"


def test_run_status_endpoint(monkeypatch):
    c = _client(monkeypatch, FakeMgr())
    r = c.get("/api/image-training/runs/current")
    assert r.status_code == 200 and r.json()["status"] == "idle"


def test_start_run_validates_body(monkeypatch):
    mgr = FakeMgr()
    mgr.start = lambda cfg: {"started": True, "run_id": "run-x"}
    c = _client(monkeypatch, mgr)
    r = c.post("/api/image-training/runs",
              json={"dataset_name": "ds1", "output_name": "my-lora", "steps": 2})
    assert r.status_code == 200 and r.json()["started"] is True


def test_start_run_surfaces_manager_error(monkeypatch):
    mgr = FakeMgr()
    mgr.start = lambda cfg: {"error": "bad config"}
    c = _client(monkeypatch, mgr)
    r = c.post("/api/image-training/runs", json={"dataset_name": "", "output_name": "", "steps": 2})
    assert r.status_code == 400


def test_stop_endpoint(monkeypatch):
    mgr = FakeMgr()
    mgr.stop = lambda: {"stopped": True}
    c = _client(monkeypatch, mgr)
    r = c.post("/api/image-training/runs/stop")
    assert r.status_code == 200 and r.json()["stopped"] is True
