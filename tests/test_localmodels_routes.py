"""Route behavior for /api/localmodels using a fake manager and an admin
override. Verifies path-safety validation and manager delegation."""
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.localmodels_routes as lmr
from core.middleware import require_admin


class FakeManager:
    def __init__(self):
        self.started = None
        self.stopped = False
    def list_models(self):
        return [{"name": "m.gguf", "path": "/x/m.gguf", "size": 4}]
    def status(self):
        return {"running": bool(self.started), "model": self.started,
                "port": 8123, "endpoint_id": "local-0"}
    def start(self, model_path):
        self.started = os.path.basename(model_path)
        return self.status()
    def stop(self):
        self.stopped = True
        self.started = None
        return self.status()


@pytest.fixture
def client(monkeypatch, tmp_path):
    fake = FakeManager()
    monkeypatch.setattr(lmr, "get_manager", lambda: fake)
    monkeypatch.setattr(lmr, "MODELS_DIR", str(tmp_path))
    app = FastAPI()
    app.include_router(lmr.setup_localmodels_routes())
    app.dependency_overrides[require_admin] = lambda: None
    return TestClient(app), fake, tmp_path


def test_list_models(client):
    c, _fake, _ = client
    r = c.get("/api/localmodels/models")
    assert r.status_code == 200
    assert r.json()["models"][0]["name"] == "m.gguf"


def test_serve_rejects_path_outside_models_dir(client):
    c, _fake, _ = client
    r = c.post("/api/localmodels/serve", json={"model_path": "/etc/passwd.gguf"})
    assert r.status_code == 400


def test_serve_rejects_non_gguf(client):
    c, _fake, tmp = client
    f = tmp / "note.txt"
    f.write_text("x")
    r = c.post("/api/localmodels/serve", json={"model_path": str(f)})
    assert r.status_code == 400


def test_serve_starts_valid_model(client):
    c, fake, tmp = client
    f = tmp / "m.gguf"
    f.write_bytes(b"xxxx")
    r = c.post("/api/localmodels/serve", json={"model_path": str(f)})
    assert r.status_code == 200
    assert fake.started == "m.gguf"


def test_stop_delegates(client):
    c, fake, _ = client
    r = c.post("/api/localmodels/stop")
    assert r.status_code == 200
    assert fake.stopped is True
