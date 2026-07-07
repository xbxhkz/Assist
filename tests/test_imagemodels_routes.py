"""Route behavior for /api/imagemodels using a fake manager and an admin
override. Focus: the FLUX.2 fast-fail guard and list delegation."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.imagemodels_routes as imr
from core.middleware import require_admin


class FakeManager:
    def list_models(self):
        return [{"name": "flux1.gguf", "path": "/x/flux1.gguf", "size": 4}]

    def status(self):
        return {"running": False, "model": None, "port": None,
                "endpoint_id": None, "device": None}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(imr, "get_manager", lambda: FakeManager())
    app = FastAPI()
    app.include_router(imr.setup_imagemodels_routes())
    app.dependency_overrides[require_admin] = lambda: None
    return TestClient(app), tmp_path


def test_list_models(client):
    c, _ = client
    r = c.get("/api/imagemodels/models")
    assert r.status_code == 200
    assert r.json()["models"][0]["name"] == "flux1.gguf"


def test_serve_rejects_flux2_fast_with_clear_message(client):
    """FLUX.2 ggufs carry the same 'flux' arch tag but need a Mistral --llm
    encoder; without this guard they fail cryptically after minutes of load."""
    c, tmp = client
    f = tmp / "flux-2-klein-4b-Q8_0.gguf"
    f.write_bytes(b"x")
    r = c.post("/api/imagemodels/serve",
               json={"diffusion_model": str(f), "device": "cpu"})
    assert r.status_code == 400
    assert "FLUX.2" in r.json()["detail"]


def test_serve_rejects_missing_file(client):
    c, _ = client
    r = c.post("/api/imagemodels/serve",
               json={"diffusion_model": "/no/such/file.gguf"})
    assert r.status_code == 400
