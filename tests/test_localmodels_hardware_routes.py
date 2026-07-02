"""Hardware/recommendations/fit-annotation route tests (fakes, no real hw)."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.localmodels_routes as lmr
from core.middleware import require_admin


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(lmr, "get_hardware",
                        lambda: {"ram_gb": 16.0, "has_gpu": True,
                                 "gpu_name": "RTX 3060", "vram_gb": 8.0})
    monkeypatch.setattr(lmr, "recommend_models",
                        lambda: [{"name": "Qwen2.5-7B", "score": 0.9}])
    monkeypatch.setattr(lmr, "list_repo_gguf_files",
                        lambda repo: [{"filename": "m-7B-Q4.gguf", "size": 4_000_000_000,
                                       "url": "https://huggingface.co/a/b/resolve/main/m-7B-Q4.gguf"}])
    monkeypatch.setattr(lmr, "fit_for_file",
                        lambda f, hw: {"verdict": "gpu", "needed_gb": 4.5,
                                       "size_gb": 4.0, "param_estimate_gb": 4.2})
    app = FastAPI()
    app.include_router(lmr.setup_localmodels_routes())
    app.dependency_overrides[require_admin] = lambda: None
    return TestClient(app)


def test_hardware_route(client):
    r = client.get("/api/localmodels/hardware")
    assert r.status_code == 200
    assert r.json()["gpu_name"] == "RTX 3060"


def test_recommendations_route(client):
    r = client.get("/api/localmodels/recommendations")
    assert r.status_code == 200
    assert r.json()["recommendations"][0]["name"] == "Qwen2.5-7B"


def test_catalog_files_annotated_with_fit(client):
    r = client.get("/api/localmodels/catalog/files", params={"repo": "a/b"})
    assert r.status_code == 200
    f = r.json()["files"][0]
    assert f["fit"]["verdict"] == "gpu"
    assert f["filename"] == "m-7B-Q4.gguf"
