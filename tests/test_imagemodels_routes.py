"""Route behavior for /api/imagemodels using a fake manager and an admin
override. Focus: the FLUX.2 fast-fail guard and list delegation."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.imagemodels_routes as imr
from core.middleware import require_admin


class FakeManager:
    def __init__(self):
        self.started = None

    def list_models(self):
        return [{"name": "flux1.gguf", "path": "/x/flux1.gguf", "size": 4}]

    def status(self):
        return {"running": False, "model": None, "port": None,
                "endpoint_id": None, "device": None}

    def start(self, files, device="cpu", steps=None):
        self.started = (files, device)
        self.files = files
        self.steps = steps
        return {"running": True, "model": "m", "port": 8200,
                "endpoint_id": "img-local-0", "device": device}


@pytest.fixture
def client(monkeypatch, tmp_path):
    fake = FakeManager()
    monkeypatch.setattr(imr, "get_manager", lambda: fake)
    import src.imagemodels.encoders as enc
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(tmp_path / "img"))
    app = FastAPI()
    app.include_router(imr.setup_imagemodels_routes())
    app.dependency_overrides[require_admin] = lambda: None
    return TestClient(app), fake, tmp_path


def test_list_models(client):
    c, _fake, _ = client
    r = c.get("/api/imagemodels/models")
    assert r.status_code == 200
    assert r.json()["models"][0]["name"] == "flux1.gguf"


def test_serve_flux2_missing_encoders_names_what_to_download(client):
    """A FLUX.2 serve without its Qwen3 --llm encoder / flux2 VAE must fail
    fast (before the minutes-long model load) and say which files to add."""
    c, _fake, tmp = client
    f = tmp / "flux-2-klein-4b-Q8_0.gguf"
    f.write_bytes(b"x")
    r = c.post("/api/imagemodels/serve",
               json={"diffusion_model": str(f), "device": "cpu"})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "FLUX.2" in detail and "llm" in detail and "vae" in detail


def test_serve_flux2_with_encoders_starts_manager(client):
    c, fake, tmp = client
    f = tmp / "flux-2-klein-4b-Q8_0.gguf"
    f.write_bytes(b"x")
    (tmp / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"x")     # sibling resolution
    (tmp / "flux2_ae.safetensors").write_bytes(b"x")
    r = c.post("/api/imagemodels/serve",
               json={"diffusion_model": str(f), "device": "cpu"})
    assert r.status_code == 200
    files, device = fake.started
    assert files["llm"].endswith("Qwen3-4B-Q4_K_M.gguf")
    assert files["vae"].endswith("flux2_ae.safetensors")
    assert "t5xxl" not in files and device == "cpu"


def test_serve_rejects_missing_file(client):
    c, _fake, _ = client
    r = c.post("/api/imagemodels/serve",
               json={"diffusion_model": "/no/such/file.gguf"})
    assert r.status_code == 400


def _flux2_set(tmp):
    f = tmp / "flux-2-klein-4b-Q8_0.gguf"
    f.write_bytes(b"x")
    (tmp / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"x")
    (tmp / "flux2_ae.safetensors").write_bytes(b"x")
    return f


def test_serve_clamps_and_passes_steps(client):
    c, fake, tmp = client
    f = _flux2_set(tmp)
    r = c.post("/api/imagemodels/serve",
               json={"diffusion_model": str(f), "device": "cpu", "steps": 400})
    assert r.status_code == 200
    assert fake.steps == 50


def test_serve_ignores_junk_steps(client):
    c, fake, tmp = client
    f = _flux2_set(tmp)
    r = c.post("/api/imagemodels/serve",
               json={"diffusion_model": str(f), "steps": "banana"})
    assert r.status_code == 200
    assert fake.steps is None


def _flux1_set(tmp):
    f = tmp / "flux1-dev.gguf"
    f.write_bytes(b"x")
    (tmp / "t5xxl.gguf").write_bytes(b"x")
    (tmp / "clip_l.safetensors").write_bytes(b"x")
    (tmp / "ae.safetensors").write_bytes(b"x")
    return f


def test_serve_fast_decode_includes_taesd_when_present(client):
    c, fake, tmp = client
    f = _flux1_set(tmp)
    (tmp / "taef1.safetensors").write_bytes(b"x")
    r = c.post("/api/imagemodels/serve",
               json={"diffusion_model": str(f), "fast_decode": True})
    assert r.status_code == 200
    assert fake.files.get("taesd", "").endswith("taef1.safetensors")


def test_serve_without_fast_decode_skips_taesd(client):
    c, fake, tmp = client
    f = _flux1_set(tmp)
    (tmp / "taef1.safetensors").write_bytes(b"x")
    r = c.post("/api/imagemodels/serve", json={"diffusion_model": str(f)})
    assert r.status_code == 200
    assert "taesd" not in fake.files
