from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.dataset_routes as dr


def _client(monkeypatch, fake_call):
    monkeypatch.setattr(dr, "require_admin", lambda: None)
    monkeypatch.setattr(dr, "_default_model_call", fake_call)
    app = FastAPI(); app.include_router(dr.setup_dataset_routes())
    return TestClient(app)


def test_generate_happy(monkeypatch):
    async def fake(prompt, system=None, owner=None):
        return '{"text": "a"}\n{"text": "b"}'
    c = _client(monkeypatch, fake)
    r = c.post("/api/datasets/generate", json={"format": "text", "count": 2, "brief": "b"})
    assert r.status_code == 200
    j = r.json()
    assert j["produced"] == 2 and j["requested"] == 2 and "error" not in j


def test_generate_count_clamped(monkeypatch):
    async def fake(prompt, system=None, owner=None):
        return '{"text": "x"}'
    c = _client(monkeypatch, fake)
    r = c.post("/api/datasets/generate", json={"format": "text", "count": 9999, "brief": "b"})
    assert r.status_code == 200 and r.json()["requested"] == 200  # MAX_GENERATE


def test_generate_no_endpoint_is_error_not_500(monkeypatch):
    async def boom(prompt, system=None, owner=None):
        raise RuntimeError("no default model endpoint configured")
    c = _client(monkeypatch, boom)
    r = c.post("/api/datasets/generate", json={"format": "text", "count": 3, "brief": "b"})
    assert r.status_code == 200 and "error" in r.json() and r.json()["produced"] == 0


def test_generate_bad_body_no_500(monkeypatch):
    async def fake(prompt, system=None, owner=None):
        return '{"text": "x"}'
    c = _client(monkeypatch, fake)
    r = c.post("/api/datasets/generate", json={})  # no count/brief/format
    assert r.status_code == 200 and "requested" in r.json()
