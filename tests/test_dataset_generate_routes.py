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


def test_generate_infinite_count_no_500(monkeypatch):
    async def fake(prompt, system=None, owner=None):
        return '{"text": "x"}'
    c = _client(monkeypatch, fake)
    # `json={"count": 1e400}` can't be used here: this env's TestClient encodes
    # the outgoing request body with allow_nan=False, so it raises client-side
    # on float('inf') before the request is even sent. Send the raw bytes
    # instead so "1e400" (a syntactically valid JSON number that overflows to
    # inf) reaches the server's own json.loads the way a real client's body
    # would, exercising the route's int(body["count"]) parsing directly.
    r = c.post(
        "/api/datasets/generate",
        content=b'{"format": "text", "count": 1e400, "brief": "b"}',
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 200 and "requested" in r.json()  # inf count degrades, no 500
