import io
import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.dataset_routes as dr


def _client(monkeypatch, *, chunks=None, fake_call=None):
    monkeypatch.setattr(dr, "require_admin", lambda: None)
    async def _default(prompt, system=None, owner=None):
        return '{"text": "a"}'
    monkeypatch.setattr(dr, "_default_model_call", fake_call or _default)
    if chunks is not None:
        monkeypatch.setattr(dr, "library_chunks", lambda store, query, k=8: chunks)
        monkeypatch.setattr(dr, "document_chunks",
                            lambda path, ext, pdf_pages=None, read_text=None, filename=None, max_chars=2000: chunks)
    app = FastAPI(); app.include_router(dr.setup_dataset_routes())
    return TestClient(app)


def test_grounded_happy(monkeypatch):
    c = _client(monkeypatch, chunks=[{"source": "M, p.1", "text": "AAA"}])
    r = c.post("/api/datasets/generate/grounded", json={"format": "text", "count": 1, "query": "x"})
    assert r.status_code == 200
    j = r.json()
    assert j["produced"] == 1 and j["rows"][0]["source"] == "M, p.1"


def test_grounded_requires_query(monkeypatch):
    c = _client(monkeypatch, chunks=[{"source": "M, p.1", "text": "AAA"}])
    r = c.post("/api/datasets/generate/grounded", json={"format": "text", "count": 1, "query": "  "})
    assert r.status_code == 200 and "error" in r.json() and r.json()["produced"] == 0


def test_grounded_no_hits(monkeypatch):
    c = _client(monkeypatch, chunks=[])
    r = c.post("/api/datasets/generate/grounded", json={"format": "text", "count": 1, "query": "zzz"})
    assert r.status_code == 200 and "error" in r.json()


def test_upload_happy(monkeypatch):
    c = _client(monkeypatch, chunks=[{"source": "Guide.pdf p.1", "text": "AAA"}])
    files = {"file": ("Guide.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")}
    data = {"format": "text", "count": "1", "brief": "", "existing": json.dumps([])}
    r = c.post("/api/datasets/generate/upload", files=files, data=data)
    assert r.status_code == 200 and r.json()["produced"] == 1
    assert r.json()["rows"][0]["source"] == "Guide.pdf p.1"


def test_upload_unextractable(monkeypatch):
    c = _client(monkeypatch, chunks=[])   # extractor yields nothing
    files = {"file": ("x.bin", io.BytesIO(b"\x00\x01"), "application/octet-stream")}
    r = c.post("/api/datasets/generate/upload", files=files, data={"format": "text", "count": "1"})
    assert r.status_code == 200 and "error" in r.json() and r.json()["produced"] == 0
