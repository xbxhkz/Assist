"""Catalog + download route behavior with fakes (no network)."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.localmodels_routes as lmr
from core.middleware import require_admin


class FakeDL:
    def __init__(self):
        self.started = None
        self.cancelled = False
    def start(self, url, filename):
        self.started = (url, filename)
        return {"downloading": True, "filename": filename, "bytes": 0,
                "total": None, "pct": None, "error": None}
    def status(self):
        return {"downloading": False, "filename": None, "bytes": 0,
                "total": None, "pct": None, "error": None}
    def cancel(self):
        self.cancelled = True
        return self.status()


@pytest.fixture
def client(monkeypatch):
    fake = FakeDL()
    monkeypatch.setattr(lmr, "search_gguf_models",
                        lambda q, sort="downloads": [{"repo": "a/b", "downloads": 9, "likes": 1}])
    monkeypatch.setattr(lmr, "list_repo_gguf_files",
                        lambda repo: [{"filename": "m.gguf", "size": 10,
                                       "url": "https://huggingface.co/a/b/resolve/main/m.gguf"}])
    monkeypatch.setattr(lmr, "get_download_manager", lambda: fake)
    app = FastAPI()
    app.include_router(lmr.setup_localmodels_routes())
    app.dependency_overrides[require_admin] = lambda: None
    return TestClient(app), fake


def test_search(client):
    c, _ = client
    r = c.get("/api/localmodels/catalog/search", params={"q": "qwen"})
    assert r.status_code == 200
    assert r.json()["results"][0]["repo"] == "a/b"


def test_files(client):
    c, _ = client
    r = c.get("/api/localmodels/catalog/files", params={"repo": "a/b"})
    assert r.status_code == 200
    assert r.json()["files"][0]["filename"] == "m.gguf"


def test_download_valid(client):
    c, fake = client
    r = c.post("/api/localmodels/download",
               json={"url": "https://huggingface.co/a/b/resolve/main/m.gguf",
                     "filename": "m.gguf"})
    assert r.status_code == 200
    assert fake.started == ("https://huggingface.co/a/b/resolve/main/m.gguf", "m.gguf")


def test_download_rejects_non_hf_url(client):
    c, _ = client
    r = c.post("/api/localmodels/download",
               json={"url": "https://evil.com/m.gguf", "filename": "m.gguf"})
    assert r.status_code == 400


def test_download_rejects_bad_filename(client):
    c, _ = client
    r = c.post("/api/localmodels/download",
               json={"url": "https://huggingface.co/a/b/resolve/main/m.gguf",
                     "filename": "../evil.gguf"})
    assert r.status_code == 400


def test_download_cancel(client):
    c, fake = client
    r = c.post("/api/localmodels/download/cancel")
    assert r.status_code == 200
    assert fake.cancelled is True
