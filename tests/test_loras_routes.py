from fastapi import FastAPI
from fastapi.testclient import TestClient
from core.middleware import require_admin
import routes.loras_routes as lr


def make_client(monkeypatch):
    monkeypatch.setattr(lr.loras, "list_loras", lambda: [{"name": "a", "filename": "a.safetensors", "size": 9}])
    monkeypatch.setattr(lr.loras, "delete_lora", lambda n: n == "a")
    monkeypatch.setattr(lr.loras, "download_to_loras",
                        lambda url, fn, **k: {"name": "x", "filename": "x.safetensors", "size": 5})
    monkeypatch.setattr(lr.civitai, "search", lambda q, token=None: [{"name": "L", "download_url": "u"}])
    monkeypatch.setattr(lr, "get_setting", lambda k, d=None: "")
    app = FastAPI()
    app.include_router(lr.setup_loras_routes())
    app.dependency_overrides[require_admin] = lambda: None
    return TestClient(app)


def test_list(monkeypatch):
    r = make_client(monkeypatch).get("/api/loras")
    assert r.status_code == 200 and r.json()["loras"][0]["name"] == "a"


def test_civitai_search(monkeypatch):
    r = make_client(monkeypatch).get("/api/loras/civitai/search", params={"q": "anime"})
    assert r.status_code == 200 and r.json()["results"][0]["name"] == "L"


def test_download_civitai(monkeypatch):
    r = make_client(monkeypatch).post("/api/loras/download",
        json={"source": "civitai", "download_url": "http://c/dl", "file_name": "x.safetensors"})
    assert r.status_code == 200 and r.json()["lora"]["name"] == "x"


def test_download_bad_source(monkeypatch):
    r = make_client(monkeypatch).post("/api/loras/download", json={"source": "nope"})
    assert r.status_code == 400


def test_delete_ok_and_404(monkeypatch):
    c = make_client(monkeypatch)
    assert c.delete("/api/loras/a").status_code == 200
    assert c.delete("/api/loras/missing").status_code == 404


def test_delete_invalid_name(monkeypatch):
    # delete_lora raises ValueError for unsafe names → route maps to 400.
    c = make_client(monkeypatch)

    def boom(n):
        raise ValueError("unsafe")
    monkeypatch.setattr(lr.loras, "delete_lora", boom)
    assert c.delete("/api/loras/badname").status_code == 400
