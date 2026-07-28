from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.dataset_routes as dr


class FakeStore:
    def __init__(self): self.saved = None
    def save(self, name, rows): self.saved = (name, rows); return {"ok": True, "path": "p/x.jsonl", "name": "x"}
    def list(self): return [{"name": "x", "path": "p/x.jsonl", "rows": 2, "size": 20}]
    def load(self, name): return {"rows": [{"text": "a"}], "name": name, "path": "p"}
    def delete(self, name): return {"ok": True}


def _client(monkeypatch, store=None):
    monkeypatch.setattr(dr, "require_admin", lambda: None)
    monkeypatch.setattr(dr, "get_dataset_store", lambda: store or FakeStore())
    app = FastAPI(); app.include_router(dr.setup_dataset_routes())
    return TestClient(app)


def test_validate_rows(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/api/datasets/validate", json={"rows": [{"text": "hi"}, {"nope": 1}]})
    assert r.status_code == 200 and r.json()["valid"] == 1 and r.json()["invalid"] == 1


def test_validate_text(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/api/datasets/validate", json={"text": '{"text":"hi"}\n{bad'})
    assert r.status_code == 200 and r.json()["invalid"] == 1


def test_save_list_load_delete(monkeypatch):
    store = FakeStore()
    c = _client(monkeypatch, store)
    assert c.post("/api/datasets", json={"name": "x", "rows": [{"text": "a"}]}).status_code == 200
    assert store.saved[0] == "x"
    assert c.get("/api/datasets").json()["datasets"][0]["name"] == "x"
    assert c.get("/api/datasets/x").json()["rows"] == [{"text": "a"}]
    assert c.request("DELETE", "/api/datasets/x").status_code == 200


def test_save_error_is_400(monkeypatch):
    class Bad(FakeStore):
        def save(self, name, rows): return {"error": "invalid dataset name"}
    r = _client(monkeypatch, Bad()).post("/api/datasets", json={"name": "", "rows": []})
    assert r.status_code == 400
