from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.training_routes as tr


class FakeMgr:
    def list_adapters(self):
        return [{"run_id": "run-1", "complete": True, "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
                 "path": "p", "converted": False, "adapter_gguf": None}]


def _client(monkeypatch, exporter):
    monkeypatch.setattr(tr, "require_admin", lambda: None)
    monkeypatch.setattr(tr, "get_training_manager", lambda: FakeMgr())
    monkeypatch.setattr(tr, "get_adapter_exporter", lambda: exporter)
    app = FastAPI(); app.include_router(tr.setup_training_routes())
    return TestClient(app)


class Exp:
    def __init__(self, result): self._r = result
    def export(self, adapter_dir, quant, base_model=None): self.seen = (adapter_dir, quant); return self._r
    def list_exports(self, run_id): return ["p/exports/x.gguf"]
    def exports_dir(self): return "p/exports"


def test_export_ok(monkeypatch):
    e = Exp({"ok": True, "gguf": "p/exports/Qwen-run-1-Q4_K_M.gguf"})
    c = _client(monkeypatch, e)
    r = c.post("/api/training/adapters/run-1/export", json={"quant": "Q4_K_M"})
    assert r.status_code == 200 and r.json()["ok"] is True and e.seen[1] == "Q4_K_M"


def test_export_invalid_quant_400(monkeypatch):
    c = _client(monkeypatch, Exp({"ok": True}))
    r = c.post("/api/training/adapters/run-1/export", json={"quant": "BOGUS"})
    assert r.status_code == 400


def test_export_manager_error_400(monkeypatch):
    c = _client(monkeypatch, Exp({"error": "quantize failed"}))
    r = c.post("/api/training/adapters/run-1/export", json={"quant": "Q4_K_M"})
    assert r.status_code == 400


def test_adapter_status_lists_exports(monkeypatch):
    c = _client(monkeypatch, Exp({"ok": True}))
    r = c.get("/api/training/adapters/run-1")
    assert r.status_code == 200 and r.json()["exports"] == ["p/exports/x.gguf"]
