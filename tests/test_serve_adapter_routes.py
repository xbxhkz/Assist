from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.training_routes as tr


class FakeMgr:
    def list_adapters(self):
        return [{"run_id": "run-1", "complete": True, "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
                 "path": "p", "converted": True, "adapter_gguf": "p/adapter.gguf"}]


def _client(monkeypatch, **patches):
    monkeypatch.setattr(tr, "require_admin", lambda: None)
    monkeypatch.setattr(tr, "get_training_manager", lambda: FakeMgr())
    for k, v in patches.items():
        monkeypatch.setattr(tr, k, v)
    app = FastAPI(); app.include_router(tr.setup_training_routes())
    return TestClient(app)


def test_adapter_status(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/api/training/adapters/run-1")
    assert r.status_code == 200 and r.json()["converted"] is True


def test_convert_endpoint(monkeypatch):
    class Conv:
        def convert(self, d, base=None): return {"ok": True, "adapter_gguf": "p/adapter.gguf"}
    c = _client(monkeypatch, get_adapter_converter=lambda: Conv())
    r = c.post("/api/training/adapters/run-1/convert")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_convert_error_is_400(monkeypatch):
    class Conv:
        def convert(self, d, base=None): return {"error": "bad arch"}
    c = _client(monkeypatch, get_adapter_converter=lambda: Conv())
    r = c.post("/api/training/adapters/run-1/convert")
    assert r.status_code == 400


def test_serve_endpoint(monkeypatch):
    served = {}
    class LM:
        def start(self, path, device="cpu", lora=None, alias=None):
            served.update(path=path, lora=lora, alias=alias); return {"running": True}
    c = _client(monkeypatch, get_local_manager=lambda: LM())
    r = c.post("/api/training/adapters/run-1/serve", json={"base_gguf": "C:/m/qwen2.5-0.5b-instruct-q4_k_m.gguf"})
    assert r.status_code == 200
    assert served["lora"] == "p/adapter.gguf" and "run-1" in served["alias"]


def test_serve_surfaces_llama_server_failure_as_503(monkeypatch):
    # LocalModelManager.start RAISES RuntimeError when llama-server fails to come up
    # (the wrong/mismatched-base path). The route must surface it with the log tail,
    # never let it escape as a 500.
    class LM:
        def start(self, path, device="cpu", lora=None, alias=None):
            raise RuntimeError("llama-server exited on startup.\n--- tail ---\nbad base")
    c = _client(monkeypatch, get_local_manager=lambda: LM())
    r = c.post("/api/training/adapters/run-1/serve", json={"base_gguf": "b.gguf"})
    assert r.status_code == 503 and "bad base" in r.json()["detail"]


def test_serve_requires_converted(monkeypatch):
    class Mgr2(FakeMgr):
        def list_adapters(self):
            a = super().list_adapters(); a[0]["converted"] = False; a[0]["adapter_gguf"] = None; return a
    monkeypatch.setattr(tr, "get_training_manager", lambda: Mgr2())
    monkeypatch.setattr(tr, "require_admin", lambda: None)
    app = FastAPI(); app.include_router(tr.setup_training_routes())
    r = TestClient(app).post("/api/training/adapters/run-1/serve", json={"base_gguf": "b.gguf"})
    assert r.status_code == 400
