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


# --- served-local-model fallback (fixes "no model endpoint configured" when a
#     local model is served but no Default Chat Model is set in Settings) ---
import asyncio


class _FakeMgr:
    def __init__(self, st): self._st = st
    def status(self): return self._st


def test_served_local_endpoint_resolves_running_model():
    mgr = _FakeMgr({"running": True, "endpoint_id": "ep1", "model": "qwen.gguf"})
    seen = {}

    def fake_resolve(ep_id, model, owner=None):
        seen["args"] = (ep_id, model, owner)
        return ("http://127.0.0.1:8100/v1/chat/completions", "qwen-real-id", {})

    url, model, headers = dr._served_local_endpoint(owner="admin", manager=mgr, resolve_by_id=fake_resolve)
    assert url.endswith("/chat/completions") and model == "qwen-real-id"
    assert seen["args"][0] == "ep1"  # resolved by the manager-tracked endpoint id


def test_served_local_endpoint_none_when_not_running():
    mgr = _FakeMgr({"running": False, "endpoint_id": None, "model": None})
    assert dr._served_local_endpoint(manager=mgr, resolve_by_id=lambda *a, **k: ("u", "m", {})) == (None, None, None)


def test_served_local_endpoint_never_raises_on_bad_resolver():
    mgr = _FakeMgr({"running": True, "endpoint_id": "ep1", "model": "m"})
    def boom(*a, **k): raise RuntimeError("db down")
    assert dr._served_local_endpoint(manager=mgr, resolve_by_id=boom) == (None, None, None)


def test_default_model_call_errors_clearly_when_nothing_available(monkeypatch):
    import src.endpoint_resolver as er
    monkeypatch.setattr(er, "resolve_endpoint", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(dr, "_served_local_endpoint", lambda *a, **k: (None, None, None))
    try:
        asyncio.run(dr._default_model_call("hi"))
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "serve a local model" in str(e).lower() or "default chat model" in str(e).lower()


def test_served_local_endpoint_port_fallback_when_id_unresolvable():
    # endpoint id present but unresolvable (e.g. empty model list) -> build from port
    mgr = _FakeMgr({"running": True, "endpoint_id": "ep1", "model": "qwen.gguf", "port": 8137})
    url, model, headers = dr._served_local_endpoint(manager=mgr, resolve_by_id=lambda *a, **k: None)
    assert url == "http://127.0.0.1:8137/v1/chat/completions" and model == "qwen.gguf"


def test_default_model_call_resolves_dataset_generation_prefix_first(monkeypatch):
    import src.endpoint_resolver as er
    calls = []

    def fake_resolve(prefix, *a, **k):
        calls.append(prefix)
        if prefix == "dataset_generation":
            return ("http://ep/v1/chat/completions", "pinned-model", {})
        return (None, None, None)

    monkeypatch.setattr(er, "resolve_endpoint", fake_resolve)
    # served-local fallback must NOT be consulted when the new prefix resolves
    monkeypatch.setattr(dr, "_served_local_endpoint", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("should not fall back to served-local when dataset_generation resolves")))

    class _FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "hi"}}]}

    class _FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k): return _FakeResp()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", lambda **k: _FakeClient())

    result = asyncio.run(dr._default_model_call("hi"))
    assert calls[0] == "dataset_generation"
    assert result == "hi"
