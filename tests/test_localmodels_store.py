"""The local-endpoint store creates/removes a ModelEndpoint row (endpoint_kind
='local'). Uses an in-memory SQLite session so it exercises the real ORM."""
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.database import Base, ModelEndpoint
import src.localmodels.store as store


def _mem_session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def test_register_creates_local_endpoint():
    sf = _mem_session_factory()
    eid = store.register_local_endpoint("m.gguf", "http://127.0.0.1:8123/v1",
                                        session_factory=sf)
    assert eid.startswith("local-")
    db = sf()
    ep = db.query(ModelEndpoint).filter(ModelEndpoint.id == eid).one()
    assert ep.base_url == "http://127.0.0.1:8123/v1"
    assert ep.endpoint_kind == "local"
    assert ep.is_enabled is True
    db.close()


def test_register_marks_supports_tools_true():
    """Local llama.cpp (llama-server) endpoints support native OpenAI tool
    calling; without supports_tools=True the agent falls back to fenced-block
    mode where local models often just *reason* about a tool instead of
    emitting the call (agent tools silently no-op)."""
    sf = _mem_session_factory()
    eid = store.register_local_endpoint("m.gguf", "http://127.0.0.1:8123/v1",
                                        session_factory=sf)
    db = sf()
    assert db.query(ModelEndpoint).filter(ModelEndpoint.id == eid).one().supports_tools is True
    db.close()


def test_register_updates_existing_same_url():
    sf = _mem_session_factory()
    first = store.register_local_endpoint("a.gguf", "http://127.0.0.1:8123/v1",
                                          session_factory=sf)
    second = store.register_local_endpoint("b.gguf", "http://127.0.0.1:8123/v1",
                                           session_factory=sf)
    assert first == second  # same row reused for the same base_url
    db = sf()
    assert db.query(ModelEndpoint).count() == 1
    assert db.query(ModelEndpoint).one().name == "b.gguf"
    db.close()


def test_unregister_deletes_row():
    sf = _mem_session_factory()
    eid = store.register_local_endpoint("a.gguf", "http://127.0.0.1:8123/v1",
                                        session_factory=sf)
    store.unregister_local_endpoint(eid, session_factory=sf)
    db = sf()
    assert db.query(ModelEndpoint).count() == 0
    db.close()


def test_prune_serve_endpoints_removes_only_per_serve_rows():
    """Per-serve rows (local-*/img-local-*) die with their process — at boot
    they are garbage by construction (dynamic ports), and letting them pile
    up broke Deep Research (its fallback resolved to a dead endpoint) and
    spammed probe failures. Manual endpoints must survive the sweep."""
    sf = _mem_session_factory()
    store.register_local_endpoint("a.gguf", "http://127.0.0.1:50001/v1",
                                  session_factory=sf)
    db = sf()
    db.add(ModelEndpoint(id="img-local-abc", name="flux", is_enabled=True,
                         base_url="http://127.0.0.1:50002/v1"))
    db.add(ModelEndpoint(id="manual-ollama", name="Ollama", is_enabled=True,
                         base_url="http://127.0.0.1:11434/v1"))
    db.commit(); db.close()

    removed = store.prune_serve_endpoints(session_factory=sf)
    assert removed == 2
    db = sf()
    assert [e.id for e in db.query(ModelEndpoint).all()] == ["manual-ollama"]
    db.close()


def test_register_populates_cached_models_from_probe():
    """Registering must synchronously probe /v1/models and persist the result
    to cached_models, so the freshly-served model appears in the chat picker on
    the next /api/models call instead of being 'offline' until the ~seconds-
    later background refresh fires. Mirrors the Cookbook auto-register."""
    sf = _mem_session_factory()
    probed = []

    def fake_probe(base_url):
        probed.append(base_url)
        return ["served-model-id"]

    eid = store.register_local_endpoint(
        "m.gguf", "http://127.0.0.1:8123/v1", session_factory=sf, probe=fake_probe)
    assert probed == ["http://127.0.0.1:8123/v1"]  # probed exactly once
    db = sf()
    ep = db.query(ModelEndpoint).filter(ModelEndpoint.id == eid).one()
    assert ep.cached_models == json.dumps(["served-model-id"])
    db.close()


def test_register_update_path_repopulates_cached_models():
    """Re-registering the same base_url (e.g. re-serving) must refresh
    cached_models too, not just on first create."""
    sf = _mem_session_factory()
    store.register_local_endpoint(
        "a.gguf", "http://127.0.0.1:8123/v1", session_factory=sf,
        probe=lambda url: ["old-id"])
    store.register_local_endpoint(
        "b.gguf", "http://127.0.0.1:8123/v1", session_factory=sf,
        probe=lambda url: ["new-id"])
    db = sf()
    ep = db.query(ModelEndpoint).one()
    assert ep.cached_models == json.dumps(["new-id"])
    db.close()


def test_register_survives_probe_failure():
    """A probe that raises (server briefly unreachable) must not break
    registration; the endpoint is still created, just without cached_models."""
    sf = _mem_session_factory()

    def boom(base_url):
        raise RuntimeError("connection refused")

    eid = store.register_local_endpoint(
        "m.gguf", "http://127.0.0.1:8123/v1", session_factory=sf, probe=boom)
    db = sf()
    ep = db.query(ModelEndpoint).filter(ModelEndpoint.id == eid).one()
    assert ep.cached_models is None
    assert ep.is_enabled is True
    db.close()
