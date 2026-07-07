"""register_image_endpoint creates a ModelEndpoint(model_type='image'). Uses an
in-memory SQLite session so it exercises the real ORM."""
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.database import Base, ModelEndpoint
import src.imagemodels.store as store


def _mem_session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def test_register_creates_image_endpoint():
    sf = _mem_session_factory()
    eid = store.register_image_endpoint("flux.gguf", "http://127.0.0.1:8200/v1",
                                        session_factory=sf, probe=lambda u: ["flux.gguf"])
    assert eid.startswith("img-local-")
    db = sf()
    ep = db.query(ModelEndpoint).filter(ModelEndpoint.id == eid).one()
    assert ep.base_url == "http://127.0.0.1:8200/v1"
    assert ep.model_type == "image"
    assert ep.endpoint_kind == "local"
    assert ep.is_enabled is True
    assert ep.cached_models == json.dumps(["flux.gguf"])
    db.close()


def test_register_updates_existing_same_url():
    sf = _mem_session_factory()
    first = store.register_image_endpoint("a.gguf", "http://127.0.0.1:8200/v1",
                                          session_factory=sf, probe=lambda u: ["a"])
    second = store.register_image_endpoint("b.gguf", "http://127.0.0.1:8200/v1",
                                           session_factory=sf, probe=lambda u: ["b"])
    assert first == second
    db = sf()
    assert db.query(ModelEndpoint).count() == 1
    assert db.query(ModelEndpoint).one().name == "b.gguf"
    db.close()


def test_unregister_deletes_row():
    sf = _mem_session_factory()
    eid = store.register_image_endpoint("a.gguf", "http://127.0.0.1:8200/v1",
                                        session_factory=sf, probe=lambda u: [])
    store.unregister_image_endpoint(eid, session_factory=sf)
    db = sf()
    assert db.query(ModelEndpoint).count() == 0
    db.close()
