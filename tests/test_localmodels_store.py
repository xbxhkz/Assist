"""The local-endpoint store creates/removes a ModelEndpoint row (endpoint_kind
='local'). Uses an in-memory SQLite session so it exercises the real ORM."""
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
