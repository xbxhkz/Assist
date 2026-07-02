"""Persistence for native-local model endpoints (Phase 3a).

Mirrors the auto-register pattern in routes/cookbook_routes.py, but tags rows
with endpoint_kind="local" and an id prefix `local-` so native-local endpoints
are distinct from Cookbook ones. `session_factory` is injectable for tests.
"""
import uuid


def register_local_endpoint(name: str, base_url: str, session_factory=None) -> str:
    """Create or update the local ModelEndpoint for `base_url`; return its id."""
    from core.database import SessionLocal, ModelEndpoint
    sf = session_factory or SessionLocal
    db = sf()
    try:
        existing = db.query(ModelEndpoint).filter(
            ModelEndpoint.base_url == base_url).first()
        if existing:
            existing.is_enabled = True
            existing.name = name
            existing.endpoint_kind = "local"
            db.commit()
            return existing.id
        eid = f"local-{uuid.uuid4().hex[:8]}"
        ep = ModelEndpoint(id=eid, name=name, base_url=base_url, api_key=None,
                           is_enabled=True, endpoint_kind="local")
        db.add(ep)
        db.commit()
        return eid
    finally:
        db.close()


def unregister_local_endpoint(endpoint_id: str, session_factory=None) -> None:
    """Delete the ModelEndpoint row with `endpoint_id`, if present."""
    from core.database import SessionLocal, ModelEndpoint
    sf = session_factory or SessionLocal
    db = sf()
    try:
        ep = db.query(ModelEndpoint).filter(
            ModelEndpoint.id == endpoint_id).first()
        if ep:
            db.delete(ep)
            db.commit()
    finally:
        db.close()
