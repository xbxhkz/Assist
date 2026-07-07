"""Persistence for native-local image (diffusion) model endpoints.

Mirrors src/localmodels/store.py but tags rows with model_type="image" and an id
prefix `img-local-`. On register it synchronously probes the sd-server /v1/models
so the served model shows up in the picker immediately (same fix as the LLM path).
`session_factory`/`probe` are injectable for tests.
"""
import json
import uuid


def _default_probe(base_url: str):
    """Probe the just-started sd-server's /v1/models. Lazy import avoids a
    circular import at module load."""
    from routes.model_routes import _probe_endpoint
    return _probe_endpoint(base_url, None, timeout=5)


def _apply_cached_models(ep, base_url: str, probe) -> None:
    try:
        ids = probe(base_url)
        if ids:
            ep.cached_models = json.dumps(ids)
    except Exception:
        pass


def register_image_endpoint(name: str, base_url: str, session_factory=None,
                            probe=None) -> str:
    """Create or update the local image ModelEndpoint for `base_url`; return id."""
    from core.database import SessionLocal, ModelEndpoint
    sf = session_factory or SessionLocal
    probe = probe or _default_probe
    db = sf()
    try:
        existing = db.query(ModelEndpoint).filter(
            ModelEndpoint.base_url == base_url).first()
        if existing:
            existing.is_enabled = True
            existing.name = name
            existing.endpoint_kind = "local"
            existing.model_type = "image"
            existing.model_refresh_mode = "auto"
            _apply_cached_models(existing, base_url, probe)
            db.commit()
            return existing.id
        eid = f"img-local-{uuid.uuid4().hex[:8]}"
        ep = ModelEndpoint(id=eid, name=name, base_url=base_url, api_key=None,
                           is_enabled=True, endpoint_kind="local",
                           model_type="image", model_refresh_mode="auto")
        db.add(ep)
        _apply_cached_models(ep, base_url, probe)
        db.commit()
        return eid
    finally:
        db.close()


def unregister_image_endpoint(endpoint_id: str, session_factory=None) -> None:
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
