"""Persistence for native-local model endpoints (Phase 3a).

Mirrors the auto-register pattern in routes/cookbook_routes.py, but tags rows
with endpoint_kind="local" and an id prefix `local-` so native-local endpoints
are distinct from Cookbook ones. `session_factory` is injectable for tests.
"""
import json
import uuid


def _default_probe(base_url: str):
    """Probe the just-started llama-server's /v1/models. Lazy import avoids a
    circular import at module load (routes.model_routes imports the DB layer)."""
    from routes.model_routes import _probe_endpoint
    return _probe_endpoint(base_url, None, timeout=5)


def _apply_cached_models(ep, base_url: str, probe) -> None:
    """Synchronously probe /v1/models and store the result on `ep.cached_models`.

    Without this the chat picker treats the endpoint as offline (empty model
    list) until the next background refresh fires seconds later — so a freshly
    served model appears to "not show up". Mirrors the Cookbook auto-register's
    probe-after-create. A probe failure is non-fatal: the endpoint is still
    registered and the background refresh will fill the cache later.
    """
    try:
        ids = probe(base_url)
        if ids:
            ep.cached_models = json.dumps(ids)
    except Exception:
        pass


def register_local_endpoint(name: str, base_url: str, session_factory=None,
                            probe=None) -> str:
    """Create or update the local ModelEndpoint for `base_url`; return its id.

    Probes the endpoint's /v1/models immediately so the served model shows up
    in the chat picker on the next /api/models call. `probe` is injectable for
    tests; it defaults to a live probe of the running llama-server.
    """
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
            existing.model_refresh_mode = "auto"
            _apply_cached_models(existing, base_url, probe)
            db.commit()
            return existing.id
        eid = f"local-{uuid.uuid4().hex[:8]}"
        ep = ModelEndpoint(id=eid, name=name, base_url=base_url, api_key=None,
                           is_enabled=True, endpoint_kind="local",
                           model_refresh_mode="auto")
        db.add(ep)
        _apply_cached_models(ep, base_url, probe)
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
