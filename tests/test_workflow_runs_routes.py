"""GET /api/workflow-runs/active exposes workflow_runs.list_active(), gated by
the same admin-only dependency the rest of the workflows subsystem uses
(routes/workflow_routes.py's own router). See
docs/superpowers/specs/2026-08-11-mission-control-workflow-runs-design.md."""
import asyncio

from core.middleware import require_admin
import routes.workflow_runs_routes as workflow_runs_routes_module
from src import workflow_runs


def _route(router, path, method):
    for r in router.routes:
        if r.path == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(path)


def test_route_returns_active_workflow_runs(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})
    workflow_runs.start("wf-1", "My Workflow", "alice", "api")

    router = workflow_runs_routes_module.setup_workflow_runs_routes()
    handler = _route(router, "/api/workflow-runs/active", "GET")

    out = asyncio.run(handler())

    assert len(out["active"]) == 1
    assert out["active"][0]["workflow_id"] == "wf-1"


def test_router_is_admin_gated():
    router = workflow_runs_routes_module.setup_workflow_runs_routes()
    dependency_callables = [d.dependency for d in router.dependencies]
    assert require_admin in dependency_callables
