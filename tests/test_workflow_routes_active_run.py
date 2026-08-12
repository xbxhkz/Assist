"""POST /api/workflows/{wid}/run registers/deregisters an active run around
run_workflow(), even when run_workflow raises. See
docs/superpowers/specs/2026-08-11-mission-control-workflow-runs-design.md."""
import asyncio
from types import SimpleNamespace

import pytest

import routes.workflow_routes as workflow_routes_module
from src import workflow_runs


def _route(router, path, method):
    for r in router.routes:
        if r.path == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(path)


def _request(user):
    return SimpleNamespace(state=SimpleNamespace(current_user=user))


def test_run_route_registers_and_deregisters_active_run(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})
    monkeypatch.setattr(
        workflow_routes_module.store, "get_workflow",
        lambda wid: {"id": "wf-1", "name": "My Workflow", "nodes": [], "edges": []},
    )

    captured = {}

    async def fake_run_workflow(wf, inputs, ctx):
        captured["active_during_run"] = list(workflow_runs.list_active())
        return {"outputs": {}, "log": []}

    monkeypatch.setattr(workflow_routes_module, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(workflow_routes_module, "get_current_user", lambda request: "alice")

    router = workflow_routes_module.setup_workflow_routes()
    handler = _route(router, "/api/workflows/{wid}/run", "POST")

    asyncio.run(handler(wid="wf-1", request=_request("alice"), body={}))

    assert len(captured["active_during_run"]) == 1
    entry = captured["active_during_run"][0]
    assert entry["workflow_id"] == "wf-1"
    assert entry["workflow_name"] == "My Workflow"
    assert entry["owner"] == "alice"
    assert entry["trigger"] == "api"
    assert workflow_runs.list_active() == []


def test_run_route_deregisters_even_on_error(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})
    monkeypatch.setattr(
        workflow_routes_module.store, "get_workflow",
        lambda wid: {"id": "wf-1", "name": "My Workflow", "nodes": [], "edges": []},
    )

    async def fake_run_workflow(wf, inputs, ctx):
        raise RuntimeError("boom")

    monkeypatch.setattr(workflow_routes_module, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(workflow_routes_module, "get_current_user", lambda request: "alice")

    router = workflow_routes_module.setup_workflow_routes()
    handler = _route(router, "/api/workflows/{wid}/run", "POST")

    with pytest.raises(RuntimeError):
        asyncio.run(handler(wid="wf-1", request=_request("alice"), body={}))

    assert workflow_runs.list_active() == []
