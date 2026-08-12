"""_execute_workflow_task registers/deregisters an active run around
run_workflow(), even when run_workflow raises. See
docs/superpowers/specs/2026-08-11-mission-control-workflow-runs-design.md.

_execute_workflow_task's body never references `self`, so it's called here
via the class directly with self=None -- a lightweight way to unit-test one
method without constructing a full TaskScheduler (which needs a live DB
session). Its 3 local imports (get_workflow, run_workflow,
resolve_trigger_inputs) are patched at their real source modules, since a
`from X import Y` done freshly inside the method picks up whatever X.Y is at
call time.
"""
import asyncio
from types import SimpleNamespace

import pytest

import src.task_scheduler as task_scheduler_module
import src.workflows.engine as engine_module
import src.workflows.store as store_module
from src import workflow_runs


def _fake_task(owner="alice", action="wf-1", prompt=None):
    return SimpleNamespace(owner=owner, action=action, prompt=prompt)


def test_execute_workflow_task_registers_and_deregisters_active_run(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})
    monkeypatch.setattr(
        store_module, "get_workflow",
        lambda wid: {"id": "wf-1", "name": "My Workflow", "nodes": [], "edges": []},
    )

    captured = {}

    async def fake_run_workflow(wf, inputs, ctx):
        captured["active_during_run"] = list(workflow_runs.list_active())
        return {"outputs": {}, "log": []}

    monkeypatch.setattr(engine_module, "run_workflow", fake_run_workflow)

    task = _fake_task()
    summary, success = asyncio.run(
        task_scheduler_module.TaskScheduler._execute_workflow_task(None, task, context=None)
    )

    assert len(captured["active_during_run"]) == 1
    entry = captured["active_during_run"][0]
    assert entry["workflow_id"] == "wf-1"
    assert entry["workflow_name"] == "My Workflow"
    assert entry["owner"] == "alice"
    assert entry["trigger"] == "scheduled"
    assert workflow_runs.list_active() == []
    assert success is True


def test_execute_workflow_task_deregisters_even_on_error(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})
    monkeypatch.setattr(
        store_module, "get_workflow",
        lambda wid: {"id": "wf-1", "name": "My Workflow", "nodes": [], "edges": []},
    )

    async def fake_run_workflow(wf, inputs, ctx):
        raise RuntimeError("boom")

    monkeypatch.setattr(engine_module, "run_workflow", fake_run_workflow)

    task = _fake_task()
    with pytest.raises(RuntimeError):
        asyncio.run(
            task_scheduler_module.TaskScheduler._execute_workflow_task(None, task, context=None)
        )

    assert workflow_runs.list_active() == []
