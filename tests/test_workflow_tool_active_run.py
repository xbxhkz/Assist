"""run_workflow_tool registers/deregisters an active run around run_workflow(),
even when run_workflow raises -- workflow_runs.finish() must always run via a
finally block. See
docs/superpowers/specs/2026-08-11-mission-control-workflow-runs-design.md."""
import asyncio

import pytest

import src.agent_tools.workflow_tool as workflow_tool_module
from src import workflow_runs


def test_run_workflow_tool_registers_and_deregisters_active_run(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})
    monkeypatch.setattr(
        workflow_tool_module.store, "get_workflow",
        lambda wid: {"id": "wf-1", "name": "My Workflow", "nodes": [], "edges": []},
    )

    captured = {}

    async def fake_run_workflow(wf, inputs, ctx):
        captured["active_during_run"] = list(workflow_runs.list_active())
        return {"outputs": {}, "log": []}

    monkeypatch.setattr(workflow_tool_module, "run_workflow", fake_run_workflow)

    asyncio.run(workflow_tool_module.run_workflow_tool('{"id": "wf-1"}', {"owner": "alice"}))

    assert len(captured["active_during_run"]) == 1
    entry = captured["active_during_run"][0]
    assert entry["workflow_id"] == "wf-1"
    assert entry["workflow_name"] == "My Workflow"
    assert entry["owner"] == "alice"
    assert entry["trigger"] == "agent_tool"
    assert workflow_runs.list_active() == []


def test_run_workflow_tool_deregisters_even_on_error(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})
    monkeypatch.setattr(
        workflow_tool_module.store, "get_workflow",
        lambda wid: {"id": "wf-1", "name": "My Workflow", "nodes": [], "edges": []},
    )

    async def fake_run_workflow(wf, inputs, ctx):
        raise RuntimeError("boom")

    monkeypatch.setattr(workflow_tool_module, "run_workflow", fake_run_workflow)

    with pytest.raises(RuntimeError):
        asyncio.run(workflow_tool_module.run_workflow_tool('{"id": "wf-1"}', {"owner": "alice"}))

    assert workflow_runs.list_active() == []
