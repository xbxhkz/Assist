"""workflow_runs tracks currently-executing workflow runs in memory -- a live
snapshot mirroring src/agent_runs.py's own established pattern (Mission
Control sub-project 2b). No persisted history: a run disappears from
list_active() the instant finish() is called, success or failure alike. See
docs/superpowers/specs/2026-08-11-mission-control-workflow-runs-design.md.
"""
from src import workflow_runs


def test_start_adds_a_running_entry(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})

    run_id = workflow_runs.start("wf-1", "My Workflow", "alice", "api")

    active = workflow_runs.list_active()
    matching = [r for r in active if r["run_id"] == run_id]
    assert len(matching) == 1
    entry = matching[0]
    assert entry["workflow_id"] == "wf-1"
    assert entry["workflow_name"] == "My Workflow"
    assert entry["owner"] == "alice"
    assert entry["trigger"] == "api"
    assert "started_at" in entry


def test_finish_removes_the_entry(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})
    run_id = workflow_runs.start("wf-1", "My Workflow", "alice", "api")

    workflow_runs.finish(run_id)

    assert workflow_runs.list_active() == []


def test_finish_on_unknown_run_id_does_not_raise(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})

    workflow_runs.finish("does-not-exist")  # must not raise


def test_concurrent_runs_of_the_same_workflow_get_distinct_entries(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})

    run_id_1 = workflow_runs.start("wf-1", "My Workflow", "alice", "api")
    run_id_2 = workflow_runs.start("wf-1", "My Workflow", "bob", "scheduled")

    assert run_id_1 != run_id_2
    ids = {r["run_id"] for r in workflow_runs.list_active()}
    assert run_id_1 in ids
    assert run_id_2 in ids
