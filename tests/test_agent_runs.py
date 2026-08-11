"""list_active() enumerates running session ids -- a pure read added
alongside src.agent_runs' existing per-session run tracking, without
touching its write path (start/stop/eviction). See
docs/superpowers/specs/2026-08-11-mission-control-active-agents-design.md.
"""
from src import agent_runs


def _put_run(session_id, status):
    run = agent_runs._Run()
    run.status = status
    agent_runs._RUNS[session_id] = run


def test_list_active_returns_only_running_sessions(monkeypatch):
    monkeypatch.setattr(agent_runs, "_RUNS", {})
    _put_run("running-1", "running")
    _put_run("done-1", "done")
    _put_run("error-1", "error")
    _put_run("stopped-1", "stopped")
    _put_run("running-2", "running")

    assert sorted(agent_runs.list_active()) == ["running-1", "running-2"]


def test_list_active_empty_when_nothing_running(monkeypatch):
    monkeypatch.setattr(agent_runs, "_RUNS", {})
    _put_run("done-1", "done")

    assert agent_runs.list_active() == []
