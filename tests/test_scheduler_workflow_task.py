import asyncio
import types

import pytest

import src.workflows.store as store
import src.workflows.engine as engine
from src.task_scheduler import TaskScheduler


def _run(coro):
    return asyncio.run(coro)


def _bare_scheduler():
    # Bypass __init__ — _execute_workflow_task uses no instance state.
    return TaskScheduler.__new__(TaskScheduler)


def _task(action="flow1", prompt='{"topic": "AI"}', owner="admin"):
    return types.SimpleNamespace(action=action, prompt=prompt, owner=owner, name="t")


def test_missing_workflow_returns_error_not_raise(monkeypatch):
    monkeypatch.setattr(store, "get_workflow", lambda wid: None)
    s = _bare_scheduler()
    result, success = _run(s._execute_workflow_task(_task(), context=None))
    assert success is False and "not found" in result


def test_runs_workflow_and_summarizes_success(monkeypatch):
    wf = {"nodes": [{"id": "i", "type": "input", "config": {"name": "topic"}}]}
    monkeypatch.setattr(store, "get_workflow", lambda wid: wf)
    seen = {}

    async def fake_run(w, inputs, ctx, **kw):
        seen["inputs"] = inputs
        seen["ctx"] = ctx
        return {"outputs": {"answer": "hi"}, "log": [{"node": "o", "status": "ok"}]}
    monkeypatch.setattr(engine, "run_workflow", fake_run)

    s = _bare_scheduler()
    result, success = _run(s._execute_workflow_task(_task(prompt='{"topic":"cats"}'),
                                                     context={"topic": "dogs"}))
    assert success is True
    assert "answer=hi" in result
    assert seen["inputs"] == {"topic": "dogs"}     # context overrides fixed
    assert seen["ctx"] == {"owner": "admin"}


def test_node_error_marks_run_failed(monkeypatch):
    wf = {"nodes": []}
    monkeypatch.setattr(store, "get_workflow", lambda wid: wf)

    async def fake_run(w, inputs, ctx, **kw):
        return {"outputs": {}, "log": [{"node": "l", "status": "error", "error": "boom"}]}
    monkeypatch.setattr(engine, "run_workflow", fake_run)

    s = _bare_scheduler()
    result, success = _run(s._execute_workflow_task(_task(prompt=None), context=None))
    assert success is False and "1 error" in result


def test_run_task_now_threads_context(monkeypatch):
    s = TaskScheduler.__new__(TaskScheduler)
    s._executing = set()
    s._executing_lock = asyncio.Lock()
    captured = {}

    async def fake_execute(task_id, **kw):
        captured["task_id"] = task_id
        captured["context"] = kw.get("context")
    s._execute_task = fake_execute

    async def go():
        await s.run_task_now("t1", context={"a": 1})
        await asyncio.sleep(0)      # let the created task run
    _run(go())
    assert captured["task_id"] == "t1" and captured["context"] == {"a": 1}
