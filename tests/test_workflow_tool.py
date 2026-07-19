import asyncio
import json

import src.agent_tools.workflow_tool as wt


def _run(coro):
    return asyncio.run(coro)


def _exec(content, ctx=None):
    return _run(wt.RunWorkflowTool().execute(content, ctx or {}))


def test_list_mode_returns_workflows_with_input_names(monkeypatch):
    monkeypatch.setattr(wt.store, "list_workflows", lambda: [{"id": "flow1", "name": "Flow One", "nodes": 2}])
    monkeypatch.setattr(wt.store, "get_workflow", lambda wid: {
        "id": "flow1", "name": "Flow One",
        "nodes": [{"id": "i", "type": "input", "config": {"name": "topic"}},
                  {"id": "o", "type": "output", "config": {"name": "answer"}}], "edges": []})
    out = _exec("")
    data = json.loads(out["output"])
    assert data["workflows"] == [{"id": "flow1", "name": "Flow One", "inputs": ["topic"]}]


def test_action_list_also_lists(monkeypatch):
    monkeypatch.setattr(wt.store, "list_workflows", lambda: [])
    out = _exec(json.dumps({"action": "list"}))
    assert json.loads(out["output"]) == {"workflows": []}


def test_list_mode_tolerates_a_path_unsafe_id_without_raising(monkeypatch):
    # a hand-edited workflow file whose JSON id is path-unsafe must not crash the whole listing
    monkeypatch.setattr(wt.store, "list_workflows",
                        lambda: [{"id": "../evil", "name": "Bad"}, {"id": "ok", "name": "OK"}])

    def _get(wid):
        if wid == "../evil":
            raise ValueError("unsafe workflow id")
        return {"nodes": [{"id": "i", "type": "input", "config": {"name": "q"}}]}
    monkeypatch.setattr(wt.store, "get_workflow", _get)
    out = _exec("")            # must return, not raise
    data = json.loads(out["output"])
    assert data["workflows"] == [{"id": "../evil", "name": "Bad", "inputs": []},
                                 {"id": "ok", "name": "OK", "inputs": ["q"]}]


def test_run_mode_runs_and_summarizes(monkeypatch):
    wf = {"id": "flow1", "name": "F", "nodes": [], "edges": []}
    monkeypatch.setattr(wt.store, "get_workflow", lambda wid: wf if wid == "flow1" else None)
    seen = {}

    async def fake_run(w, inputs, ctx, **kw):
        seen["inputs"] = inputs
        seen["ctx"] = ctx
        return {"outputs": {"answer": "hi"}, "log": [{"node": "o", "status": "ok"}]}
    monkeypatch.setattr(wt, "run_workflow", fake_run)

    out = _exec(json.dumps({"id": "flow1", "inputs": {"topic": "AI"}}), {"owner": "admin"})
    assert "answer" in out["output"] and "1 ok" in out["output"]
    assert seen["inputs"] == {"topic": "AI"}
    assert seen["ctx"]["owner"] == "admin" and seen["ctx"]["_in_workflow"] is True


def test_missing_workflow_is_error_not_raise(monkeypatch):
    monkeypatch.setattr(wt.store, "get_workflow", lambda wid: None)
    out = _exec(json.dumps({"id": "nope"}))
    assert "not found" in out["error"]


def test_invalid_graph_surfaces_as_error(monkeypatch):
    monkeypatch.setattr(wt.store, "get_workflow", lambda wid: {"nodes": [], "edges": []})

    async def bad_run(w, inputs, ctx, **kw):
        raise wt.WorkflowError(["cycle detected"])
    monkeypatch.setattr(wt, "run_workflow", bad_run)
    out = _exec(json.dumps({"id": "flow1"}))
    assert "cycle detected" in out["error"]


def test_recursion_guard_refuses_when_in_workflow(monkeypatch):
    # even a run request refuses if called from inside a workflow
    called = {"run": False}

    async def fake_run(*a, **k):
        called["run"] = True
        return {"outputs": {}, "log": []}
    monkeypatch.setattr(wt, "run_workflow", fake_run)
    out = _exec(json.dumps({"id": "flow1"}), {"_in_workflow": True})
    assert "error" in out and called["run"] is False


def test_path_unsafe_id_is_error_not_raise():
    # Exercises the real store._safe_id guard (no monkeypatch of get_workflow) —
    # a path-unsafe id must be rejected before any file access, so this never
    # touches the filesystem. store.get_workflow raises ValueError for these;
    # the handler must convert that into a returned {"error": ...}, not raise.
    for bad_id in ("../evil", "a:b"):
        out = _exec(json.dumps({"id": bad_id}))
        assert isinstance(out, dict)
        assert "error" in out


def test_child_ctx_does_not_mutate_caller(monkeypatch):
    monkeypatch.setattr(wt.store, "get_workflow", lambda wid: {"nodes": [], "edges": []})

    async def fake_run(w, inputs, ctx, **kw):
        return {"outputs": {}, "log": []}
    monkeypatch.setattr(wt, "run_workflow", fake_run)
    caller_ctx = {"owner": "admin"}
    _exec(json.dumps({"id": "flow1"}), caller_ctx)
    assert "_in_workflow" not in caller_ctx     # caller's ctx untouched
