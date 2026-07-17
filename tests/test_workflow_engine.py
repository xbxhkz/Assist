import asyncio
import pytest
import src.workflows.engine as eng
from src.workflows.model import WorkflowError


def _run(coro):
    return asyncio.run(coro)


def _wf():
    return {"id": "w", "name": "W", "nodes": [
        {"id": "i", "type": "input", "config": {"name": "q"}},
        {"id": "t", "type": "template", "config": {"template": "Q: {q}"}},
        {"id": "l", "type": "llm", "config": {"prompt": "{p}"}},
        {"id": "o", "type": "output", "config": {"name": "answer"}},
    ], "edges": [
        {"from_node": "i", "from_port": "value", "to_node": "t", "to_port": "q"},
        {"from_node": "t", "from_port": "text", "to_node": "l", "to_port": "p"},
        {"from_node": "l", "from_port": "text", "to_node": "o", "to_port": "value"},
    ]}


async def _fake_model(prompt, model=None, system=None):
    return f"ECHO[{prompt}]"


def test_runs_linear_workflow_and_logs():
    res = _run(eng.run_workflow(_wf(), {"q": "why"}, model_call=_fake_model))
    assert res["outputs"] == {"answer": "ECHO[Q: why]"}
    assert [e["status"] for e in res["log"]] == ["ok", "ok", "ok", "ok"]
    assert [e["node"] for e in res["log"]] == ["i", "t", "l", "o"]
    assert all("ms" in e for e in res["log"])


def test_invalid_graph_raises():
    bad = _wf()
    bad["edges"].append({"from_node": "zz", "from_port": "value", "to_node": "o", "to_port": "value"})
    with pytest.raises(WorkflowError):
        _run(eng.run_workflow(bad, {}, model_call=_fake_model))


def test_node_failure_skips_dependents_and_returns_partial():
    async def boom(prompt, model=None, system=None):
        raise RuntimeError("model down")
    res = _run(eng.run_workflow(_wf(), {"q": "why"}, model_call=boom))
    status = {e["node"]: e["status"] for e in res["log"]}
    assert status["i"] == "ok" and status["t"] == "ok"
    assert status["l"] == "error" and status["o"] == "skipped"
    assert "model down" in (next(e for e in res["log"] if e["node"] == "l")["error"] or "")
    assert res["outputs"] == {}          # partial: the output never resolved


def test_tool_node_uses_injected_dispatch():
    wf = {"id": "w", "name": "W", "nodes": [
        {"id": "i", "type": "input", "config": {"name": "p"}},
        {"id": "tl", "type": "tool", "config": {"tool": "find_files", "args": "{p}"}},
        {"id": "o", "type": "output", "config": {"name": "files"}},
    ], "edges": [
        {"from_node": "i", "from_port": "value", "to_node": "tl", "to_port": "p"},
        {"from_node": "tl", "from_port": "result", "to_node": "o", "to_port": "value"},
    ]}
    async def fake_dispatch(tool, args, ctx):
        return f"{tool}:{args}"
    res = _run(eng.run_workflow(wf, {"p": "/tmp"}, {"owner": "u"}, tool_dispatch=fake_dispatch))
    assert res["outputs"] == {"files": "find_files:/tmp"}
