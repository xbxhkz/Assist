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


def test_transitive_skip_propagates_through_chain():
    # a -> b -> c -> d ; b fails. c's only upstream is b (failed), d's only
    # upstream is c (skipped, not failed) -- this only skips if the skip path
    # itself marks c as failed too.
    wf = {"id": "w", "name": "W", "nodes": [
        {"id": "a", "type": "input", "config": {"name": "q"}},
        {"id": "b", "type": "llm", "config": {"prompt": "{q}"}},
        {"id": "c", "type": "template", "config": {"template": "got {x}"}},
        {"id": "d", "type": "template", "config": {"template": "final {y}"}},
    ], "edges": [
        {"from_node": "a", "from_port": "value", "to_node": "b", "to_port": "q"},
        {"from_node": "b", "from_port": "text", "to_node": "c", "to_port": "x"},
        {"from_node": "c", "from_port": "text", "to_node": "d", "to_port": "y"},
    ]}

    async def boom(prompt, model=None, system=None):
        raise RuntimeError("model down")

    res = _run(eng.run_workflow(wf, {"q": "why"}, model_call=boom))
    status = {e["node"]: e["status"] for e in res["log"]}
    assert status["a"] == "ok"
    assert status["b"] == "error"
    assert status["c"] == "skipped"
    assert status["d"] == "skipped"


def test_diamond_fanin_skips_only_via_failed_branch():
    # a -> b, a -> c, b -> d, c -> d ; only b fails. c succeeds but d must
    # still be skipped because one of its two upstreams (b) failed.
    wf = {"id": "w", "name": "W", "nodes": [
        {"id": "a", "type": "input", "config": {"name": "q"}},
        {"id": "b", "type": "llm", "config": {"prompt": "{q}"}},
        {"id": "c", "type": "template", "config": {"template": "c:{q}"}},
        {"id": "d", "type": "template", "config": {"template": "{bv}-{cv}"}},
    ], "edges": [
        {"from_node": "a", "from_port": "value", "to_node": "b", "to_port": "q"},
        {"from_node": "a", "from_port": "value", "to_node": "c", "to_port": "q"},
        {"from_node": "b", "from_port": "text", "to_node": "d", "to_port": "bv"},
        {"from_node": "c", "from_port": "text", "to_node": "d", "to_port": "cv"},
    ]}

    async def boom(prompt, model=None, system=None):
        raise RuntimeError("model down")

    res = _run(eng.run_workflow(wf, {"q": "why"}, model_call=boom))
    status = {e["node"]: e["status"] for e in res["log"]}
    assert status["a"] == "ok"
    assert status["b"] == "error"
    assert status["c"] == "ok"
    assert status["d"] == "skipped"


def test_log_output_field_for_template_and_output_nodes():
    res = _run(eng.run_workflow(_wf(), {"q": "why"}, model_call=_fake_model))
    by_node = {e["node"]: e for e in res["log"]}
    assert by_node["t"]["output"] == "Q: why"
    # output node has no produced value of its own; its logged "output" falls
    # back to the recorded input value rather than being blank.
    assert by_node["o"]["output"] == "ECHO[Q: why]"


def test_log_output_is_truncated_to_LOG_MAX():
    long_value = "x" * (eng._LOG_MAX + 250)

    async def fake_model(prompt, model=None, system=None):
        return long_value

    res = _run(eng.run_workflow(_wf(), {"q": "why"}, model_call=fake_model))
    by_node = {e["node"]: e for e in res["log"]}
    assert len(by_node["l"]["output"]) == eng._LOG_MAX
    assert len(by_node["o"]["output"]) == eng._LOG_MAX


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
