import asyncio
import src.workflows.engine as eng


def _run(coro):
    return asyncio.run(coro)


def _wf():
    # input -> branch(match) -> [yes -> out_yes] / [no -> out_no]
    return {"id": "w", "name": "W", "nodes": [
        {"id": "i", "type": "input", "config": {"name": "q"}},
        {"id": "b", "type": "branch", "config": {"mode": "match", "cases": ["yes", "no"]}},
        {"id": "ty", "type": "template", "config": {"template": "Y:{v}"}},
        {"id": "oy", "type": "output", "config": {"name": "yes_out"}},
        {"id": "on", "type": "output", "config": {"name": "no_out"}},
    ], "edges": [
        {"from_node": "i", "from_port": "value", "to_node": "b", "to_port": "value"},
        {"from_node": "b", "from_port": "yes", "to_node": "ty", "to_port": "v"},
        {"from_node": "ty", "from_port": "text", "to_node": "oy", "to_port": "value"},
        {"from_node": "b", "from_port": "no", "to_node": "on", "to_port": "value"},
    ]}


def test_taken_branch_runs_untaken_skips_and_cascades():
    res = _run(eng.run_workflow(_wf(), {"q": "yes"}))
    status = {e["node"]: e["status"] for e in res["log"]}
    assert status["b"] == "ok"
    assert status["ty"] == "ok" and status["oy"] == "ok"   # taken path (+ cascade through ty)
    assert status["on"] == "skipped"                        # un-taken path
    assert res["outputs"] == {"yes_out": "Y:yes"}           # only the taken output resolved


def test_other_branch_taken():
    res = _run(eng.run_workflow(_wf(), {"q": "no"}))
    status = {e["node"]: e["status"] for e in res["log"]}
    assert status["on"] == "ok"
    assert status["ty"] == "skipped" and status["oy"] == "skipped"   # cascade skip
    assert res["outputs"] == {"no_out": "no"}


def test_llm_branch_routes():
    async def fake_model(prompt, model=None, system=None):
        return "no"
    wf = _wf()
    wf["nodes"][1]["config"] = {"mode": "llm", "cases": ["yes", "no"]}
    res = _run(eng.run_workflow(wf, {"q": "whatever"}, model_call=fake_model))
    status = {e["node"]: e["status"] for e in res["log"]}
    assert status["on"] == "ok" and status["oy"] == "skipped"
    assert res["outputs"] == {"no_out": "whatever"}
