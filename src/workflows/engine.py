"""Workflow execution: validate -> topo sort -> run each node, flowing text
values along edges. Partial-on-failure: a node error is logged, its dependents
are skipped, and the run still returns {outputs, log}."""
import logging
import time

from src.workflows import nodes as N
from src.workflows.model import WorkflowError, topo_sort, validate

logger = logging.getLogger(__name__)

_LOG_MAX = 500  # truncate node output in the run log


async def _run_node(node, node_inputs, run_inputs, ctx, model_call, tool_dispatch):
    t = node.get("type")
    cfg = node.get("config") or {}
    if t == "input":
        return await N.run_input(cfg, run_inputs)
    if t == "template":
        return await N.run_template(cfg, node_inputs)
    if t == "llm":
        return await N.run_llm(cfg, node_inputs, model_call=model_call)
    if t == "tool":
        return await N.run_tool(cfg, node_inputs, ctx, tool_dispatch=tool_dispatch)
    if t == "output":
        return await N.run_output(cfg, node_inputs)
    raise RuntimeError(f"unknown node type: {t}")


async def run_workflow(wf, inputs=None, ctx=None, *, model_call=None, tool_dispatch=None):
    """Run `wf` with `inputs` -> {"outputs": {name: value}, "log": [entry]}.

    Raises WorkflowError if the graph is invalid (the route maps that to 400)."""
    errs = validate(wf)
    if errs:
        raise WorkflowError(errs)
    model_call = model_call or N.default_model_call
    tool_dispatch = tool_dispatch or N.default_tool_dispatch
    run_inputs = inputs or {}
    by_id = {n["id"]: n for n in wf.get("nodes") or []}
    edges = wf.get("edges") or []

    produced = {}      # node id -> {port: value}
    failed = set()     # nodes that errored or were skipped
    outputs = {}
    log = []

    for nid in topo_sort(wf):
        node = by_id[nid]
        incoming = [e for e in edges if e.get("to_node") == nid]
        upstream_bad = any(e.get("from_node") in failed for e in incoming)
        if upstream_bad:
            failed.add(nid)
            log.append({"node": nid, "type": node.get("type"), "status": "skipped",
                        "output": "", "error": None, "ms": 0})
            continue
        node_inputs = {e["to_port"]: produced.get(e["from_node"], {}).get(e["from_port"], "")
                       for e in incoming}
        started = time.monotonic()
        try:
            out = await _run_node(node, node_inputs, run_inputs, ctx, model_call, tool_dispatch)
            produced[nid] = out
            if node.get("type") == "output":
                outputs[(node.get("config") or {}).get("name", nid)] = node_inputs.get("value", "")
            shown = str(next(iter(out.values()), "")) if out else str(node_inputs.get("value", ""))
            log.append({"node": nid, "type": node.get("type"), "status": "ok",
                        "output": shown[:_LOG_MAX], "error": None,
                        "ms": int((time.monotonic() - started) * 1000)})
        except Exception as e:
            failed.add(nid)
            logger.info("workflow node %s failed: %s", nid, e)
            log.append({"node": nid, "type": node.get("type"), "status": "error",
                        "output": "", "error": str(e),
                        "ms": int((time.monotonic() - started) * 1000)})
    return {"outputs": outputs, "log": log}
