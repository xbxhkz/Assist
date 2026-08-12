"""The `run_workflow` builtin tool: list saved workflows or run one by id via the
shipped engine. Admin-only (registered in NON_ADMIN_BLOCKED_TOOLS) — the engine's
inner-tool dispatch is ungated, so only admins may run a workflow. A ctx
`_in_workflow` flag blocks workflow-invokes-workflow nesting."""
import json

from src import workflow_runs
from src.workflows import store
from src.workflows.engine import run_workflow
from src.workflows.model import WorkflowError


def _input_names(wf):
    return [nm for n in (wf.get("nodes") or [])
            if n.get("type") == "input"
            for nm in [(n.get("config") or {}).get("name")] if nm]


def _parse_args(content):
    try:
        args = json.loads(content) if content and content.strip() else {}
    except (ValueError, TypeError):
        return {}
    return args if isinstance(args, dict) else {}


async def run_workflow_tool(content, ctx):
    ctx = ctx or {}
    if ctx.get("_in_workflow"):
        return {"error": "run_workflow cannot be called from within a workflow (no nesting)"}
    args = _parse_args(content)
    wid = (args.get("id") or "").strip()

    if not wid or args.get("action") == "list":
        items = []
        for w in store.list_workflows():
            try:                                   # a hand-edited path-unsafe id must not crash the listing
                wf = store.get_workflow(w.get("id")) or {}
            except ValueError:
                wf = {}
            items.append({"id": w.get("id"), "name": w.get("name"), "inputs": _input_names(wf)})
        return {"output": json.dumps({"workflows": items}, indent=2)}

    try:
        wf = store.get_workflow(wid)
    except ValueError:
        return {"error": f"invalid workflow id: {wid}"}
    if not wf:
        return {"error": f"workflow '{wid}' not found"}
    inputs = args.get("inputs")
    inputs = inputs if isinstance(inputs, dict) else {}
    child = dict(ctx)
    child["_in_workflow"] = True
    child["owner"] = ctx.get("owner")
    run_id = workflow_runs.start(wid, wf.get("name") or wid, ctx.get("owner"), "agent_tool")
    try:
        result = await run_workflow(wf, inputs, child)
    except WorkflowError as e:
        return {"error": "; ".join(e.errors)}
    finally:
        workflow_runs.finish(run_id)
    outputs = result.get("outputs") or {}
    log = result.get("log") or []
    oks = sum(1 for e in log if e.get("status") == "ok")
    errs = sum(1 for e in log if e.get("status") == "error")
    skips = sum(1 for e in log if e.get("status") == "skipped")
    return {"output": f"Outputs: {json.dumps(outputs)} · {oks} ok, {errs} error, {skips} skipped"}


class RunWorkflowTool:
    async def execute(self, content, ctx):
        return await run_workflow_tool(content, ctx)
