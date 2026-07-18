"""Resolve a triggered workflow run's inputs from the trigger's fixed inputs and
the firing context (webhook body / event payload). Pure — no scheduler, engine, or
DB. Precedence: input-node default (applied by the engine) < fixed inputs < context.

Context rule (one rule for every source): use context["inputs"] if it is a dict,
otherwise the context dict itself; overlay only keys that name an actual input node.
This covers a webhook body, a webhook {"inputs": {...}} envelope, and the event
{"message": "..."} payload alike."""


def _input_names(workflow):
    names = set()
    for n in (workflow.get("nodes") or []):
        if n.get("type") == "input":
            nm = (n.get("config") or {}).get("name")
            if nm:
                names.add(nm)
    return names


def resolve_trigger_inputs(workflow, fixed_inputs=None, context=None):
    """Return {name: value} for the workflow's input nodes. Unknown names are
    ignored; any input name omitted here is defaulted by the engine's run_input."""
    names = _input_names(workflow)
    out = {}
    for k, v in (fixed_inputs or {}).items():
        if k in names:
            out[k] = v
    ctx = context or {}
    inner = ctx.get("inputs") if isinstance(ctx, dict) else None
    candidate = inner if isinstance(inner, dict) else ctx
    if isinstance(candidate, dict):
        for k, v in candidate.items():
            if k in names:
                out[k] = v
    return out
