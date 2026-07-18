"""Per-type node executors + the default LLM/tool calls.

`model_call` and `tool_dispatch` are injected by the engine so tests never hit a
real endpoint or tool. Defaults reuse the app's existing infrastructure:
resolve_endpoint (chat/completions) and TOOL_HANDLERS."""
import re

_SLOT_RE = re.compile(r"\{(\w+)\}")


def _fill(template, inputs):
    """Replace each {slot} with str(inputs[slot]); missing slots become ''."""
    def sub(match):
        return str(inputs.get(match.group(1), ""))
    return _SLOT_RE.sub(sub, template or "")


async def run_input(config, run_inputs):
    name = config.get("name", "")
    if name in (run_inputs or {}):
        return {"value": str((run_inputs or {})[name])}
    return {"value": str(config.get("default", ""))}


async def run_template(config, inputs):
    return {"text": _fill(config.get("template", ""), inputs)}


async def run_llm(config, inputs, *, model_call):
    prompt = _fill(config.get("prompt", ""), inputs)
    text = await model_call(prompt, model=config.get("model"), system=config.get("system"))
    return {"text": str(text)}


async def run_tool(config, inputs, ctx, *, tool_dispatch):
    args = _fill(config.get("args", ""), inputs)
    result = await tool_dispatch(config.get("tool", ""), args, ctx)
    return {"result": str(result)}


async def run_output(config, inputs):
    # The engine records inputs["value"] under config["name"]; nothing to emit.
    return {}


# ── defaults (reuse existing app infrastructure) ──

def _tool_handlers():
    from src.agent_tools import TOOL_HANDLERS  # lazy: avoids import cycles
    return TOOL_HANDLERS


async def default_model_call(prompt, model=None, system=None, owner=None):
    import httpx
    from src.endpoint_resolver import resolve_endpoint
    url, resolved_model, headers = resolve_endpoint("default", owner=owner)
    if not url:
        raise RuntimeError("no default model endpoint configured")
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    body = {"model": model or resolved_model, "messages": messages, "stream": False}
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(url, json=body, headers=headers or {})
        r.raise_for_status()
        data = r.json()
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")


async def default_tool_dispatch(tool, args, ctx):
    handler = _tool_handlers().get(tool)
    if not handler:
        raise RuntimeError(f"unknown tool: {tool}")
    res = await handler(args, ctx or {})
    if isinstance(res, dict):
        if res.get("error"):
            raise RuntimeError(str(res["error"]))
        return str(res.get("output", ""))
    return str(res)


def _match_case(resp, cases):
    """Map an llm response to a case: exact (trimmed, case-insensitive) first, then
    the first case whose label appears as a substring; else 'else'."""
    r = (resp or "").strip().lower()
    for c in cases:
        if c.strip().lower() == r:
            return c
    for c in cases:
        if c.strip().lower() in r:
            return c
    return "else"


async def run_branch(config, inputs, *, model_call):
    """Route `value` to one case output. match: equal-label (else 'else'). llm: the
    model classifies value into a case. Returns {"active": chosen, "value": value}."""
    value = str((inputs or {}).get("value", ""))
    cases = [c for c in (config.get("cases") or []) if isinstance(c, str) and c.strip()]
    if config.get("mode", "match") == "llm":
        guidance = config.get("prompt") or ""
        prompt = ((guidance + "\n\n") if guidance else "") + (
            f"Input:\n{value}\n\nChoose exactly one of these labels: "
            f"{', '.join(cases)}.\nAnswer with only the label.")
        resp = str(await model_call(prompt, model=config.get("model"), system=config.get("system")))
        chosen = _match_case(resp, cases)
    else:
        chosen = next((c for c in cases if c.strip().lower() == value.strip().lower()), "else")
    return {"active": chosen, "value": value}
