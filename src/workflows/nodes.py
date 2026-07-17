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
