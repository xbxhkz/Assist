import asyncio
import pytest
import src.workflows.nodes as nd


def _run(coro):
    return asyncio.run(coro)


def test_fill_substitutes_and_blanks_missing():
    assert nd._fill("Hi {name} ({name})", {"name": "Ada"}) == "Hi Ada (Ada)"
    assert nd._fill("Hi {missing}", {}) == "Hi "
    assert nd._fill("plain", {}) == "plain"


def test_run_input_uses_run_value_then_default():
    assert _run(nd.run_input({"name": "q"}, {"q": "hello"})) == {"value": "hello"}
    assert _run(nd.run_input({"name": "q", "default": "d"}, {})) == {"value": "d"}
    assert _run(nd.run_input({"name": "q"}, {})) == {"value": ""}


def test_run_template_fills():
    assert _run(nd.run_template({"template": "Q: {q}"}, {"q": "why"})) == {"text": "Q: why"}


def test_run_llm_calls_injected_model():
    seen = {}
    async def fake_model(prompt, model=None, system=None):
        seen.update(prompt=prompt, model=model, system=system)
        return "ANSWER"
    out = _run(nd.run_llm({"prompt": "sum {doc}", "model": "m1", "system": "sys"},
                          {"doc": "text"}, model_call=fake_model))
    assert out == {"text": "ANSWER"}
    assert seen["prompt"] == "sum text" and seen["model"] == "m1" and seen["system"] == "sys"


def test_run_tool_calls_injected_dispatch():
    seen = {}
    async def fake_dispatch(tool, args, ctx):
        seen.update(tool=tool, args=args, ctx=ctx)
        return "RESULT"
    out = _run(nd.run_tool({"tool": "find_files", "args": "{path}"},
                           {"path": "/tmp"}, {"owner": "u"}, tool_dispatch=fake_dispatch))
    assert out == {"result": "RESULT"}
    assert seen["tool"] == "find_files" and seen["args"] == "/tmp" and seen["ctx"] == {"owner": "u"}


def test_run_output_passthrough():
    assert _run(nd.run_output({"name": "answer"}, {"value": "v"})) == {}


def test_default_tool_dispatch_unknown_tool_raises(monkeypatch):
    monkeypatch.setattr(nd, "_tool_handlers", lambda: {})
    with pytest.raises(RuntimeError):
        _run(nd.default_tool_dispatch("nope", "", {}))


def test_default_tool_dispatch_returns_output_and_raises_on_error(monkeypatch):
    async def ok(content, ctx):
        return {"output": "OK", "exit_code": 0}
    async def bad(content, ctx):
        return {"error": "boom", "exit_code": 1}
    monkeypatch.setattr(nd, "_tool_handlers", lambda: {"ok": ok, "bad": bad})
    assert _run(nd.default_tool_dispatch("ok", "args", {})) == "OK"
    with pytest.raises(RuntimeError):
        _run(nd.default_tool_dispatch("bad", "args", {}))
