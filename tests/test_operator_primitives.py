import asyncio
import src.operator.primitives as p
from src.operator.actions import Action


def test_real_perceive_merges_windows_and_elements():
    async def run():
        return await p.real_perceive(
            get_root=lambda w: "ROOT",
            list_elements=lambda root: [{"name": "Save", "control_type": "Button"}],
            list_windows=lambda: [{"id": 1, "title": "Notepad"}])
    got = asyncio.run(run())
    assert got["windows"][0]["title"] == "Notepad"
    assert got["elements"][0]["name"] == "Save"


def test_real_perceive_tolerates_uia_failure():
    def boom(root): raise RuntimeError("no uia")
    async def run():
        return await p.real_perceive(get_root=lambda w: "ROOT", list_elements=boom,
                                     list_windows=lambda: [])
    got = asyncio.run(run())
    assert got["elements"] == []  # degrades, doesn't crash


def test_real_execute_dispatches_to_handler():
    seen = {}
    async def fake_handler(content, ctx):
        seen["content"] = content
        return {"output": "clicked", "exit_code": 0}
    act = Action(kind="act", tool="click_element", args={"name": "Save"})
    r = asyncio.run(p.real_execute(act, {"owner": "u"}, handlers={"click_element": fake_handler}))
    assert r["output"] == "clicked"
    import json
    assert json.loads(seen["content"]) == {"name": "Save"}


def test_real_execute_unknown_tool():
    act = Action(kind="act", tool="nope", args={})
    r = asyncio.run(p.real_execute(act, {}, handlers={}))
    assert r["exit_code"] == 1


def test_build_decide_prompt_includes_goal_and_elements():
    msgs = p.build_decide_prompt("do X", [], {"windows": [], "elements": [{"name": "Save"}]})
    blob = " ".join(m["content"] for m in msgs)
    assert "do X" in blob and "Save" in blob and "JSON" in blob


def test_real_decide_parses_model_reply():
    async def call_model(messages):
        return '{"kind":"act","tool":"click_element","args":{"name":"Save"}}'
    act = asyncio.run(p.real_decide("g", [], {"windows": [], "elements": []}, call_model=call_model))
    assert act.kind == "act" and act.tool == "click_element"
