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


def test_real_execute_bounds_slow_capture_screen():
    # capture_screen drives the VLM, which can hang for minutes on a small GPU.
    # The operator must not block on it — a timeout degrades to a steer message.
    async def slow_handler(content, ctx):
        await asyncio.sleep(5)
        return {"output": "too late", "exit_code": 0}
    act = Action(kind="act", tool="capture_screen", args={})
    r = asyncio.run(p.real_execute(act, {}, handlers={"capture_screen": slow_handler},
                                   capture_timeout=0.05))
    assert r["exit_code"] == 1 and "timed out" in r["error"].lower()
    assert "element" in r["error"].lower()  # steers back to the UIA element list


def test_real_execute_does_not_bound_fast_tools():
    async def fast_handler(content, ctx):
        return {"output": "ok", "exit_code": 0}
    act = Action(kind="act", tool="click_element", args={})
    r = asyncio.run(p.real_execute(act, {}, handlers={"click_element": fast_handler},
                                   capture_timeout=0.05))
    assert r["output"] == "ok"


def test_build_decide_prompt_includes_goal_and_elements():
    msgs = p.build_decide_prompt("do X", [], {"windows": [], "elements": [{"name": "Save"}]})
    blob = " ".join(m["content"] for m in msgs)
    assert "do X" in blob and "Save" in blob and "JSON" in blob


def test_build_decide_prompt_steers_dialogs_to_element_list():
    # A dialog (e.g. Save As) must be driven via UIA controls, not a slow screenshot.
    blob = " ".join(m["content"] for m in
                    p.build_decide_prompt("g", [], {"windows": [], "elements": []}))
    assert "dialog" in blob.lower()


def test_real_decide_parses_model_reply():
    async def call_model(messages):
        return '{"kind":"act","tool":"click_element","args":{"name":"Save"}}'
    act = asyncio.run(p.real_decide("g", [], {"windows": [], "elements": []}, call_model=call_model))
    assert act.kind == "act" and act.tool == "click_element"
