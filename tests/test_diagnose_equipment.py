import asyncio
import base64
import json

import src.agent_tools.industrial_tools as it


def _run(coro):
    return asyncio.run(coro)


def _img(tmp_path, name="p.png"):
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    p = tmp_path / name
    p.write_bytes(png)
    return str(p)


def _exec(content, ctx=None, vision_call=None):
    return _run(it.diagnose_equipment(content, ctx or {}, vision_call=vision_call))


def _capture_call():
    seen = {}
    def vc(image_path, owner=None, *, prompt=None):
        seen["image_path"] = image_path
        seen["owner"] = owner
        seen["prompt"] = prompt
        return {"text": "DIAGNOSIS", "model": "vl-model"}
    return vc, seen


def test_each_mode_selects_its_expert_prompt_and_returns_output(tmp_path):
    for mode in ("schematic", "fault_code", "vfd", "component"):
        vc, seen = _capture_call()
        out = _exec(json.dumps({"image": _img(tmp_path), "task": mode}), {"owner": "admin"}, vision_call=vc)
        assert out == {"output": "DIAGNOSIS"}
        # the EXACT mode fragment is present (distinguishes modes), plus shared safety + structure
        assert it.TASK_MODES[mode] in seen["prompt"]
        assert "not a substitute for a qualified person" in seen["prompt"]
        assert "never" in seen["prompt"].lower()          # safety constraint
        assert "likely_causes" in seen["prompt"]           # structured-output request


def test_safety_clause_in_every_mode_prompt(tmp_path):
    for mode in list(it.TASK_MODES) + ["auto"]:
        vc, seen = _capture_call()
        _exec(json.dumps({"image": _img(tmp_path), "task": mode}), {"owner": "admin"}, vision_call=vc)
        assert it.SAFETY_CLAUSE in seen["prompt"]


def test_context_is_folded_into_the_prompt(tmp_path):
    vc, seen = _capture_call()
    _exec(json.dumps({"image": _img(tmp_path), "task": "vfd", "context": "trips on start, overcurrent"}),
          {"owner": "admin"}, vision_call=vc)
    assert "trips on start, overcurrent" in seen["prompt"]


def test_default_task_is_auto(tmp_path):
    vc, seen = _capture_call()
    _exec(json.dumps({"image": _img(tmp_path)}), {"owner": "admin"}, vision_call=vc)
    assert it.TASK_MODES["auto"] in seen["prompt"]
    # an unknown task also falls back to auto
    vc2, seen2 = _capture_call()
    _exec(json.dumps({"image": _img(tmp_path), "task": "bogus"}), {"owner": "admin"}, vision_call=vc2)
    assert it.TASK_MODES["auto"] in seen2["prompt"]


def test_missing_image_arg_is_error(tmp_path):
    out = _exec(json.dumps({"task": "vfd"}), {"owner": "admin"}, vision_call=_capture_call()[0])
    assert "error" in out and "image" in out["error"].lower()


def test_nonexistent_or_non_image_path_is_error(tmp_path):
    out = _exec(json.dumps({"image": str(tmp_path / "nope.png")}), {"owner": "admin"}, vision_call=_capture_call()[0])
    assert "error" in out
    txt = tmp_path / "notes.txt"; txt.write_text("hi")
    out2 = _exec(json.dumps({"image": str(txt)}), {"owner": "admin"}, vision_call=_capture_call()[0])
    assert "error" in out2


def test_oversized_image_is_error(tmp_path, monkeypatch):
    big = tmp_path / "big.png"
    big.write_bytes(b"x" * 10)                 # small file, but shrink the cap so it trips
    monkeypatch.setattr(it, "MAX_IMAGE_BYTES", 5)
    out = _exec(json.dumps({"image": str(big)}), {"owner": "admin"}, vision_call=_capture_call()[0])
    assert "error" in out


def test_no_vision_model_available_is_error(tmp_path):
    def vc(image_path, owner=None, *, prompt=None):
        return {"text": "[No vision model configured — set one in Settings → Vision]", "model": ""}
    out = _exec(json.dumps({"image": _img(tmp_path)}), {"owner": "admin"}, vision_call=vc)
    assert "error" in out


def test_never_raises_on_vision_exception(tmp_path):
    def vc(image_path, owner=None, *, prompt=None):
        raise RuntimeError("boom")
    out = _exec(json.dumps({"image": _img(tmp_path)}), {"owner": "admin"}, vision_call=vc)
    assert "error" in out and "boom" in out["error"]


def test_bad_json_content_is_error(tmp_path):
    out = _exec("not json", {"owner": "admin"}, vision_call=_capture_call()[0])
    assert "error" in out
