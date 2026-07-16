import asyncio
import src.agent_tools.desktop_tools as dt


def _run(coro):
    return asyncio.run(coro)


def _patch_capture(monkeypatch, dets=None, vision=False, desc_text="a desk"):
    monkeypatch.setattr(dt, "capture_frame_jpeg", lambda: b"\xff\xd8jpeg")
    monkeypatch.setattr(dt, "detect", lambda jpeg: dets if dets is not None else [])
    monkeypatch.setattr(dt, "annotate", lambda jpeg, d: b"\xff\xd8annot")
    monkeypatch.setattr(dt, "summarize", lambda d: "1 person (92%; center)" if d else "No recognizable objects detected.")
    monkeypatch.setattr(dt, "_vision_ready", lambda: vision)
    monkeypatch.setattr(dt, "analyze_image_with_vl_result", lambda p, o: {"text": desc_text})


def test_refuses_when_camera_access_off(monkeypatch):
    monkeypatch.setattr(dt, "get_setting", lambda k, d=None: False)
    r = _run(dt.WebcamLookTool().execute("{}", {"owner": "u"}))
    assert r["exit_code"] == 1 and "Camera access" in r["error"]


def test_returns_objects_and_image(monkeypatch):
    monkeypatch.setattr(dt, "get_setting", lambda k, d=None: True if k == "camera_access_enabled" else d)
    _patch_capture(monkeypatch, dets=[{"label": "person"}])
    r = _run(dt.WebcamLookTool().execute("{}", {"owner": "u"}))
    assert r["exit_code"] == 0
    assert "person" in r["output"]
    assert r["image_url"].startswith("data:image/jpeg;base64,")


def test_describe_appends_vision_text(monkeypatch):
    monkeypatch.setattr(dt, "get_setting", lambda k, d=None: True if k == "camera_access_enabled" else d)
    _patch_capture(monkeypatch, dets=[{"label": "person"}], vision=True, desc_text="a person at a desk")
    r = _run(dt.WebcamLookTool().execute('{"describe": true}', {"owner": "u"}))
    assert "a person at a desk" in r["output"]


def test_describe_without_vision_notes_it(monkeypatch):
    monkeypatch.setattr(dt, "get_setting", lambda k, d=None: True if k == "camera_access_enabled" else d)
    _patch_capture(monkeypatch, dets=[], vision=False)
    r = _run(dt.WebcamLookTool().execute('{"describe": true}', {"owner": "u"}))
    assert "no vision model" in r["output"].lower()


def test_camera_error_is_reported(monkeypatch):
    monkeypatch.setattr(dt, "get_setting", lambda k, d=None: True if k == "camera_access_enabled" else d)
    def boom():
        raise RuntimeError("no camera 0")
    monkeypatch.setattr(dt, "capture_frame_jpeg", boom)
    r = _run(dt.WebcamLookTool().execute("{}", {"owner": "u"}))
    assert r["exit_code"] == 1 and "webcam_look" in r["error"]
