"""detect_shapes_tool resolves a chat attachment or filesystem path/
filename, runs it through src.shape_detect, and returns a text summary +
annotated image via the established image_url convention. Never raises
into the agent loop, matching every other image tool's established
pattern. See docs/superpowers/specs/2026-08-15-shape-detection-design.md.
"""
import asyncio
import json

from src.agent_tools.image_tools import DetectShapesTool, detect_shapes_tool


def _fake_upload_resolver(found=True, path="/tmp/fake.png"):
    def resolver(upload_id, owner=None):
        if not found:
            return None
        return {"id": upload_id, "path": path, "name": "fake.png", "mime": "image/png"}
    return resolver


def _fake_gallery_saver(image_id="gallery-img-1", filename="abc123def456.png"):
    calls = []

    def saver(image_bytes, owner):
        calls.append((image_bytes, owner))
        return {"id": image_id, "filename": filename}

    saver.calls = calls
    return saver


_ONE_PERSON = [{"label": "person", "confidence": 0.92, "box": [1, 2, 3, 4], "position": "left"}]


def test_missing_attachment_id_and_image_path_returns_error():
    result = asyncio.run(detect_shapes_tool("{}", {"owner": "alice"}))
    assert "error" in result


def test_invalid_json_returns_error():
    result = asyncio.run(detect_shapes_tool("not json", {"owner": "alice"}))
    assert "error" in result


def test_unresolvable_attachment_returns_error():
    content = json.dumps({"attachment_id": "missing-id"})
    result = asyncio.run(detect_shapes_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=False),
        detector=lambda image_bytes: _ONE_PERSON,
    ))
    assert "error" in result


def test_successful_detection_returns_summary_and_short_url(tmp_path):
    real_file = tmp_path / "photo.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})
    gallery_saver = _fake_gallery_saver(image_id="gallery-img-1", filename="abc123def456.png")

    result = asyncio.run(detect_shapes_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        detector=lambda image_bytes: _ONE_PERSON,
        gallery_saver=gallery_saver,
    ))

    assert "error" not in result
    assert "person" in result["output"]
    assert result["image_url"] == "/api/generated-image/abc123def456.png"
    assert result["gallery_image_id"] == "gallery-img-1"
    assert gallery_saver.calls[0][1] == "alice"


def test_zero_detections_is_not_an_error(tmp_path):
    real_file = tmp_path / "photo.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})

    result = asyncio.run(detect_shapes_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        detector=lambda image_bytes: [],
        gallery_saver=_fake_gallery_saver(),
    ))

    assert "error" not in result
    assert "No recognizable objects detected" in result["output"]


def test_detector_failure_returns_error_not_raise(tmp_path):
    real_file = tmp_path / "photo.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})

    def failing_detector(image_bytes):
        raise RuntimeError("model download failed")

    result = asyncio.run(detect_shapes_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        detector=failing_detector,
        gallery_saver=_fake_gallery_saver(),
    ))

    assert "error" in result


def test_gallery_saver_receives_detect_shapes_prompt_and_model_via_default_saver(tmp_path, monkeypatch):
    import src.agent_tools.image_tools as image_tools

    captured = {}

    def fake_default_saver(image_bytes, owner, *, prompt="Background removed", model="remove_background"):
        captured["prompt"] = prompt
        captured["model"] = model
        return {"id": "gid", "filename": "f.png"}

    monkeypatch.setattr(image_tools, "_default_gallery_saver", fake_default_saver)

    real_file = tmp_path / "photo.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})

    asyncio.run(detect_shapes_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        detector=lambda image_bytes: _ONE_PERSON,
    ))

    assert captured["prompt"] == "Shapes detected"
    assert captured["model"] == "detect_shapes"


def test_gallery_save_failure_falls_back_to_inline_data_uri(tmp_path):
    real_file = tmp_path / "photo.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})

    def failing_saver(image_bytes, owner):
        raise RuntimeError("db unavailable")

    result = asyncio.run(detect_shapes_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        detector=lambda image_bytes: _ONE_PERSON,
        gallery_saver=failing_saver,
    ))

    assert result["image_url"].startswith("data:image/png;base64,")
    assert "gallery_image_id" not in result


def test_tool_class_delegates_to_module_function():
    tool = DetectShapesTool()
    result = asyncio.run(tool.execute("{}", {"owner": "alice"}))
    assert "error" in result
