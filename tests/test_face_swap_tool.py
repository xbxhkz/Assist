"""face_swap_tool resolves TWO chat attachments (source face, target
image), runs them through src.face_swap, and returns a short served URL
via the established image_url convention. Never raises into the agent
loop, matching remove_background_tool/edit_image_prompt_tool's established
pattern. See
docs/superpowers/specs/2026-08-13-image-editing-face-swap-design.md.
"""
import asyncio
import json

from src.agent_tools.image_tools import FaceSwapTool, face_swap_tool


def _fake_upload_resolver(paths_by_id):
    def resolver(upload_id, owner=None):
        path = paths_by_id.get(upload_id)
        if path is None:
            return None
        return {"id": upload_id, "path": path, "name": "fake.png", "mime": "image/png"}
    return resolver


def _fake_swapper(output=b"fake-swapped-png"):
    def swapper(source_face_bytes, target_image_bytes):
        return output
    return swapper


def _fake_gallery_saver(image_id="gallery-img-1", filename="abc123def456.png"):
    calls = []

    def saver(image_bytes, owner):
        calls.append((image_bytes, owner))
        return {"id": image_id, "filename": filename}

    saver.calls = calls
    return saver


def test_missing_source_face_id_returns_error():
    content = json.dumps({"target_image_id": "up-2"})
    result = asyncio.run(face_swap_tool(content, {"owner": "alice"}))
    assert "error" in result


def test_missing_target_image_id_returns_error():
    content = json.dumps({"source_face_id": "up-1"})
    result = asyncio.run(face_swap_tool(content, {"owner": "alice"}))
    assert "error" in result


def test_invalid_json_returns_error():
    result = asyncio.run(face_swap_tool("not json", {"owner": "alice"}))
    assert "error" in result


def test_unresolvable_source_returns_error():
    content = json.dumps({"source_face_id": "missing", "target_image_id": "up-2"})
    result = asyncio.run(face_swap_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver({"up-2": "/tmp/target.png"}),
        swapper=_fake_swapper(),
    ))
    assert "error" in result


def test_unresolvable_target_returns_error():
    content = json.dumps({"source_face_id": "up-1", "target_image_id": "missing"})
    result = asyncio.run(face_swap_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver({"up-1": "/tmp/source.png"}),
        swapper=_fake_swapper(),
    ))
    assert "error" in result


def test_successful_swap_returns_short_url_and_saves_to_gallery(tmp_path):
    source_file = tmp_path / "source.png"
    source_file.write_bytes(b"source-bytes")
    target_file = tmp_path / "target.png"
    target_file.write_bytes(b"target-bytes")
    content = json.dumps({"source_face_id": "up-1", "target_image_id": "up-2"})
    gallery_saver = _fake_gallery_saver(image_id="gallery-img-1", filename="abc123def456.png")

    captured = {}

    def capturing_swapper(source_face_bytes, target_image_bytes):
        captured["source"] = source_face_bytes
        captured["target"] = target_image_bytes
        return b"swapped-bytes"

    result = asyncio.run(face_swap_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver({"up-1": str(source_file), "up-2": str(target_file)}),
        swapper=capturing_swapper,
        gallery_saver=gallery_saver,
    ))

    # Pin that source_bytes reaches the swapper's first argument and
    # target_bytes reaches its second -- a transposition here would be
    # silent and ship a wrong-and-more-harmful image in production.
    assert captured["source"] == b"source-bytes"
    assert captured["target"] == b"target-bytes"

    assert result["image_url"] == "/api/generated-image/abc123def456.png"
    assert result["gallery_image_id"] == "gallery-img-1"
    assert gallery_saver.calls == [(b"swapped-bytes", "alice")]


def test_gallery_saver_receives_face_swap_prompt_and_model_via_default_saver(tmp_path, monkeypatch):
    """The real (non-injected) saver path must label the Gallery row with
    this tool's own prompt/model ('Face swapped'/'face_swap') -- not
    _default_gallery_saver's hardcoded 'Background removed'/'remove_background'
    defaults, which are remove_background_tool's own values, not face_swap's."""
    import src.agent_tools.image_tools as image_tools

    captured = {}

    def fake_default_saver(image_bytes, owner, *, prompt="Background removed", model="remove_background"):
        captured["prompt"] = prompt
        captured["model"] = model
        return {"id": "gid", "filename": "f.png"}

    monkeypatch.setattr(image_tools, "_default_gallery_saver", fake_default_saver)

    source_file = tmp_path / "source.png"
    source_file.write_bytes(b"source-bytes")
    target_file = tmp_path / "target.png"
    target_file.write_bytes(b"target-bytes")
    content = json.dumps({"source_face_id": "up-1", "target_image_id": "up-2"})

    asyncio.run(face_swap_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver({"up-1": str(source_file), "up-2": str(target_file)}),
        swapper=_fake_swapper(output=b"swapped-bytes"),
    ))

    assert captured["prompt"] == "Face swapped"
    assert captured["model"] == "face_swap"


def test_license_not_accepted_returns_clear_error(tmp_path, monkeypatch):
    import src.agent_tools.image_tools as image_tools
    from src.face_swap import LicenseNotAcceptedError

    def failing_swapper(source_face_bytes, target_image_bytes):
        raise LicenseNotAcceptedError("Face-swap requires accepting InsightFace's model license first")

    source_file = tmp_path / "source.png"
    source_file.write_bytes(b"source-bytes")
    target_file = tmp_path / "target.png"
    target_file.write_bytes(b"target-bytes")
    content = json.dumps({"source_face_id": "up-1", "target_image_id": "up-2"})

    result = asyncio.run(face_swap_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver({"up-1": str(source_file), "up-2": str(target_file)}),
        swapper=failing_swapper,
        gallery_saver=_fake_gallery_saver(),
    ))

    assert "error" in result
    assert "license" in result["error"].lower()


def test_no_face_detected_returns_clear_error(tmp_path):
    from src.face_swap import NoFaceDetectedError

    def failing_swapper(source_face_bytes, target_image_bytes):
        raise NoFaceDetectedError("No face detected in the source image")

    source_file = tmp_path / "source.png"
    source_file.write_bytes(b"source-bytes")
    target_file = tmp_path / "target.png"
    target_file.write_bytes(b"target-bytes")
    content = json.dumps({"source_face_id": "up-1", "target_image_id": "up-2"})

    result = asyncio.run(face_swap_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver({"up-1": str(source_file), "up-2": str(target_file)}),
        swapper=failing_swapper,
        gallery_saver=_fake_gallery_saver(),
    ))

    assert "error" in result
    assert "face" in result["error"].lower()


def test_gallery_save_failure_falls_back_to_inline_data_uri(tmp_path):
    import base64

    source_file = tmp_path / "source.png"
    source_file.write_bytes(b"source-bytes")
    target_file = tmp_path / "target.png"
    target_file.write_bytes(b"target-bytes")
    content = json.dumps({"source_face_id": "up-1", "target_image_id": "up-2"})

    def failing_saver(image_bytes, owner):
        raise RuntimeError("db unavailable")

    result = asyncio.run(face_swap_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver({"up-1": str(source_file), "up-2": str(target_file)}),
        swapper=_fake_swapper(output=b"swapped-bytes"),
        gallery_saver=failing_saver,
    ))

    assert result["image_url"].startswith("data:image/png;base64,")
    decoded = base64.b64decode(result["image_url"].split(",", 1)[1])
    assert decoded == b"swapped-bytes"
    assert "gallery_image_id" not in result


def test_tool_class_delegates_to_module_function():
    tool = FaceSwapTool()
    content = json.dumps({"target_image_id": "up-2"})
    result = asyncio.run(tool.execute(content, {"owner": "alice"}))
    assert "error" in result
