"""remove_background_tool resolves a chat attachment (the first builtin
tool to call upload_handler.resolve_upload() directly -- no existing tool
does this today), runs it through src.bg_removal, and returns an inline
data: URI via the established image_url tool-result convention (the same
mechanism generate_image/webcam_look already use). Never raises into the
agent loop, matching diagnose_equipment's established pattern. See
docs/superpowers/specs/2026-08-12-image-editing-background-removal-design.md.
"""
import asyncio
import base64
import json

import pytest

from src.agent_tools.image_tools import RemoveBackgroundTool, remove_background_tool


def _fake_upload_resolver(found=True, path="/tmp/fake.png"):
    def resolver(upload_id, owner=None):
        if not found:
            return None
        return {"id": upload_id, "path": path, "name": "fake.png", "mime": "image/png"}
    return resolver


def _fake_remover(output=b"fake-png-bytes"):
    def remover(image_bytes, **kwargs):
        return output
    return remover


def test_missing_attachment_id_returns_error():
    result = asyncio.run(remove_background_tool("{}", {"owner": "alice"}))
    assert "error" in result


def test_invalid_json_returns_error():
    result = asyncio.run(remove_background_tool("not json", {"owner": "alice"}))
    assert "error" in result


def test_unresolvable_attachment_returns_error():
    content = json.dumps({"attachment_id": "missing-id"})
    result = asyncio.run(remove_background_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=False),
        remover=_fake_remover(),
    ))
    assert "error" in result


def _fake_gallery_saver(image_id="gallery-img-1"):
    calls = []

    def saver(image_bytes, owner):
        calls.append((image_bytes, owner))
        return image_id

    saver.calls = calls
    return saver


def test_successful_removal_returns_image_url_and_saves_to_gallery(tmp_path):
    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})
    gallery_saver = _fake_gallery_saver(image_id="gallery-img-1")

    result = asyncio.run(remove_background_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        remover=_fake_remover(output=b"removed-bg-bytes"),
        gallery_saver=gallery_saver,
    ))

    assert "image_url" in result
    assert result["image_url"].startswith("data:image/png;base64,")
    decoded = base64.b64decode(result["image_url"].split(",", 1)[1])
    assert decoded == b"removed-bg-bytes"
    assert result["gallery_image_id"] == "gallery-img-1"
    assert gallery_saver.calls == [(b"removed-bg-bytes", "alice")]


def test_model_failure_returns_error_not_raise(tmp_path):
    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})

    def failing_remover(image_bytes, **kwargs):
        raise RuntimeError("boom")

    result = asyncio.run(remove_background_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        remover=failing_remover,
        gallery_saver=_fake_gallery_saver(),
    ))

    assert "error" in result


def test_gallery_save_failure_still_returns_image_url(tmp_path):
    # Saving to Gallery is a best-effort convenience, not the primary
    # deliverable -- if it fails, the user should still get the image
    # inline rather than losing the whole result.
    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})

    def failing_saver(image_bytes, owner):
        raise RuntimeError("db unavailable")

    result = asyncio.run(remove_background_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        remover=_fake_remover(output=b"removed-bg-bytes"),
        gallery_saver=failing_saver,
    ))

    assert "image_url" in result
    assert "gallery_image_id" not in result


def test_tool_class_delegates_to_module_function():
    tool = RemoveBackgroundTool()
    result = asyncio.run(tool.execute("{}", {"owner": "alice"}))
    assert "error" in result


def test_gallery_save_failure_logs_a_warning(tmp_path, caplog):
    # The Gallery-save except block must not discard the failure reason
    # entirely -- it should be surfaced via the module logger (as a
    # non-fatal warning, since remove_background still succeeds and
    # returns image_url) rather than silently swallowed.
    import logging

    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})

    def failing_saver(image_bytes, owner):
        raise RuntimeError("db unavailable")

    with caplog.at_level(logging.WARNING, logger="src.agent_tools.image_tools"):
        result = asyncio.run(remove_background_tool(
            content, {"owner": "alice"},
            upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
            remover=_fake_remover(output=b"removed-bg-bytes"),
            gallery_saver=failing_saver,
        ))

    assert "image_url" in result
    assert "gallery_image_id" not in result
    assert any(r.levelno == logging.WARNING and "Gallery" in r.message for r in caplog.records)
