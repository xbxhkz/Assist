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
import os

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


def _fake_gallery_saver(image_id="gallery-img-1", filename="abc123def456.png"):
    calls = []

    def saver(image_bytes, owner):
        calls.append((image_bytes, owner))
        return {"id": image_id, "filename": filename}

    saver.calls = calls
    return saver


def test_successful_removal_returns_short_url_and_saves_to_gallery(tmp_path):
    """On the happy path image_url must be the SHORT served URL, never an
    inline data: URI. The base64 form is multi-KB-to-multi-MB and gets copied
    into the LLM's own context (tool_execution.format_tool_result JSON-dumps
    keys it doesn't special-case) AND into persisted session history
    (agent_loop's tool_event), which is replayed on every future session load
    and never shrinks. generate_image avoids this by returning a URL."""
    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})
    gallery_saver = _fake_gallery_saver(image_id="gallery-img-1", filename="abc123def456.png")

    result = asyncio.run(remove_background_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        remover=_fake_remover(output=b"removed-bg-bytes"),
        gallery_saver=gallery_saver,
    ))

    assert result["image_url"] == "/api/generated-image/abc123def456.png"
    assert "base64" not in result["image_url"]
    assert len(result["image_url"]) < 200
    assert result["gallery_image_id"] == "gallery-img-1"
    assert gallery_saver.calls == [(b"removed-bg-bytes", "alice")]


def test_result_carries_no_base64_payload_anywhere_on_the_happy_path(tmp_path):
    """Not just image_url: NO field may smuggle the image bytes back into the
    LLM context / persisted history."""
    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})

    big = b"\x89PNG" + b"x" * 100_000
    result = asyncio.run(remove_background_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        remover=_fake_remover(output=big),
        gallery_saver=_fake_gallery_saver(),
    ))

    assert all(len(str(v)) < 500 for v in result.values()), result.keys()


def test_real_formatter_output_carries_no_base64(tmp_path):
    """The actual consequence, checked against the REAL formatter: whatever
    remove_background returns is run through tool_execution.format_tool_result
    and fed back into the LLM's own context every subsequent turn. image_url is
    not in _FORMATTER_HANDLED_KEYS, so a data: URI would be JSON-dumped in
    whole (up to the 8000-char cap) as meaningless base64."""
    from src.tool_execution import format_tool_result

    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})

    result = asyncio.run(remove_background_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        remover=_fake_remover(output=b"\x89PNG" + b"y" * 200_000),
        gallery_saver=_fake_gallery_saver(),
    ))

    text = format_tool_result("remove_background: up-1", result)
    assert "base64" not in text
    assert "/api/generated-image/abc123def456.png" in text
    assert len(text) < 1000, f"tool result text is {len(text)} chars"


def test_default_gallery_saver_returns_id_and_filename(tmp_path, monkeypatch):
    """The served-URL path depends on the REAL saver handing back the filename
    it wrote, not just the row id -- and on that filename living in the exact
    directory app.py's /api/generated-image/{filename} serves from."""
    import src.agent_tools.image_tools as image_tools

    class _FakeDb:
        def __init__(self):
            self.added = []

        def add(self, row):
            self.added.append(row)

        def commit(self):
            pass

        def close(self):
            pass

    db = _FakeDb()
    monkeypatch.setattr("core.database.SessionLocal", lambda: db)
    monkeypatch.setattr("src.constants.GENERATED_IMAGES_DIR", str(tmp_path))

    saved = image_tools._default_gallery_saver(b"png-bytes", "alice")

    assert set(saved) == {"id", "filename"}
    assert saved["filename"].endswith(".png")
    # The bytes really landed where the serving route will look for them.
    assert (tmp_path / saved["filename"]).read_bytes() == b"png-bytes"
    assert db.added and db.added[0].filename == saved["filename"]
    assert db.added[0].owner == "alice"


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


def test_gallery_save_failure_falls_back_to_inline_data_uri(tmp_path):
    # Saving to Gallery is a best-effort convenience, not the primary
    # deliverable -- if it fails there is no served URL to hand back, so the
    # tool must fall back to the inline data: URI rather than losing the image
    # the model already produced.
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

    assert result["image_url"].startswith("data:image/png;base64,")
    decoded = base64.b64decode(result["image_url"].split(",", 1)[1])
    assert decoded == b"removed-bg-bytes"
    assert "gallery_image_id" not in result


def test_saver_without_a_filename_falls_back_to_inline_data_uri(tmp_path):
    """A saver that reports an id but no filename (e.g. a caller-injected one)
    leaves nothing for /api/generated-image/ to serve -- keep the gallery id,
    but still return a viewable image."""
    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})

    result = asyncio.run(remove_background_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        remover=_fake_remover(output=b"removed-bg-bytes"),
        gallery_saver=lambda image_bytes, owner: {"id": "gallery-img-9"},
    ))

    assert result["gallery_image_id"] == "gallery-img-9"
    assert result["image_url"].startswith("data:image/png;base64,")


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


# ── image_path (filesystem path or bare filename, mirrors face_swap's
# source_face_path/target_image_path) ──

def test_image_path_full_path_resolves(tmp_path, monkeypatch):
    import src.agent_tools.image_tools as image_tools

    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path) if p == "~" else os.path.expanduser(p))
    monkeypatch.setattr(image_tools, "get_setting", lambda key, default=None: [], raising=False)

    source_file = tmp_path / "photo.jpg"
    source_file.write_bytes(b"source-jpeg-bytes")

    captured = {}

    def capturing_remover(image_bytes, **kwargs):
        captured["image"] = image_bytes
        return b"removed-bg-bytes"

    content = json.dumps({"image_path": str(source_file)})
    result = asyncio.run(remove_background_tool(
        content, {"owner": "alice"},
        remover=capturing_remover,
        gallery_saver=_fake_gallery_saver(),
    ))

    assert "error" not in result
    assert captured["image"] == b"source-jpeg-bytes"


def test_image_path_bare_filename_resolves(tmp_path, monkeypatch):
    import src.agent_tools.image_tools as image_tools

    home_dir = tmp_path / "home"
    home_dir.mkdir()
    source_file = home_dir / "photo.png"
    source_file.write_bytes(b"source-bytes")

    monkeypatch.setattr(os.path, "expanduser", lambda p: str(home_dir) if p == "~" else os.path.expanduser(p))
    monkeypatch.setattr(image_tools, "get_setting", lambda key, default=None: [], raising=False)

    content = json.dumps({"image_path": "photo.png"})
    result = asyncio.run(remove_background_tool(
        content, {"owner": "alice"},
        remover=_fake_remover(output=b"removed-bg-bytes"),
        gallery_saver=_fake_gallery_saver(),
    ))

    assert "error" not in result


def test_giving_both_attachment_id_and_image_path_is_error(tmp_path):
    content = json.dumps({"attachment_id": "up-1", "image_path": str(tmp_path / "photo.png")})
    result = asyncio.run(remove_background_tool(content, {"owner": "alice"}))
    assert "error" in result
    assert "attachment_id" in result["error"] and "image_path" in result["error"]
