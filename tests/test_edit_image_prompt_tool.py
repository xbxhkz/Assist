"""edit_image_prompt_tool resolves a chat attachment, auto-serves the default
image model, runs it through src.image_edit, and returns an inline
data: URI or short served URL via the established image_url convention
(same as remove_background_tool). Never raises into the agent loop. See
docs/superpowers/specs/2026-08-12-image-editing-prompt-edit-design.md.
"""
import asyncio
import base64
import json

from src.agent_tools.image_tools import EditImagePromptTool, edit_image_prompt_tool


def _fake_upload_resolver(found=True, path="/tmp/fake.png"):
    def resolver(upload_id, owner=None):
        if not found:
            return None
        return {"id": upload_id, "path": path, "name": "fake.png", "mime": "image/png"}
    return resolver


def _fake_editor(output=b"fake-edited-png"):
    def editor(image_bytes, prompt, base_url, *, headers):
        return output
    return editor


def _fake_gallery_saver(image_id="gallery-img-1", filename="abc123def456.png"):
    calls = []

    def saver(image_bytes, owner):
        calls.append((image_bytes, owner))
        return {"id": image_id, "filename": filename}

    saver.calls = calls
    return saver


def test_missing_attachment_id_returns_error():
    content = json.dumps({"prompt": "add a hat"})
    result = asyncio.run(edit_image_prompt_tool(content, {"owner": "alice"}))
    assert "error" in result


def test_missing_prompt_returns_error():
    content = json.dumps({"attachment_id": "up-1"})
    result = asyncio.run(edit_image_prompt_tool(content, {"owner": "alice"}))
    assert "error" in result


def test_invalid_json_returns_error():
    result = asyncio.run(edit_image_prompt_tool("not json", {"owner": "alice"}))
    assert "error" in result


def test_unresolvable_attachment_returns_error():
    content = json.dumps({"attachment_id": "missing-id", "prompt": "add a hat"})
    result = asyncio.run(edit_image_prompt_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=False),
        editor=_fake_editor(),
    ))
    assert "error" in result


def test_successful_edit_returns_short_url_and_saves_to_gallery(tmp_path):
    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1", "prompt": "add a red hat"})
    gallery_saver = _fake_gallery_saver(image_id="gallery-img-1", filename="abc123def456.png")

    result = asyncio.run(edit_image_prompt_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        editor=_fake_editor(output=b"edited-bytes"),
        gallery_saver=gallery_saver,
    ))

    assert result["image_url"] == "/api/generated-image/abc123def456.png"
    assert "base64" not in result["image_url"]
    assert result["gallery_image_id"] == "gallery-img-1"
    assert gallery_saver.calls == [(b"edited-bytes", "alice")]


def test_model_failure_returns_error_not_raise(tmp_path):
    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1", "prompt": "add a red hat"})

    def failing_editor(image_bytes, prompt, base_url, *, headers):
        raise RuntimeError("boom")

    result = asyncio.run(edit_image_prompt_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        editor=failing_editor,
        gallery_saver=_fake_gallery_saver(),
    ))

    assert "error" in result


def test_gallery_save_failure_falls_back_to_inline_data_uri(tmp_path):
    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1", "prompt": "add a red hat"})

    def failing_saver(image_bytes, owner):
        raise RuntimeError("db unavailable")

    result = asyncio.run(edit_image_prompt_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        editor=_fake_editor(output=b"edited-bytes"),
        gallery_saver=failing_saver,
    ))

    assert result["image_url"].startswith("data:image/png;base64,")
    decoded = base64.b64decode(result["image_url"].split(",", 1)[1])
    assert decoded == b"edited-bytes"
    assert "gallery_image_id" not in result


def test_gallery_saver_receives_prompt_and_model_via_default_saver(tmp_path, monkeypatch):
    """The real (non-injected) saver path must write the user's actual edit
    prompt and the serving model id into the Gallery row -- not
    remove_background's hardcoded 'Background removed'/'remove_background'
    defaults, which _default_gallery_saver still uses for ITS OWN caller."""
    import src.agent_tools.image_tools as image_tools

    captured = {}

    def fake_default_saver(image_bytes, owner, *, prompt="Background removed", model="remove_background"):
        captured["prompt"] = prompt
        captured["model"] = model
        return {"id": "gid", "filename": "f.png"}

    monkeypatch.setattr(image_tools, "_default_gallery_saver", fake_default_saver)

    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1", "prompt": "add a red hat"})

    def fake_editor(image_bytes, prompt, base_url, *, headers):
        return b"edited-bytes"

    asyncio.run(edit_image_prompt_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        editor=fake_editor,
    ))

    assert captured["prompt"] == "add a red hat"


def test_tool_class_delegates_to_module_function():
    tool = EditImagePromptTool()
    content = json.dumps({"prompt": "add a hat"})
    result = asyncio.run(tool.execute(content, {"owner": "alice"}))
    assert "error" in result
