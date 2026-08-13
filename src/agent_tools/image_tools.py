"""The builtin image-editing tools that operate on an already-uploaded chat
image attachment: `remove_background` (strips the background via the bundled
U2Net ONNX model, src/bg_removal.py -- no rembg/transformers dependency) and
`edit_image_prompt` (applies a natural-language edit via img2img on the
bundled sd-server, src/image_edit.py). Both share three helpers here --
_resolve_attachment_bytes (attachment -> bytes), _image_result (result
shaping) and _default_gallery_saver (best-effort Gallery persistence).

Both return their result via the established image_url convention -- as a
SHORT /api/generated-image/<file> URL (like generate_image), falling back to
an inline data: URI only when the Gallery save failed. This is the first
builtin tool module to call
upload_handler.resolve_upload() directly; no existing accessor for the
app's UploadHandler singleton is reachable from a Tool's ctx, so this
constructs its own throwaway instance, mirroring
routes/document_helpers.py's existing precedent for the same reason (the
read path needs no cross-request state). NEVER raises into the agent --
every failure returns {"error": ...}, matching diagnose_equipment's
established pattern. See
docs/superpowers/specs/2026-08-12-image-editing-background-removal-design.md
and docs/superpowers/specs/2026-08-12-image-editing-prompt-edit-design.md.
"""
import asyncio
import base64
import json
import logging

logger = logging.getLogger(__name__)


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


async def _resolve_attachment_bytes(tool_name, attachment_id, owner, upload_resolver):
    """Resolve a chat attachment to bytes. Returns (image_bytes, None) on
    success, (None, error_dict) on failure. Shared by every image tool that
    resolves a chat-uploaded attachment (mirrors remove_background_tool's
    original inline logic)."""
    try:
        info = upload_resolver(attachment_id, owner=owner)
    except Exception as e:
        return None, {"error": f"{tool_name}: could not resolve attachment: {e}"}

    if not info or not info.get("path"):
        return None, {"error": f"{tool_name}: attachment '{attachment_id}' not found"}

    try:
        # Off the event loop, matching desktop_tools.py's capture_screen/
        # webcam_look pattern -- reading the source file can take real time.
        image_bytes = await asyncio.to_thread(_read_bytes, info["path"])
    except OSError as e:
        return None, {"error": f"{tool_name}: could not read attachment: {e}"}

    return image_bytes, None


def _image_result(output_message, result_bytes, saved):
    """Shape a tool's successful result: image_url (short served URL on a
    successful Gallery save, data: URI fallback if the save failed) plus an
    optional gallery_image_id. Shared by every image tool that follows the
    best-effort-Gallery-save pattern (mirrors remove_background_tool's
    original inline logic)."""
    result = {"output": output_message}

    filename = None
    if isinstance(saved, dict):
        if saved.get("id"):
            result["gallery_image_id"] = saved["id"]
        _fn = saved.get("filename")
        # str-check, not truthiness: a non-str here would silently build a
        # nonsense URL (this repo's recurring truthy-non-str bug class).
        filename = _fn if isinstance(_fn, str) and _fn.strip() else None
    elif saved:
        # An injected saver that only returns an id: still usable for the
        # gallery link, just not for a served URL.
        result["gallery_image_id"] = saved

    if filename:
        # Short, stable URL served by app.py's GET /api/generated-image/
        # {filename} (per-row owner enforcement included) -- exactly what
        # generate_image returns. A data: URI here would be copied verbatim
        # into TWO places that keep it forever: tool_execution.format_tool_
        # result JSON-dumps unhandled keys back into the LLM's own context,
        # and agent_loop's tool_event persists the untruncated value into
        # session history that is replayed on every future session load.
        result["image_url"] = f"/api/generated-image/{filename}"
    else:
        # Gallery save failed (or produced no file): fall back to the inline
        # data: URI so the user still gets a viewable image, preserving the
        # best-effort-persistence semantics.
        result["image_url"] = "data:image/png;base64," + base64.b64encode(result_bytes).decode("ascii")

    return result


def _default_gallery_saver(image_bytes, owner, *, prompt="Background removed", model="remove_background"):
    """Persist a new Gallery image, mirroring POST /api/gallery/upload's own
    GalleryImage field set exactly (routes/gallery/gallery_routes.py:230-248),
    minus the EXIF-derived fields that don't apply to a synthetically
    generated PNG.

    Returns {"id": <gallery row id>, "filename": <name under
    GENERATED_IMAGES_DIR>}. The filename matters as much as the id: it is what
    app.py's GET /api/generated-image/{filename} serves (with per-row owner
    enforcement), which lets the tool hand back a SHORT url instead of an
    inline multi-MB data: URI. `prompt`/`model` default to
    remove_background's own values so its existing 2-positional-arg call
    (saver(result_bytes, owner)) is unaffected; edit_image_prompt passes its
    own values instead of duplicating this file-write + DB-insert logic."""
    import hashlib
    import uuid
    from pathlib import Path

    from core.database import GalleryImage, SessionLocal
    from src.constants import GENERATED_IMAGES_DIR

    db = SessionLocal()
    try:
        img_dir = Path(GENERATED_IMAGES_DIR)
        img_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex[:12]}.png"
        (img_dir / filename).write_bytes(image_bytes)

        img_id = str(uuid.uuid4())
        db.add(GalleryImage(
            id=img_id,
            filename=filename,
            prompt=prompt,
            model=model,
            owner=owner,
            file_hash=hashlib.sha256(image_bytes).hexdigest(),
            file_size=len(image_bytes),
        ))
        db.commit()
        return {"id": img_id, "filename": filename}
    finally:
        db.close()


async def remove_background_tool(content, ctx, *, remover=None, upload_resolver=None, gallery_saver=None):
    ctx = ctx or {}
    owner = ctx.get("owner")

    try:
        args = json.loads(content) if content and content.strip() else {}
        if not isinstance(args, dict):
            return {"error": "remove_background: arguments must be a JSON object"}
    except (ValueError, TypeError):
        return {"error": "remove_background: arguments must be valid JSON"}

    attachment_id = args.get("attachment_id")
    if not isinstance(attachment_id, str) or not attachment_id.strip():
        return {"error": "remove_background: an 'attachment_id' is required"}

    if upload_resolver is None:
        from src.constants import DATA_DIR, UPLOAD_DIR
        from src.upload_handler import UploadHandler
        upload_resolver = UploadHandler(DATA_DIR, UPLOAD_DIR).resolve_upload

    image_bytes, err = await _resolve_attachment_bytes("remove_background", attachment_id, owner, upload_resolver)
    if err:
        return err

    if remover is None:
        from src.bg_removal import remove_background as remover

    try:
        result_bytes = await asyncio.to_thread(remover, image_bytes)
    except Exception as e:
        return {"error": f"remove_background: model failed: {e}"}

    # Best-effort: saving to Gallery makes the result findable later, but
    # isn't the primary deliverable -- a save failure must not lose the
    # image the model already successfully produced.
    saver = gallery_saver or _default_gallery_saver
    try:
        saved = saver(result_bytes, owner)
    except Exception:
        logger.warning("remove_background: failed to save result to Gallery", exc_info=True)
        saved = None

    return _image_result("Background removed.", result_bytes, saved)


class RemoveBackgroundTool:
    async def execute(self, content, ctx):
        return await remove_background_tool(content, ctx)


async def edit_image_prompt_tool(content, ctx, *, editor=None, upload_resolver=None, gallery_saver=None):
    ctx = ctx or {}
    owner = ctx.get("owner")

    try:
        args = json.loads(content) if content and content.strip() else {}
        if not isinstance(args, dict):
            return {"error": "edit_image_prompt: arguments must be a JSON object"}
    except (ValueError, TypeError):
        return {"error": "edit_image_prompt: arguments must be valid JSON"}

    attachment_id = args.get("attachment_id")
    if not isinstance(attachment_id, str) or not attachment_id.strip():
        return {"error": "edit_image_prompt: an 'attachment_id' is required"}

    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "edit_image_prompt: a 'prompt' describing the edit is required"}

    if upload_resolver is None:
        from src.constants import DATA_DIR, UPLOAD_DIR
        from src.upload_handler import UploadHandler
        upload_resolver = UploadHandler(DATA_DIR, UPLOAD_DIR).resolve_upload

    image_bytes, err = await _resolve_attachment_bytes("edit_image_prompt", attachment_id, owner, upload_resolver)
    if err:
        return err

    if editor is None:
        from src.ai_interaction import _apply_image_autoserve, _resolve_model
        from src.image_edit import edit_image as editor

        model_spec, autoserve_err = await _apply_image_autoserve("", False, owner)
        if autoserve_err:
            return {"error": f"edit_image_prompt: {autoserve_err}"}
        if not model_spec:
            # Reached when _apply_image_autoserve reports a local model IS
            # serving but its advertised id could not be probed (it returns ""
            # in that case -- see src/ai_interaction.py). The genuinely-not-
            # configured case surfaces through the autoserve_err branch above.
            return {"error": "edit_image_prompt: a local image model is serving but did "
                              "not report a model id -- try restarting it from "
                              "Admin -> Image Generation"}
        try:
            url, model_id, headers = await asyncio.to_thread(_resolve_model, model_spec, owner=owner)
        except ValueError as e:
            return {"error": f"edit_image_prompt: no endpoint found for image model "
                              f"'{model_spec}': {e}"}
        base_url = url.replace("/chat/completions", "").replace("/v1/messages", "").rstrip("/")
    else:
        # Injected editor (tests): skip real auto-serve/resolve entirely.
        base_url, model_id, headers = "", "edit_image_prompt", {}

    try:
        # headers must be passed as a keyword here: src.image_edit.edit_image's
        # real signature makes it keyword-only (after `*`), and asyncio.to_thread
        # forwards **kwargs as well as *args -- a positional call would raise
        # TypeError against the real function. Test-injected editors declare
        # `headers` keyword-only too (see tests/test_edit_image_prompt_tool.py),
        # so a regression back to a positional call here fails the test suite
        # instead of only failing in production.
        result_bytes = await asyncio.to_thread(editor, image_bytes, prompt, base_url, headers=headers)
    except Exception as e:
        return {"error": f"edit_image_prompt: model failed: {e}"}

    def _saver(image_bytes, owner):
        return _default_gallery_saver(image_bytes, owner, prompt=prompt, model=model_id)

    saver = gallery_saver or _saver
    try:
        saved = saver(result_bytes, owner)
    except Exception:
        logger.warning("edit_image_prompt: failed to save result to Gallery", exc_info=True)
        saved = None

    return _image_result("Image edited.", result_bytes, saved)


class EditImagePromptTool:
    async def execute(self, content, ctx):
        return await edit_image_prompt_tool(content, ctx)
