"""The `remove_background` builtin tool: strip the background from an
already-uploaded chat image attachment, returning a transparent PNG inline
in the chat response via the established image_url convention. Runs the
bundled U2Net ONNX model (src/bg_removal.py) -- no rembg/transformers
dependency. This is the first builtin tool to call
upload_handler.resolve_upload() directly; no existing accessor for the
app's UploadHandler singleton is reachable from a Tool's ctx, so this
constructs its own throwaway instance, mirroring
routes/document_helpers.py's existing precedent for the same reason (the
read path needs no cross-request state). NEVER raises into the agent --
every failure returns {"error": ...}, matching diagnose_equipment's
established pattern. See
docs/superpowers/specs/2026-08-12-image-editing-background-removal-design.md.
"""
import base64
import json


def _default_gallery_saver(image_bytes, owner):
    """Persist a new Gallery image, mirroring POST /api/gallery/upload's own
    GalleryImage field set exactly (routes/gallery/gallery_routes.py:230-248),
    minus the EXIF-derived fields that don't apply to a synthetically
    generated PNG. Returns the new image's id."""
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
            prompt="Background removed",
            model="remove_background",
            owner=owner,
            file_hash=hashlib.sha256(image_bytes).hexdigest(),
            file_size=len(image_bytes),
        ))
        db.commit()
        return img_id
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

    try:
        info = upload_resolver(attachment_id, owner=owner)
    except Exception as e:
        return {"error": f"remove_background: could not resolve attachment: {e}"}

    if not info or not info.get("path"):
        return {"error": f"remove_background: attachment '{attachment_id}' not found"}

    try:
        with open(info["path"], "rb") as f:
            image_bytes = f.read()
    except OSError as e:
        return {"error": f"remove_background: could not read attachment: {e}"}

    if remover is None:
        from src.bg_removal import remove_background as remover

    try:
        result_bytes = remover(image_bytes)
    except Exception as e:
        return {"error": f"remove_background: model failed: {e}"}

    image_url = "data:image/png;base64," + base64.b64encode(result_bytes).decode("ascii")
    result = {"output": "Background removed.", "image_url": image_url}

    # Best-effort: saving to Gallery makes the result findable later, but
    # isn't the primary deliverable -- a save failure must not lose the
    # image the model already successfully produced.
    saver = gallery_saver or _default_gallery_saver
    try:
        result["gallery_image_id"] = saver(result_bytes, owner)
    except Exception:
        pass

    return result


class RemoveBackgroundTool:
    async def execute(self, content, ctx):
        return await remove_background_tool(content, ctx)
