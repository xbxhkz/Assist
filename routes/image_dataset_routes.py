"""Admin-gated Image Dataset prep API (Image AI Studio, sub-project 1)."""
from fastapi import APIRouter, Body, Depends, File, HTTPException, UploadFile

from core.middleware import require_admin
from core.database import SessionLocal, GalleryImage
from src.image_dataset_tools import working_set
from src.image_dataset_tools.caption import caption_image
from src.image_dataset_tools.validate import validate_image_set
from src.image_dataset_tools.store import ImageDatasetStore, get_image_dataset_store


def _gallery_image_path(filename):
    from routes.gallery.gallery_routes import _gallery_image_path as real_resolve
    return str(real_resolve(filename))


def setup_image_dataset_routes() -> APIRouter:
    router = APIRouter(prefix="/api/image-datasets",
                       dependencies=[Depends(require_admin)])

    @router.post("/upload")
    async def upload(files: list[UploadFile] = File(...)):
        wid = working_set.new_working_set()
        pairs = []
        for f in files:
            try:
                data = await f.read()
                pairs.append((f.filename or "image.png", data))
            except Exception:  # noqa: BLE001
                continue
        added = working_set.add_images(wid, pairs)
        return {"working_set_id": wid, "images": added}

    @router.post("/from-gallery")
    async def from_gallery(body: dict = Body(...)):
        ids = body.get("ids")
        owner = body.get("owner")
        wid = body.get("working_set_id") or working_set.new_working_set()
        if not isinstance(ids, list) or not ids:
            return {"working_set_id": wid, "images": []}
        db = SessionLocal()
        try:
            rows = db.query(GalleryImage).filter(GalleryImage.id.in_(ids)).all()
        except Exception:  # noqa: BLE001
            rows = []
        finally:
            db.close()
        pairs = []
        for row in rows:
            if owner is not None and getattr(row, "owner", None) != owner:
                continue  # owner-scoped: never pull another user's images
            try:
                path = _gallery_image_path(row.filename)
                with open(path, "rb") as f:
                    data = f.read()
                pairs.append((row.filename, data, getattr(row, "caption", "") or ""))
            except Exception:  # noqa: BLE001
                continue
        added = working_set.add_images(wid, pairs)
        return {"working_set_id": wid, "images": added}

    @router.get("/working/{working_set_id}/{image_id}")
    async def get_working_image(working_set_id: str, image_id: str):
        from fastapi.responses import FileResponse
        path = working_set.get_image_path(working_set_id, image_id)
        if not path:
            raise HTTPException(404, "image not found")
        return FileResponse(path)

    @router.post("/caption")
    async def caption(body: dict = Body(...)):
        wid = body.get("working_set_id")
        only_ids = body.get("only_ids")
        results = []
        for img in working_set.list_images(wid):
            if isinstance(only_ids, list) and img["id"] not in only_ids:
                results.append(img)
                continue
            path = working_set.get_image_path(wid, img["id"])
            if not path:
                results.append({**img, "error": "image not found"})
                continue
            cap, err = caption_image(path)
            if cap is not None:
                working_set.set_caption(wid, img["id"], cap)
                results.append({"id": img["id"], "caption": cap})
            else:
                results.append({"id": img["id"], "caption": img.get("caption", ""), "error": err})
        return {"images": results}

    @router.post("/validate")
    async def validate(body: dict = Body(...)):
        wid = body.get("working_set_id")
        captions = body.get("captions") if isinstance(body.get("captions"), dict) else {}
        trigger = body.get("trigger_word") or ""
        entries = []
        for img in working_set.list_images(wid):
            path = working_set.get_image_path(wid, img["id"])
            cap = captions.get(img["id"], img.get("caption", ""))
            full_caption = f"{trigger} {cap}".strip() if trigger else cap
            entries.append({"id": img["id"], "path": path, "caption": full_caption})
        return validate_image_set(entries)

    @router.post("")
    async def save(body: dict = Body(...)):
        wid = body.get("working_set_id")
        name = body.get("name")
        trigger = body.get("trigger_word") or ""
        captions = body.get("captions") if isinstance(body.get("captions"), dict) else {}
        entries = []
        for img in working_set.list_images(wid):
            path = working_set.get_image_path(wid, img["id"])
            if not path:
                continue
            cap = captions.get(img["id"], img.get("caption", ""))
            full_caption = f"{trigger} {cap}".strip() if trigger else cap
            entries.append({"path": path, "caption": full_caption})
        out = get_image_dataset_store().save(name, entries, trigger_word=trigger)
        if "error" in out:
            raise HTTPException(400, out["error"])
        working_set.delete_working_set(wid)
        return out

    @router.get("")
    async def list_datasets():
        return {"datasets": get_image_dataset_store().list()}

    @router.get("/{name}")
    async def load(name: str):
        out = get_image_dataset_store().load(name)
        if "error" in out:
            raise HTTPException(404, out["error"])
        return out

    @router.delete("/{name}")
    async def delete(name: str):
        out = get_image_dataset_store().delete(name)
        if "error" in out:
            raise HTTPException(404, out["error"])
        return out

    return router
