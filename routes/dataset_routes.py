"""Admin-gated Dataset builder/validator API (AI Studio)."""
from fastapi import APIRouter, Body, Depends, HTTPException

from core.middleware import require_admin
from src.dataset_tools.validate import validate_rows, validate_jsonl_text
from src.dataset_tools.store import get_dataset_store


def setup_dataset_routes() -> APIRouter:
    router = APIRouter(prefix="/api/datasets",
                       dependencies=[Depends(require_admin)])

    @router.post("/validate")
    async def validate(body: dict = Body(...)):
        if isinstance(body.get("text"), str):
            return validate_jsonl_text(body["text"])
        return validate_rows(body.get("rows", []))

    @router.post("")
    async def save(body: dict = Body(...)):
        out = get_dataset_store().save(body.get("name"), body.get("rows", []))
        if "error" in out:
            raise HTTPException(400, out["error"])
        return out

    @router.get("")
    async def list_datasets():
        return {"datasets": get_dataset_store().list()}

    @router.get("/{name}")
    async def load(name: str):
        out = get_dataset_store().load(name)
        if "error" in out:
            raise HTTPException(404, out["error"])
        return out

    @router.delete("/{name}")
    async def delete(name: str):
        out = get_dataset_store().delete(name)
        if "error" in out:
            raise HTTPException(404, out["error"])
        return out

    return router
