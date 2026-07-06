"""HTTP control surface for native local models (Phase 3a).

Admin-guarded. Serve accepts only a .gguf path resolved inside MODELS_DIR.
"""
import os
from urllib.parse import urlparse

from fastapi import APIRouter, Body, Depends, HTTPException

from core.middleware import require_admin
from src.constants import MODELS_DIR
from src.localmodels.manager import get_manager
from src.localmodels.catalog import search_gguf_models, list_repo_gguf_files
from src.localmodels.downloader import get_download_manager, _safe_filename
from src.localmodels.hardware import get_hardware, recommend_models, fit_for_file
from src.localmodels.external import (
    add_external_model, remove_external_model, is_registered_external,
)


def _validate_model_path(model_path: str) -> str:
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path is required")
    real_models = os.path.realpath(MODELS_DIR)
    real = os.path.realpath(model_path)
    try:
        inside = os.path.commonpath([real, real_models]) == real_models
    except ValueError:
        inside = False  # different drive on Windows
    # Serve is allowed for downloaded models (inside MODELS_DIR) or user-linked
    # ones; arbitrary unregistered paths stay rejected.
    if not inside and not is_registered_external(real):
        raise HTTPException(status_code=400,
                            detail="model_path must be a downloaded or linked model")
    if not real.lower().endswith(".gguf"):
        raise HTTPException(status_code=400,
                            detail="model_path must be a .gguf file")
    if not os.path.isfile(real):
        raise HTTPException(status_code=404, detail="model file not found")
    return real


def _validate_hf_download(url: str, filename: str):
    if not _safe_filename(filename):
        raise HTTPException(status_code=400,
                            detail="filename must be a plain .gguf name")
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        host = ""
    if host.lower() != "huggingface.co":
        raise HTTPException(status_code=400,
                            detail="url must be a huggingface.co download URL")


def setup_localmodels_routes() -> APIRouter:
    router = APIRouter(prefix="/api/localmodels",
                       dependencies=[Depends(require_admin)])

    @router.get("/models")
    async def list_models():
        models = get_manager().list_models()
        return {"models": models,
                "disk_bytes": sum(int(m.get("size") or 0) for m in models)}

    @router.get("/status")
    async def status():
        return get_manager().status()

    @router.post("/serve")
    async def serve(payload: dict = Body(...)):
        safe = _validate_model_path((payload.get("model_path") or "").strip())
        try:
            return get_manager().start(safe)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.post("/stop")
    async def stop():
        return get_manager().stop()

    @router.post("/add-external")
    async def add_external(payload: dict = Body(...)):
        """Register a user-picked .gguf from anywhere on disk as a linked model."""
        path = (payload.get("path") or "").strip()
        try:
            return add_external_model(path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/remove-external")
    async def remove_external(payload: dict = Body(...)):
        """Unlink a linked model from the list (the file on disk is untouched)."""
        remove_external_model((payload.get("path") or "").strip())
        return {"ok": True}

    @router.get("/catalog/search")
    async def catalog_search(q: str = "", sort: str = "downloads"):
        return {"results": search_gguf_models(q, sort=sort)}

    @router.get("/catalog/files")
    async def catalog_files(repo: str):
        files = list_repo_gguf_files(repo)
        hw = get_hardware()
        for f in files:
            f["fit"] = fit_for_file(f, hw)
        return {"files": files}

    @router.post("/download")
    async def download(payload: dict = Body(...)):
        url = (payload.get("url") or "").strip()
        filename = (payload.get("filename") or "").strip()
        _validate_hf_download(url, filename)
        try:
            return get_download_manager().start(url, filename)
        except (RuntimeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/download/status")
    async def download_status():
        return get_download_manager().status()

    @router.post("/download/cancel")
    async def download_cancel():
        return get_download_manager().cancel()

    @router.post("/delete")
    async def delete_model(payload: dict = Body(...)):
        filename = (payload.get("filename") or "").strip()
        try:
            return get_manager().delete_model(filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/hardware")
    async def hardware():
        return get_hardware()

    @router.get("/recommendations")
    async def recommendations():
        return {"recommendations": recommend_models()}

    return router
