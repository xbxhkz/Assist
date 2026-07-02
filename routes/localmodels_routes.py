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


def _validate_model_path(model_path: str) -> str:
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path is required")
    real_models = os.path.realpath(MODELS_DIR)
    real = os.path.realpath(model_path)
    try:
        inside = os.path.commonpath([real, real_models]) == real_models
    except ValueError:
        inside = False  # different drive on Windows
    if not inside:
        raise HTTPException(status_code=400,
                            detail="model_path must be inside the models directory")
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
        return {"models": get_manager().list_models()}

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

    @router.get("/catalog/search")
    async def catalog_search(q: str = "", sort: str = "downloads"):
        return {"results": search_gguf_models(q, sort=sort)}

    @router.get("/catalog/files")
    async def catalog_files(repo: str):
        return {"files": list_repo_gguf_files(repo)}

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

    return router
