"""HTTP control surface for native local models (Phase 3a).

Admin-guarded. Serve accepts only a .gguf path resolved inside MODELS_DIR.
"""
import os

from fastapi import APIRouter, Body, Depends, HTTPException

from core.middleware import require_admin
from src.constants import MODELS_DIR
from src.localmodels.manager import get_manager


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

    return router
