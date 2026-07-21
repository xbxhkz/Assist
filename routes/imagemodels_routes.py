"""HTTP control surface for native local image models (sd-server).

Admin-guarded. Serve takes a diffusion .gguf + device (cpu/gpu), resolves the
shared FLUX encoders/VAE, and runs the (slow) model load off the event loop.
A served model auto-registers as a model_type="image" endpoint, so the existing
gallery/chat image flow uses it with no changes.
"""
import asyncio
import os

from fastapi import APIRouter, Body, Depends, HTTPException

from core.middleware import require_admin
from src.imagemodels.manager import get_manager
from src.imagemodels.serve_resolve import resolve_image_files
from src.imagemodels.encoders import MissingEncoderError


def setup_imagemodels_routes() -> APIRouter:
    router = APIRouter(prefix="/api/imagemodels",
                       dependencies=[Depends(require_admin)])

    @router.get("/models")
    async def list_models():
        return {"models": get_manager().list_models()}

    @router.get("/status")
    async def status():
        return get_manager().status()

    @router.post("/serve")
    async def serve(payload: dict = Body(...)):
        diff = (payload.get("diffusion_model") or "").strip()
        device = (payload.get("device") or "cpu").strip().lower()
        if device not in ("cpu", "gpu"):
            device = "cpu"
        real = os.path.realpath(diff)
        if not diff or not real.lower().endswith(".gguf") or not os.path.isfile(real):
            raise HTTPException(400, "diffusion_model must be an existing .gguf file")
        try:
            files = resolve_image_files(
                real, llm=payload.get("llm"), vae=payload.get("vae"),
                t5xxl=payload.get("t5xxl"), clip_l=payload.get("clip_l"))
        except MissingEncoderError as e:
            raise HTTPException(400, getattr(e, "hint", str(e)))
        steps = payload.get("steps")
        try:
            steps = max(1, min(50, int(steps))) if steps else None
        except (TypeError, ValueError):
            steps = None
        if payload.get("fast_decode") and "checkpoint" not in files:
            from src.imagemodels.encoders import find_taesd
            tae = find_taesd(real)
            if tae:
                files["taesd"] = tae
        try:
            # Off the event loop: image models load slowly (large mmap).
            return await asyncio.to_thread(get_manager().start, files, device, steps)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.post("/stop")
    async def stop():
        return get_manager().stop()

    return router
