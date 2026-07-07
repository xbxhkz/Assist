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
from src.imagemodels.encoders import resolve_flux_files, MissingEncoderError
from src.imagemodels.runtime import looks_like_flux2


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
        if looks_like_flux2(real):
            # Same 'flux' GGUF arch tag as FLUX.1, but needs a Mistral --llm
            # text encoder — with the FLUX.1 t5xxl/clip_l set it fails only
            # after minutes of loading, so reject up front.
            raise HTTPException(
                400, "FLUX.2 models aren't supported yet — they need a Mistral "
                "(--llm) text encoder instead of the FLUX.1 t5xxl/clip_l set. "
                "Use a FLUX.1 GGUF for now.")
        try:
            files = resolve_flux_files(
                real, t5xxl=payload.get("t5xxl"),
                clip_l=payload.get("clip_l"), vae=payload.get("vae"))
        except MissingEncoderError as e:
            raise HTTPException(
                400, "Missing FLUX files: " + ", ".join(e.missing) +
                ". Put t5xxl / clip_l / vae next to the model, or download the "
                "FLUX encoders.")
        try:
            # Off the event loop: image models load slowly (large mmap).
            return await asyncio.to_thread(get_manager().start, files, device)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.post("/stop")
    async def stop():
        return get_manager().stop()

    return router
