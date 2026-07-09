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
from src.gguf_meta import read_gguf_architecture
from src.imagemodels.manager import get_manager
from src.imagemodels.encoders import (
    resolve_flux_files, resolve_flux2_files, resolve_zimage_files,
    MissingEncoderError,
)
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
        base = os.path.basename(real).lower()
        # Z-Image (lumina2 arch): Qwen3 --llm encoder + the FLUX.1 ae VAE.
        if (read_gguf_architecture(real) == "lumina2"
                or "z-image" in base or "z_image" in base):
            try:
                files = resolve_zimage_files(
                    real, llm=payload.get("llm"), vae=payload.get("vae"))
            except MissingEncoderError as e:
                raise HTTPException(
                    400, "Missing Z-Image files: " + ", ".join(e.missing) +
                    ". Z-Image needs a Qwen3-4B text encoder (llm, e.g. "
                    "Qwen3-4B-Q4_K_M.gguf) and the FLUX.1 VAE (vae, "
                    "ae.safetensors) next to the model or in the shared "
                    "encoders folder.")
        # FLUX.2 (klein) shares the 'flux' GGUF arch tag but takes a Qwen3/
        # Mistral --llm text encoder + the flux2 VAE instead of t5xxl/clip_l.
        elif looks_like_flux2(real):
            try:
                files = resolve_flux2_files(
                    real, llm=payload.get("llm"), vae=payload.get("vae"))
            except MissingEncoderError as e:
                raise HTTPException(
                    400, "Missing FLUX.2 files: " + ", ".join(e.missing) +
                    ". klein needs a Qwen3 text encoder (llm, e.g. "
                    "Qwen3-4B-Q4_K_M.gguf) and the FLUX.2 VAE (vae, "
                    "flux2_ae.safetensors) next to the model or in the "
                    "shared encoders folder.")
        else:
            try:
                files = resolve_flux_files(
                    real, t5xxl=payload.get("t5xxl"),
                    clip_l=payload.get("clip_l"), vae=payload.get("vae"))
            except MissingEncoderError as e:
                raise HTTPException(
                    400, "Missing FLUX files: " + ", ".join(e.missing) +
                    ". Put t5xxl / clip_l / vae next to the model, or download "
                    "the FLUX encoders.")
        steps = payload.get("steps")
        try:
            steps = max(1, min(50, int(steps))) if steps else None
        except (TypeError, ValueError):
            steps = None
        if payload.get("fast_decode"):
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
