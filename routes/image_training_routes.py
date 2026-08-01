"""Admin-gated Image LoRA Training API. All heavy work happens in the
image-training sidecar (the existing training venv extended with
diffusers); these routes just orchestrate it. Mirrors
routes/training_routes.py's shape, scoped to the single family/toolchain
the feasibility spike proved: SDXL via diffusers (see
docs/superpowers/specs/2026-07-31-image-lora-training-engine-design.md)."""
import asyncio

from fastapi import APIRouter, Body, Depends, HTTPException

from core.middleware import require_admin
from src.image_training.manager import get_image_training_manager
from src.image_training.config import ImageTrainingConfig


def setup_image_training_routes() -> APIRouter:
    router = APIRouter(prefix="/api/image-training",
                       dependencies=[Depends(require_admin)])

    @router.get("/env")
    async def env():
        return get_image_training_manager().env_status()

    @router.post("/env/setup")
    async def env_setup():
        # the install may pull diffusers (multi-hundred-MB); run it off the event loop
        return await asyncio.to_thread(get_image_training_manager().setup_env)

    @router.post("/runs")
    async def start_run(body: dict = Body(...)):
        try:
            cfg = ImageTrainingConfig(
                dataset_name=str(body.get("dataset_name", "")),
                output_name=str(body.get("output_name", "")),
                base_model=str(body.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")),
                rank=int(body.get("rank", 4)),
                lora_alpha=int(body.get("lora_alpha", 4)),
                learning_rate=float(body.get("learning_rate", 1e-4)),
                steps=int(body.get("steps", 1000)),
                resolution=int(body.get("resolution", 1024)),
            )
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"invalid config: {e}")
        out = get_image_training_manager().start(cfg)
        if "error" in out:
            raise HTTPException(400, out["error"])
        return out

    @router.get("/runs/current")
    async def current():
        return get_image_training_manager().status()

    @router.post("/runs/stop")
    async def stop():
        return get_image_training_manager().stop()

    return router
