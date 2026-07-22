"""Admin-gated Local Training API. All heavy work happens in the training
sidecar (a separate CUDA venv); these routes just orchestrate it."""
import asyncio

from fastapi import APIRouter, Body, Depends, HTTPException

from core.middleware import require_admin
from src.training.manager import get_training_manager
from src.training.config import TrainingConfig


def setup_training_routes() -> APIRouter:
    router = APIRouter(prefix="/api/training",
                       dependencies=[Depends(require_admin)])

    @router.get("/env")
    async def env():
        return get_training_manager().env_status()

    @router.post("/env/setup")
    async def env_setup():
        # the install is long (multi-GB); run it off the event loop
        return await asyncio.to_thread(get_training_manager().setup_env)

    @router.post("/runs")
    async def start_run(body: dict = Body(...)):
        try:
            cfg = TrainingConfig(
                base_model=str(body.get("base_model", "")),
                dataset_path=str(body.get("dataset_path", "")),
                lora_r=int(body.get("lora_r", 8)),
                lora_alpha=int(body.get("lora_alpha", 16)),
                lora_dropout=float(body.get("lora_dropout", 0.05)),
                steps=body.get("steps"),
                epochs=body.get("epochs"),
                batch_size=int(body.get("batch_size", 1)),
                learning_rate=float(body.get("learning_rate", 2e-4)),
                max_seq_length=int(body.get("max_seq_length", 512)),
            )
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"invalid config: {e}")
        out = get_training_manager().start(cfg)
        if "error" in out:
            raise HTTPException(400, out["error"])
        return out

    @router.get("/runs/current")
    async def current():
        return get_training_manager().status()

    @router.post("/runs/stop")
    async def stop():
        return get_training_manager().stop()

    @router.get("/adapters")
    async def adapters():
        return {"adapters": get_training_manager().list_adapters()}

    return router
