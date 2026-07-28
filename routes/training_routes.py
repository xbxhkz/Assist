"""Admin-gated Local Training API. All heavy work happens in the training
sidecar (a separate CUDA venv); these routes just orchestrate it."""
import asyncio
import os

from fastapi import APIRouter, Body, Depends, HTTPException

from core.middleware import require_admin
from src.training.manager import get_training_manager
from src.training.config import TrainingConfig
from src.training.convert_manager import get_adapter_converter
from src.training.export_manager import get_adapter_exporter, VALID_QUANTS
from src.training.base_resolve import resolve_base_gguf
from src.localmodels.manager import get_manager as get_local_manager
from src.localmodels.runtime import list_gguf_models
from src.constants import MODELS_DIR


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

    def _find_adapter(run_id):
        for a in get_training_manager().list_adapters():
            if a.get("run_id") == run_id:
                return a
        return None

    @router.get("/adapters/{run_id}")
    async def adapter_status(run_id: str):
        a = _find_adapter(run_id)
        if not a:
            raise HTTPException(404, "adapter not found")
        names = [m["name"] for m in list_gguf_models(MODELS_DIR) if isinstance(m, dict) and m.get("name")]
        match = resolve_base_gguf(a.get("base_model"), names)
        return {**a, "base_match": match, "exports": get_adapter_exporter().list_exports(run_id)}

    @router.post("/adapters/{run_id}/convert")
    async def convert_adapter(run_id: str):
        a = _find_adapter(run_id)
        if not a:
            raise HTTPException(404, "adapter not found")
        out = await asyncio.to_thread(get_adapter_converter().convert, a["path"])
        if "error" in out:
            raise HTTPException(400, out["error"])
        return out

    @router.post("/adapters/{run_id}/serve")
    async def serve_adapter(run_id: str, body: dict = Body(...)):
        a = _find_adapter(run_id)
        if not a:
            raise HTTPException(404, "adapter not found")
        if not a.get("converted") or not a.get("adapter_gguf"):
            raise HTTPException(400, "adapter is not converted yet")
        base = str(body.get("base_gguf", "")).strip()
        if not base:
            raise HTTPException(400, "base_gguf is required")
        if not os.path.isabs(base):
            base = os.path.join(MODELS_DIR, base)  # accept a bare GGUF filename from base_match
        alias = f"{os.path.splitext(os.path.basename(base))[0]} · {run_id} (LoRA)"
        # start() blocks until llama-server is ready and RAISES RuntimeError on a
        # failed/mismatched-base serve (the wrong-base path). Offload it off the
        # event loop and surface the log tail as a clean error — never a 500.
        try:
            out = await asyncio.to_thread(
                lambda: get_local_manager().start(base, lora=a["adapter_gguf"], alias=alias))
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        if isinstance(out, dict) and out.get("error"):
            raise HTTPException(400, out["error"])
        return out

    @router.post("/adapters/{run_id}/export")
    async def export_adapter(run_id: str, body: dict = Body(...)):
        a = _find_adapter(run_id)
        if not a:
            raise HTTPException(404, "adapter not found")
        quant = str(body.get("quant", "Q4_K_M"))
        if quant not in VALID_QUANTS:
            raise HTTPException(400, f"invalid quant (allowed: {', '.join(VALID_QUANTS)})")
        out = await asyncio.to_thread(get_adapter_exporter().export, a["path"], quant, a.get("base_model"))
        if "error" in out:
            raise HTTPException(400, out["error"])
        return out

    @router.post("/exports/reveal")
    async def reveal_exports():
        d = get_adapter_exporter().exports_dir()
        try:
            os.makedirs(d, exist_ok=True)
            sf = getattr(os, "startfile", None)
            if sf:
                sf(d)  # open the exports folder in the OS file manager (Windows)
            return {"ok": True, "path": d}
        except Exception as e:
            return {"ok": False, "path": d, "error": str(e)}

    return router
