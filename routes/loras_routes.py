"""Admin-gated LoRA management: list, Civitai search, download (civitai/hf/url),
upload, delete. Downloads run off the event loop (asyncio.to_thread)."""
import asyncio
from contextlib import contextmanager

from fastapi import APIRouter, Body, Depends, File, HTTPException, UploadFile

from core.middleware import require_admin
from src.settings import get_setting
from src.imagemodels import loras, civitai


def setup_loras_routes() -> APIRouter:
    router = APIRouter(prefix="/api/loras", dependencies=[Depends(require_admin)])

    @router.get("")
    async def list_loras():
        return {"loras": loras.list_loras()}

    @router.get("/civitai/search")
    async def civitai_search(q: str = ""):
        token = get_setting("civitai_api_token", "") or None
        try:
            results = await asyncio.to_thread(civitai.search, q, token=token)
        except Exception as e:
            raise HTTPException(502, f"Civitai search failed: {e}")
        return {"results": results}

    @router.post("/download")
    async def download(body: dict = Body(...)):
        source = (body.get("source") or "").strip()
        token = get_setting("civitai_api_token", "") or None
        try:
            if source == "civitai":
                url = civitai.download_url_with_token(body.get("download_url", ""), token)
                fn = body.get("file_name") or "lora.safetensors"
                res = await asyncio.to_thread(loras.download_to_loras, url, fn)
            elif source == "hf":
                repo = (body.get("repo") or "").strip("/")
                filename = body.get("filename") or ""
                url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
                res = await asyncio.to_thread(loras.download_to_loras, url, filename)
            elif source == "url":
                res = await asyncio.to_thread(
                    loras.download_to_loras, body.get("url", ""), body.get("name") or "lora")
            else:
                raise HTTPException(400, "source must be civitai|hf|url")
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(502, f"LoRA download failed: {e}")
        return {"ok": True, "lora": res}

    @router.post("/upload")
    async def upload(file: UploadFile = File(...)):
        data = await file.read()

        @contextmanager
        def _mem(url, headers):
            yield len(data), iter([data])

        try:
            res = await asyncio.to_thread(
                loras.download_to_loras, "", file.filename or "lora", http_stream=_mem)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "lora": res}

    @router.delete("/{name}")
    async def delete(name: str):
        try:
            ok = loras.delete_lora(name)
        except ValueError:
            raise HTTPException(400, "invalid name")
        if not ok:
            raise HTTPException(404, "not found")
        return {"ok": True}

    return router
