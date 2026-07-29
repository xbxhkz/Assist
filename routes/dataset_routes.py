"""Admin-gated Dataset builder/validator API (AI Studio)."""
from fastapi import APIRouter, Body, Depends, HTTPException

from core.middleware import require_admin
from src.dataset_tools.validate import validate_rows, validate_jsonl_text
from src.dataset_tools.store import get_dataset_store
from src.dataset_tools.generate import generate_rows

MAX_GENERATE = 200


async def _default_model_call(prompt, *, system=None, owner=None):
    """Call the configured default chat model over the OpenAI-compat API.
    Mirrors src/workflows/nodes.py:default_model_call. Raises if no endpoint."""
    import httpx
    from src.endpoint_resolver import resolve_endpoint
    url, model, headers = resolve_endpoint("default", owner=owner)
    if not url:
        raise RuntimeError("no default model endpoint configured")
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    body = {"model": model, "messages": messages, "stream": False}
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(url, json=body, headers=headers or {})
        resp.raise_for_status()
        data = resp.json()
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")


def setup_dataset_routes() -> APIRouter:
    router = APIRouter(prefix="/api/datasets",
                       dependencies=[Depends(require_admin)])

    @router.post("/validate")
    async def validate(body: dict = Body(...)):
        if isinstance(body.get("text"), str):
            return validate_jsonl_text(body["text"])
        return validate_rows(body.get("rows", []))

    @router.post("/generate")
    async def generate(body: dict = Body(...)):
        fmt = body.get("format") or body.get("fmt") or "text"
        try:
            count = int(body.get("count", 10))
        except (TypeError, ValueError):
            count = 10
        count = max(1, min(count, MAX_GENERATE))
        return await generate_rows(
            fmt, count, body.get("brief", ""),
            seed_rows=body.get("seed_rows"), existing=body.get("existing"),
            model_call=_default_model_call, batch_size=10)

    @router.post("")
    async def save(body: dict = Body(...)):
        out = get_dataset_store().save(body.get("name"), body.get("rows", []))
        if "error" in out:
            raise HTTPException(400, out["error"])
        return out

    @router.get("")
    async def list_datasets():
        return {"datasets": get_dataset_store().list()}

    @router.get("/{name}")
    async def load(name: str):
        out = get_dataset_store().load(name)
        if "error" in out:
            raise HTTPException(404, out["error"])
        return out

    @router.delete("/{name}")
    async def delete(name: str):
        out = get_dataset_store().delete(name)
        if "error" in out:
            raise HTTPException(404, out["error"])
        return out

    return router
