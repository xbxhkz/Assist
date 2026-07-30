"""Admin-gated Dataset builder/validator API (AI Studio)."""
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile

from core.middleware import require_admin
from src.dataset_tools.validate import validate_rows, validate_jsonl_text
from src.dataset_tools.store import get_dataset_store
from src.dataset_tools.generate import generate_rows
from src.dataset_tools.ground import generate_grounded, library_chunks, document_chunks

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
        except (TypeError, ValueError, OverflowError):
            count = 10
        count = max(1, min(count, MAX_GENERATE))
        return await generate_rows(
            fmt, count, body.get("brief", ""),
            seed_rows=body.get("seed_rows"), existing=body.get("existing"),
            model_call=_default_model_call, batch_size=10)

    @router.post("/generate/grounded")
    async def generate_grounded_route(body: dict = Body(...)):
        fmt = body.get("format") or body.get("fmt") or "text"
        try:
            count = int(body.get("count", 10))
        except (TypeError, ValueError, OverflowError):
            count = 10
        count = max(1, min(count, MAX_GENERATE))
        query = body.get("query")
        if not isinstance(query, str) or not query.strip():
            return {"rows": [], "valid": 0, "invalid": 0, "duplicates": 0,
                    "requested": count, "produced": 0, "chunks_used": 0,
                    "error": "a topic query is required"}
        try:
            k = int(body.get("k", 8))
        except (TypeError, ValueError, OverflowError):
            k = 8
        from src.industrial.manual_store import get_manual_store
        chunks = library_chunks(get_manual_store(), query, k=k)
        if not chunks:
            return {"rows": [], "valid": 0, "invalid": 0, "duplicates": 0,
                    "requested": count, "produced": 0, "chunks_used": 0,
                    "error": "no matching manual passages found"}
        return await generate_grounded(chunks, fmt, count, model_call=_default_model_call,
                                       existing=body.get("existing"), brief=body.get("brief", ""))

    @router.post("/generate/upload")
    async def generate_upload_route(file: UploadFile = File(...),
                                    format: str = Form("text"), count: str = Form("10"),
                                    brief: str = Form(""), existing: str = Form("")):
        import json as _json
        import os as _os
        import tempfile as _tempfile
        try:
            cnt = int(count)
        except (TypeError, ValueError, OverflowError):
            cnt = 10
        cnt = max(1, min(cnt, MAX_GENERATE))
        try:
            ex = _json.loads(existing) if existing else None
        except (ValueError, TypeError):
            ex = None
        empty = {"rows": [], "valid": 0, "invalid": 0, "duplicates": 0,
                 "requested": cnt, "produced": 0, "chunks_used": 0}
        fname = file.filename or "document"
        ext = _os.path.splitext(fname)[1].lower()
        tmp = None
        try:
            data = await file.read()
            fd, tmp = _tempfile.mkstemp(suffix=ext or ".bin")
            with _os.fdopen(fd, "wb") as fh:
                fh.write(data)
            chunks = document_chunks(tmp, ext, filename=fname)
            if not chunks:
                return {**empty, "error": "could not extract text from the uploaded file"}
            return await generate_grounded(chunks, format, cnt, model_call=_default_model_call,
                                           existing=ex, brief=brief)
        except Exception as e:  # noqa: BLE001 -- never 500
            return {**empty, "error": f"upload failed: {e}"}
        finally:
            if tmp:
                try:
                    _os.remove(tmp)
                except OSError:
                    pass

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
