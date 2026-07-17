"""Admin-gated workflow management + execution.

A workflow runs the LLM and arbitrary agent tools (each tool's own admin/consent
gates still apply on top), so the whole router is behind require_admin."""
import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from core.middleware import require_admin
from src.auth_helpers import get_current_user
from src.workflows import store
from src.workflows.engine import run_workflow
from src.workflows.model import WorkflowError, validate

logger = logging.getLogger(__name__)

# A hostile id/body is untrusted input: the store may reject it as a ValueError
# (path-unsafe id), a TypeError (non-string id), or an OSError (id with
# filesystem-invalid characters). All three mean "bad request" -> 400, never 500.
_BAD_INPUT = (ValueError, TypeError, OSError)


def _bad_request(detail):
    # Every error body is {"errors": [...]} so clients parse one shape.
    errors = detail if isinstance(detail, list) else [str(detail)]
    return HTTPException(400, {"errors": errors})


def setup_workflow_routes() -> APIRouter:
    router = APIRouter(prefix="/api/workflows", dependencies=[Depends(require_admin)])

    @router.get("")
    async def list_workflows():
        return {"workflows": store.list_workflows()}

    @router.post("")
    async def save_workflow(body: dict = Body(...)):
        errs = validate(body)
        if errs:
            raise _bad_request(errs)
        try:
            return store.save_workflow(body)
        except _BAD_INPUT as e:
            raise _bad_request(e)

    @router.get("/{wid}")
    async def get_workflow(wid: str):
        try:
            wf = store.get_workflow(wid)
        except _BAD_INPUT as e:
            raise _bad_request(e)
        if not wf:
            raise HTTPException(404, {"errors": ["workflow not found"]})
        return wf

    @router.delete("/{wid}")
    async def delete_workflow(wid: str):
        try:
            return {"deleted": store.delete_workflow(wid)}
        except _BAD_INPUT as e:
            raise _bad_request(e)

    @router.post("/{wid}/run")
    async def run(wid: str, request: Request, body: dict = Body(default={})):
        try:
            wf = store.get_workflow(wid)
        except _BAD_INPUT as e:
            raise _bad_request(e)
        if not wf:
            raise HTTPException(404, {"errors": ["workflow not found"]})
        owner = get_current_user(request)
        try:
            return await run_workflow(wf, body.get("inputs") or {}, {"owner": owner})
        except WorkflowError as e:
            raise _bad_request(e.errors)

    return router
