"""Live workflow-run enumeration (Mission Control sub-project 2c).

Reads src.workflow_runs' in-memory registry of currently-executing workflow
runs. Admin-gated the same way as the rest of the workflows subsystem
(routes/workflow_routes.py) -- no new access-control decision, just reuse of
the existing gate. See
docs/superpowers/specs/2026-08-11-mission-control-workflow-runs-design.md.
"""
from fastapi import APIRouter, Depends

from core.middleware import require_admin
from src import workflow_runs


def setup_workflow_runs_routes() -> APIRouter:
    router = APIRouter(prefix="/api/workflow-runs", tags=["workflow-runs"],
                        dependencies=[Depends(require_admin)])

    @router.get("/active")
    async def get_active_workflow_runs():
        return {"active": workflow_runs.list_active()}

    return router
