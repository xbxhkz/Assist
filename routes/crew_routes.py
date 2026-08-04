"""Owner-scoped CRUD over CrewMember ("persona") rows -- NOT admin-gated,
matching routes/assistant_routes.py's own gating (every authenticated user
manages their own personas). The default Assistant (is_default_assistant=True)
appears in the same list; its timezone/check-in extras stay exclusively in
assistant_routes.py."""
import uuid

from fastapi import APIRouter, Body, HTTPException, Request

from core.database import SessionLocal, CrewMember
from src.auth_helpers import get_current_user, owner_filter
from src.crew_helpers import crew_to_dict
from src.tool_policy import known_tool_names


def _owner(request: Request) -> str:
    owner = get_current_user(request)
    if not owner:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return owner


def setup_crew_routes() -> APIRouter:
    router = APIRouter(prefix="/api/crew", tags=["crew"])

    @router.get("/tool-names")
    async def tool_names():
        # Import agent_tools early to resolve the circular import between tool_schemas
        # and agent_tools, ensuring that known_tool_names() includes cmd/powershell.
        try:
            import src.agent_tools  # noqa: F401
        except Exception:
            pass
        return {"tools": sorted(known_tool_names())}

    @router.get("")
    async def list_crew(request: Request):
        owner = _owner(request)
        db = SessionLocal()
        try:
            q = db.query(CrewMember)
            q = owner_filter(q, CrewMember, owner, include_shared=False)
            rows = q.order_by(CrewMember.sort_order.asc()).all()
            return {"crew": [crew_to_dict(c) for c in rows]}
        finally:
            db.close()

    @router.post("")
    async def create_crew(request: Request, body: dict = Body(...)):
        owner = _owner(request)
        name = str(body.get("name", "")).strip()
        if not name:
            raise HTTPException(400, "name is required")
        db = SessionLocal()
        try:
            import json
            enabled_tools = body.get("enabled_tools")
            c = CrewMember(
                id=str(uuid.uuid4()),
                owner=owner,
                name=name,
                avatar=body.get("avatar"),
                personality=body.get("personality"),
                model=body.get("model"),
                endpoint_url=body.get("endpoint_url"),
                greeting=body.get("greeting"),
                enabled_tools=json.dumps(enabled_tools) if enabled_tools is not None else None,
            )
            db.add(c)
            db.commit()
            return crew_to_dict(c)
        finally:
            db.close()

    def _find_owned(db, crew_id: str, owner: str):
        q = db.query(CrewMember).filter(CrewMember.id == crew_id)
        q = owner_filter(q, CrewMember, owner, include_shared=False)
        return q.first()

    @router.patch("/{crew_id}")
    async def update_crew(crew_id: str, request: Request, body: dict = Body(...)):
        owner = _owner(request)
        db = SessionLocal()
        try:
            c = _find_owned(db, crew_id, owner)
            if not c:
                raise HTTPException(404, "Persona not found")
            import json
            if "name" in body and str(body["name"]).strip():
                c.name = str(body["name"]).strip()
            if "avatar" in body:
                c.avatar = body["avatar"]
            if "personality" in body:
                c.personality = body["personality"]
            if "model" in body:
                c.model = body["model"]
            if "endpoint_url" in body:
                c.endpoint_url = body["endpoint_url"]
            if "greeting" in body:
                c.greeting = body["greeting"]
            if "enabled_tools" in body:
                c.enabled_tools = json.dumps(body["enabled_tools"]) if body["enabled_tools"] is not None else None
            if "is_active" in body:
                c.is_active = bool(body["is_active"])
            if "sort_order" in body:
                c.sort_order = int(body["sort_order"])
            db.commit()
            return crew_to_dict(c)
        finally:
            db.close()

    @router.delete("/{crew_id}")
    async def delete_crew(crew_id: str, request: Request):
        owner = _owner(request)
        db = SessionLocal()
        try:
            c = _find_owned(db, crew_id, owner)
            if not c:
                raise HTTPException(404, "Persona not found")
            if c.is_default_assistant:
                raise HTTPException(400, "Cannot delete the default Assistant")
            db.delete(c)
            db.commit()
            return {"ok": True}
        finally:
            db.close()

    return router
