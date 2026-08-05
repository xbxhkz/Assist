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
        endpoint_id = body.get("endpoint_id")
        if endpoint_id is not None:
            endpoint_id = str(endpoint_id).strip() or None
        db = SessionLocal()
        try:
            if endpoint_id:
                from core.database import ModelEndpoint
                q = db.query(ModelEndpoint).filter(
                    ModelEndpoint.id == endpoint_id,
                    ModelEndpoint.is_enabled == True,
                )
                q = owner_filter(q, ModelEndpoint, owner)
                if not q.first():
                    raise HTTPException(400, "Model endpoint no longer exists")
            import json
            enabled_tools = body.get("enabled_tools")
            if enabled_tools is not None:
                if not isinstance(enabled_tools, list) or not all(isinstance(t, str) for t in enabled_tools):
                    raise HTTPException(400, "enabled_tools must be a list of strings")

            def _s(key):
                v = body.get(key)
                return v if v is None or isinstance(v, str) else str(v)

            c = CrewMember(
                id=str(uuid.uuid4()),
                owner=owner,
                name=name,
                avatar=_s("avatar"),
                personality=_s("personality"),
                model=_s("model"),
                endpoint_id=endpoint_id,
                greeting=_s("greeting"),
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
                c.avatar = body["avatar"] if body["avatar"] is None or isinstance(body["avatar"], str) else str(body["avatar"])
            if "personality" in body:
                c.personality = body["personality"] if body["personality"] is None or isinstance(body["personality"], str) else str(body["personality"])
            if "model" in body:
                c.model = body["model"] if body["model"] is None or isinstance(body["model"], str) else str(body["model"])
            if "endpoint_id" in body:
                new_eid = body["endpoint_id"]
                new_eid = str(new_eid).strip() if new_eid else None
                if new_eid:
                    from core.database import ModelEndpoint
                    q = db.query(ModelEndpoint).filter(
                        ModelEndpoint.id == new_eid,
                        ModelEndpoint.is_enabled == True,
                    )
                    q = owner_filter(q, ModelEndpoint, owner)
                    if not q.first():
                        raise HTTPException(400, "Model endpoint no longer exists")
                c.endpoint_id = new_eid
            if "greeting" in body:
                c.greeting = body["greeting"] if body["greeting"] is None or isinstance(body["greeting"], str) else str(body["greeting"])
            if "enabled_tools" in body:
                _et = body["enabled_tools"]
                if _et is not None and (not isinstance(_et, list) or not all(isinstance(t, str) for t in _et)):
                    raise HTTPException(400, "enabled_tools must be a list of strings")
                c.enabled_tools = json.dumps(_et) if _et is not None else None
            if "is_active" in body:
                c.is_active = bool(body["is_active"])
            if "sort_order" in body:
                try:
                    c.sort_order = int(body["sort_order"])
                except (TypeError, ValueError):
                    raise HTTPException(400, "sort_order must be an integer")
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
