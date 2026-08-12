"""remove_background must be registered everywhere a builtin tool needs to
be -- this codebase has a documented gotcha (found building webcam_look)
that a new tool needs the full registration set, not just a handler entry.
See docs/superpowers/specs/2026-08-12-image-editing-background-removal-design.md."""
import asyncio
from pathlib import Path

_CHAT_ROUTES = Path(__file__).resolve().parent.parent / "routes" / "chat_routes.py"


def test_remove_background_registered_everywhere():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    from src.agent_loop import TOOL_SECTIONS, _DOMAIN_TOOL_MAP
    from src.tool_index import BUILTIN_TOOL_DESCRIPTIONS

    assert "remove_background" in TOOL_HANDLERS
    assert "remove_background" in TOOL_TAGS
    names = [(s.get("function") or {}).get("name") for s in FUNCTION_TOOL_SCHEMAS]
    assert "remove_background" in names
    assert "remove_background" in TOOL_SECTIONS
    assert "remove_background" in _DOMAIN_TOOL_MAP["desktop"]
    assert "remove_background" in BUILTIN_TOOL_DESCRIPTIONS


def test_remove_background_matches_generate_image_admin_gating():
    from src.tool_security import NON_ADMIN_BLOCKED_TOOLS
    generate_image_blocked = "generate_image" in NON_ADMIN_BLOCKED_TOOLS
    remove_background_blocked = "remove_background" in NON_ADMIN_BLOCKED_TOOLS
    assert remove_background_blocked == generate_image_blocked, (
        "remove_background's NON_ADMIN_BLOCKED_TOOLS membership must mirror "
        "generate_image's, not default to admin-only"
    )


def test_remove_background_blocked_when_can_generate_images_disabled():
    """routes/chat_routes.py must disable remove_background in chat when a
    user's can_generate_images privilege is False -- the same privilege that
    gates the identical capability at the HTTP route
    (routes/gallery/gallery_routes.py's require_privilege(request,
    "can_generate_images")) and at the generate_image chat tool. Without
    this, a user without the privilege could still reach remove_background
    even though they can't reach generate_image."""
    source = _CHAT_ROUTES.read_text(encoding="utf-8")
    assert 'if not _privs.get("can_generate_images", True):' in source
    # Find the disabled_tools statement(s) immediately following that guard
    # and confirm remove_background is added alongside generate_image.
    idx = source.index('if not _privs.get("can_generate_images", True):')
    following = source[idx: idx + 300]
    assert "remove_background" in following, (
        "remove_background must be added to disabled_tools in the same "
        "can_generate_images privilege branch as generate_image"
    )


def test_dispatcher_threads_owner_and_session_into_tool_ctx(monkeypatch):
    """The REAL dispatcher (execute_tool_block, not a mock of it) must thread
    owner= into remove_background's ctx.

    Without an explicit dispatch branch the tool falls into the generic
    dynamic_handlers catch-all, which calls _direct_fallback(tool, content,
    progress_cb=...) with no owner -- so ctx["owner"] is None, the tool calls
    upload_handler.resolve_upload(attachment_id, owner=None), and resolve_upload
    denies any upload record that HAS an owner (i.e. every real upload in an
    auth-enabled install). The tool then returns "attachment not found" for
    every real user, every time, while still appearing to work in no-auth mode.
    """
    import src.agent_tools as agent_tools
    from src.agent_tools import ToolBlock
    from src.tool_execution import execute_tool_block

    seen = {}

    async def spy(content, ctx):
        seen["content"] = content
        seen["ctx"] = ctx
        return {"output": "Background removed.", "exit_code": 0}

    monkeypatch.setitem(agent_tools.TOOL_HANDLERS, "remove_background", spy)

    desc, result = asyncio.run(execute_tool_block(
        ToolBlock("remove_background", '{"attachment_id": "up-1"}'),
        session_id="sess-1",
        owner="alice",
    ))

    assert result.get("exit_code") == 0
    assert seen["content"] == '{"attachment_id": "up-1"}'
    assert seen["ctx"].get("owner") == "alice", (
        "remove_background's ctx lost the owner -- resolve_upload will deny "
        "every owned attachment"
    )
    assert seen["ctx"].get("session_id") == "sess-1"


def test_remove_background_in_plan_mode_known_mutators():
    """remove_background writes a PNG to disk and inserts a Gallery DB row --
    the same class of mutator as generate_image/edit_image/
    ingest_equipment_manual, all of which are members of
    _PLAN_MODE_KNOWN_MUTATORS, the defense-in-depth backstop that keeps known
    mutators blocked in plan mode even if the schema-derived denylist logic
    fails to import. Mirrors tests/test_manuals_registration.py's precedent
    for ingest_equipment_manual."""
    import src.tool_security as ts
    assert "remove_background" in ts._PLAN_MODE_KNOWN_MUTATORS
    disabled = ts.plan_mode_disabled_tools()
    assert "remove_background" in disabled
