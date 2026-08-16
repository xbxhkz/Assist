"""detect_shapes must be registered everywhere a builtin tool needs to be,
applying every lesson prior image-tool sub-projects' whole-branch reviews
found (dispatcher owner-threading, the can_generate_images privilege gate,
the plan-mode backstop) from the start. Mirrors
tests/test_face_swap_registration.py's structure. See
docs/superpowers/specs/2026-08-15-shape-detection-design.md."""
import asyncio
from pathlib import Path

_CHAT_ROUTES = Path(__file__).resolve().parent.parent / "routes" / "chat_routes.py"


def test_detect_shapes_registered_everywhere():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    from src.agent_loop import TOOL_SECTIONS, _DOMAIN_TOOL_MAP
    from src.tool_index import BUILTIN_TOOL_DESCRIPTIONS

    assert "detect_shapes" in TOOL_HANDLERS
    assert "detect_shapes" in TOOL_TAGS
    names = [(s.get("function") or {}).get("name") for s in FUNCTION_TOOL_SCHEMAS]
    assert "detect_shapes" in names
    assert "detect_shapes" in TOOL_SECTIONS
    assert "detect_shapes" in _DOMAIN_TOOL_MAP["desktop"]
    assert "detect_shapes" in BUILTIN_TOOL_DESCRIPTIONS


def test_detect_shapes_matches_generate_image_admin_gating():
    from src.tool_security import NON_ADMIN_BLOCKED_TOOLS
    generate_image_blocked = "generate_image" in NON_ADMIN_BLOCKED_TOOLS
    detect_shapes_blocked = "detect_shapes" in NON_ADMIN_BLOCKED_TOOLS
    assert detect_shapes_blocked == generate_image_blocked, (
        "detect_shapes's NON_ADMIN_BLOCKED_TOOLS membership must mirror "
        "generate_image's, not default to admin-only"
    )


def test_detect_shapes_blocked_when_can_generate_images_disabled():
    source = _CHAT_ROUTES.read_text(encoding="utf-8")
    assert 'if not _privs.get("can_generate_images", True):' in source
    idx = source.index('if not _privs.get("can_generate_images", True):')
    following = source[idx: idx + 300]
    assert "detect_shapes" in following, (
        "detect_shapes must be added to disabled_tools in the same "
        "can_generate_images privilege branch as generate_image/remove_background/"
        "edit_image_prompt/face_swap"
    )


def test_dispatcher_threads_owner_and_session_into_tool_ctx(monkeypatch):
    """The REAL dispatcher (execute_tool_block, not a mock of it) must
    thread owner= into detect_shapes's ctx -- otherwise the tool falls into
    the generic dynamic_handlers catch-all, which never threads owner, and
    resolve_upload denies every real (owned) attachment."""
    import src.agent_tools as agent_tools
    from src.agent_tools import ToolBlock
    from src.tool_execution import execute_tool_block

    seen = {}

    async def spy(content, ctx):
        seen["content"] = content
        seen["ctx"] = ctx
        return {"output": "detected", "exit_code": 0}

    monkeypatch.setitem(agent_tools.TOOL_HANDLERS, "detect_shapes", spy)

    desc, result = asyncio.run(execute_tool_block(
        ToolBlock("detect_shapes", '{"attachment_id": "up-1"}'),
        session_id="sess-1",
        owner="alice",
    ))

    assert result.get("exit_code") == 0
    assert seen["ctx"].get("owner") == "alice", (
        "detect_shapes's ctx lost the owner -- resolve_upload will deny every owned attachment"
    )
    assert seen["ctx"].get("session_id") == "sess-1"


def test_detect_shapes_in_plan_mode_known_mutators():
    """detect_shapes writes an annotated PNG to disk and inserts a Gallery
    DB row (best-effort) -- the same class of mutator as
    generate_image/remove_background/edit_image_prompt/face_swap, all
    members of _PLAN_MODE_KNOWN_MUTATORS."""
    import src.tool_security as ts
    assert "detect_shapes" in ts._PLAN_MODE_KNOWN_MUTATORS
    disabled = ts.plan_mode_disabled_tools()
    assert "detect_shapes" in disabled
