"""face_swap must be registered everywhere a builtin tool needs to be,
applying every lesson sub-project 1's whole-branch review found (dispatcher
owner-threading, the can_generate_images privilege gate, the plan-mode
backstop) from the start. Mirrors
tests/test_edit_image_prompt_registration.py's structure. See
docs/superpowers/specs/2026-08-13-image-editing-face-swap-design.md."""
import asyncio
from pathlib import Path

_CHAT_ROUTES = Path(__file__).resolve().parent.parent / "routes" / "chat_routes.py"


def test_face_swap_registered_everywhere():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    from src.agent_loop import TOOL_SECTIONS, _DOMAIN_TOOL_MAP
    from src.tool_index import BUILTIN_TOOL_DESCRIPTIONS

    assert "face_swap" in TOOL_HANDLERS
    assert "face_swap" in TOOL_TAGS
    names = [(s.get("function") or {}).get("name") for s in FUNCTION_TOOL_SCHEMAS]
    assert "face_swap" in names
    assert "face_swap" in TOOL_SECTIONS
    assert "face_swap" in _DOMAIN_TOOL_MAP["desktop"]
    assert "face_swap" in BUILTIN_TOOL_DESCRIPTIONS


def test_face_swap_matches_generate_image_admin_gating():
    from src.tool_security import NON_ADMIN_BLOCKED_TOOLS
    generate_image_blocked = "generate_image" in NON_ADMIN_BLOCKED_TOOLS
    face_swap_blocked = "face_swap" in NON_ADMIN_BLOCKED_TOOLS
    assert face_swap_blocked == generate_image_blocked, (
        "face_swap's NON_ADMIN_BLOCKED_TOOLS membership must mirror "
        "generate_image's, not default to admin-only"
    )


def test_face_swap_blocked_when_can_generate_images_disabled():
    source = _CHAT_ROUTES.read_text(encoding="utf-8")
    assert 'if not _privs.get("can_generate_images", True):' in source
    idx = source.index('if not _privs.get("can_generate_images", True):')
    following = source[idx: idx + 300]
    assert "face_swap" in following, (
        "face_swap must be added to disabled_tools in the same "
        "can_generate_images privilege branch as generate_image/remove_background/edit_image_prompt"
    )


def test_dispatcher_threads_owner_and_session_into_tool_ctx(monkeypatch):
    """The REAL dispatcher (execute_tool_block, not a mock of it) must
    thread owner= into face_swap's ctx -- otherwise the tool falls into the
    generic dynamic_handlers catch-all, which never threads owner, and
    resolve_upload denies every real (owned) attachment for BOTH images
    this tool resolves."""
    import src.agent_tools as agent_tools
    from src.agent_tools import ToolBlock
    from src.tool_execution import execute_tool_block

    seen = {}

    async def spy(content, ctx):
        seen["content"] = content
        seen["ctx"] = ctx
        return {"output": "Face swapped.", "exit_code": 0}

    monkeypatch.setitem(agent_tools.TOOL_HANDLERS, "face_swap", spy)

    desc, result = asyncio.run(execute_tool_block(
        ToolBlock("face_swap", '{"source_face_id": "up-1", "target_image_id": "up-2"}'),
        session_id="sess-1",
        owner="alice",
    ))

    assert result.get("exit_code") == 0
    assert seen["ctx"].get("owner") == "alice", (
        "face_swap's ctx lost the owner -- resolve_upload will deny every owned attachment"
    )
    assert seen["ctx"].get("session_id") == "sess-1"


def test_face_swap_in_plan_mode_known_mutators():
    """face_swap writes a PNG to disk and inserts a Gallery DB row -- the
    same class of mutator as generate_image/edit_image/remove_background/
    edit_image_prompt, all members of _PLAN_MODE_KNOWN_MUTATORS."""
    import src.tool_security as ts
    assert "face_swap" in ts._PLAN_MODE_KNOWN_MUTATORS
    disabled = ts.plan_mode_disabled_tools()
    assert "face_swap" in disabled
