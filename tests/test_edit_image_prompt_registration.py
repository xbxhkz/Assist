"""edit_image_prompt must be registered everywhere a builtin tool needs to be,
applying every lesson sub-project 1's whole-branch review found (the
dispatcher owner-threading branch, the can_generate_images privilege gate,
and the plan-mode backstop) from the start, not discovered again after the
fact. Mirrors tests/test_remove_background_registration.py's structure. See
docs/superpowers/specs/2026-08-12-image-editing-prompt-edit-design.md."""
import asyncio
from pathlib import Path

_CHAT_ROUTES = Path(__file__).resolve().parent.parent / "routes" / "chat_routes.py"


def test_edit_image_prompt_registered_everywhere():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    from src.agent_loop import TOOL_SECTIONS, _DOMAIN_TOOL_MAP
    from src.tool_index import BUILTIN_TOOL_DESCRIPTIONS

    assert "edit_image_prompt" in TOOL_HANDLERS
    assert "edit_image_prompt" in TOOL_TAGS
    names = [(s.get("function") or {}).get("name") for s in FUNCTION_TOOL_SCHEMAS]
    assert "edit_image_prompt" in names
    assert "edit_image_prompt" in TOOL_SECTIONS
    assert "edit_image_prompt" in _DOMAIN_TOOL_MAP["desktop"]
    assert "edit_image_prompt" in BUILTIN_TOOL_DESCRIPTIONS


def test_edit_image_prompt_matches_generate_image_admin_gating():
    from src.tool_security import NON_ADMIN_BLOCKED_TOOLS
    generate_image_blocked = "generate_image" in NON_ADMIN_BLOCKED_TOOLS
    edit_image_prompt_blocked = "edit_image_prompt" in NON_ADMIN_BLOCKED_TOOLS
    assert edit_image_prompt_blocked == generate_image_blocked, (
        "edit_image_prompt's NON_ADMIN_BLOCKED_TOOLS membership must mirror "
        "generate_image's, not default to admin-only"
    )


def test_edit_image_prompt_blocked_when_can_generate_images_disabled():
    source = _CHAT_ROUTES.read_text(encoding="utf-8")
    assert 'if not _privs.get("can_generate_images", True):' in source
    idx = source.index('if not _privs.get("can_generate_images", True):')
    following = source[idx: idx + 300]
    assert "edit_image_prompt" in following, (
        "edit_image_prompt must be added to disabled_tools in the same "
        "can_generate_images privilege branch as generate_image/remove_background"
    )


def test_dispatcher_threads_owner_and_session_into_tool_ctx(monkeypatch):
    """The REAL dispatcher (execute_tool_block, not a mock of it) must thread
    owner= into edit_image_prompt's ctx -- otherwise the tool falls into the
    generic dynamic_handlers catch-all, which never threads owner, and
    resolve_upload denies every real (owned) attachment. This is the exact
    gap sub-project 1's whole-branch review found for remove_background,
    applied here from the start."""
    import src.agent_tools as agent_tools
    from src.agent_tools import ToolBlock
    from src.tool_execution import execute_tool_block

    seen = {}

    async def spy(content, ctx):
        seen["content"] = content
        seen["ctx"] = ctx
        return {"output": "Image edited.", "exit_code": 0}

    monkeypatch.setitem(agent_tools.TOOL_HANDLERS, "edit_image_prompt", spy)

    desc, result = asyncio.run(execute_tool_block(
        ToolBlock("edit_image_prompt", '{"attachment_id": "up-1", "prompt": "add a hat"}'),
        session_id="sess-1",
        owner="alice",
    ))

    assert result.get("exit_code") == 0
    assert seen["ctx"].get("owner") == "alice", (
        "edit_image_prompt's ctx lost the owner -- resolve_upload will deny "
        "every owned attachment"
    )
    assert seen["ctx"].get("session_id") == "sess-1"


def test_edit_image_prompt_in_plan_mode_known_mutators():
    """edit_image_prompt writes a PNG to disk and inserts a Gallery DB row --
    the same class of mutator as generate_image/edit_image/remove_background,
    all members of _PLAN_MODE_KNOWN_MUTATORS, the defense-in-depth backstop
    that keeps known mutators blocked in plan mode even if the schema-derived
    denylist logic fails to import."""
    import src.tool_security as ts
    assert "edit_image_prompt" in ts._PLAN_MODE_KNOWN_MUTATORS
    disabled = ts.plan_mode_disabled_tools()
    assert "edit_image_prompt" in disabled
