"""remove_background must be registered everywhere a builtin tool needs to
be -- this codebase has a documented gotcha (found building webcam_look)
that a new tool needs the full registration set, not just a handler entry.
See docs/superpowers/specs/2026-08-12-image-editing-background-removal-design.md."""


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
