"""remove_background must be registered everywhere a builtin tool needs to
be -- this codebase has a documented gotcha (found building webcam_look)
that a new tool needs the full registration set, not just a handler entry.
See docs/superpowers/specs/2026-08-12-image-editing-background-removal-design.md."""
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
