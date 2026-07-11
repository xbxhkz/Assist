import src.agent_tools as agent_tools
import src.tool_security as ts
import src.tool_index as ti
import src.agent_loop as al

ACT = {"click_element", "set_element_text", "mouse", "keyboard"}
ALL5 = ACT | {"list_ui_elements"}


def test_all_five_in_tool_handlers_and_tags():
    for name in ALL5:
        assert name in agent_tools.TOOL_HANDLERS, f"{name} missing from TOOL_HANDLERS"
        assert name in agent_tools.TOOL_TAGS, f"{name} missing from TOOL_TAGS"


def test_all_five_admin_blocked():
    for name in ALL5:
        assert name in ts.NON_ADMIN_BLOCKED_TOOLS, f"{name} not admin-gated"


def test_plan_mode_partition():
    # Only the reader is plan-mode read-only; the four actors are blocked.
    assert "list_ui_elements" in ts.PLAN_MODE_READONLY_TOOLS
    for name in ACT:
        assert name not in ts.PLAN_MODE_READONLY_TOOLS, f"{name} must be blocked in plan mode"


def test_index_and_prompt_sections_present():
    for name in ALL5:
        assert name in ti.BUILTIN_TOOL_DESCRIPTIONS, f"{name} missing from tool index"
        assert name in al.TOOL_SECTIONS, f"{name} missing from TOOL_SECTIONS"


def test_plan_mode_actually_blocks_actors():
    blocked = ts.plan_mode_disabled_tools()
    for name in ACT:
        assert name in blocked, f"{name} must be blocked in plan mode"
    assert "list_ui_elements" not in blocked
