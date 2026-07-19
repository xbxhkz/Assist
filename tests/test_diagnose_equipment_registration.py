def test_registered_in_handlers_and_tags():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    from src.agent_tools.industrial_tools import DiagnoseEquipmentTool
    assert "diagnose_equipment" in TOOL_HANDLERS
    assert "diagnose_equipment" in TOOL_TAGS
    assert TOOL_HANDLERS["diagnose_equipment"].__self__.__class__ is DiagnoseEquipmentTool


def test_gating_matches_capture_screen():
    import src.tool_security as ts
    # a read-only vision tool: admin-blocked AND plan-mode readonly (like capture_screen)
    assert "diagnose_equipment" in ts.NON_ADMIN_BLOCKED_TOOLS
    assert "diagnose_equipment" in ts.PLAN_MODE_READONLY_TOOLS
    assert "capture_screen" in ts.NON_ADMIN_BLOCKED_TOOLS and "capture_screen" in ts.PLAN_MODE_READONLY_TOOLS


def test_has_schema():
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    names = {s["function"]["name"] for s in FUNCTION_TOOL_SCHEMAS if s.get("type") == "function"}
    assert "diagnose_equipment" in names


def test_in_index_and_agent_loop_desktop_domain():
    import src.tool_index as ti
    import src.agent_loop as al
    assert "diagnose_equipment" in ti.BUILTIN_TOOL_DESCRIPTIONS
    assert "diagnose_equipment" in al.TOOL_SECTIONS
    assert "diagnose_equipment" in al._DOMAIN_TOOL_MAP["desktop"]
