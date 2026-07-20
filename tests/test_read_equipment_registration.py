def test_registered_handler_and_tag():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    from src.agent_tools.industrial_live import ReadEquipmentTool
    assert "read_equipment" in TOOL_HANDLERS
    assert "read_equipment" in TOOL_TAGS
    assert TOOL_HANDLERS["read_equipment"].__self__.__class__ is ReadEquipmentTool


def test_gating_matches_netscan():
    import src.tool_security as ts
    # a read-only network tool: admin-blocked AND plan-mode readonly (like scan_ports)
    assert "read_equipment" in ts.NON_ADMIN_BLOCKED_TOOLS
    assert "read_equipment" in ts.PLAN_MODE_READONLY_TOOLS
    assert "scan_ports" in ts.NON_ADMIN_BLOCKED_TOOLS and "scan_ports" in ts.PLAN_MODE_READONLY_TOOLS


def test_schema_index_and_agent_loop():
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    import src.tool_index as ti
    import src.agent_loop as al
    names = {s["function"]["name"] for s in FUNCTION_TOOL_SCHEMAS if s.get("type") == "function"}
    assert "read_equipment" in names
    assert "read_equipment" in ti.BUILTIN_TOOL_DESCRIPTIONS
    assert "read_equipment" in al.TOOL_SECTIONS
    assert "read_equipment" in al._DOMAIN_TOOL_MAP["network"]
