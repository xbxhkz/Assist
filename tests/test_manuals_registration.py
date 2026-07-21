def test_registered_handlers_and_tags():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    from src.agent_tools.industrial_manuals import (
        IngestEquipmentManualTool, SearchEquipmentManualTool)
    assert TOOL_HANDLERS["ingest_equipment_manual"].__self__.__class__ is IngestEquipmentManualTool
    assert TOOL_HANDLERS["search_equipment_manual"].__self__.__class__ is SearchEquipmentManualTool
    assert {"ingest_equipment_manual", "search_equipment_manual"} <= TOOL_TAGS


def test_admin_and_plan_gating():
    import src.tool_security as ts
    for name in ("ingest_equipment_manual", "search_equipment_manual"):
        assert name in ts.NON_ADMIN_BLOCKED_TOOLS
    # search is read-only in plan mode; ingest is a mutator (must NOT be readonly)
    assert "search_equipment_manual" in ts.PLAN_MODE_READONLY_TOOLS
    assert "ingest_equipment_manual" not in ts.PLAN_MODE_READONLY_TOOLS
    assert "ingest_equipment_manual" in ts._PLAN_MODE_KNOWN_MUTATORS
    # and the derived plan-mode denylist blocks ingest but not search
    disabled = ts.plan_mode_disabled_tools()
    assert "ingest_equipment_manual" in disabled
    assert "search_equipment_manual" not in disabled


def test_schema_index_sections_domain():
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    import src.tool_index as ti
    import src.agent_loop as al
    names = {s["function"]["name"] for s in FUNCTION_TOOL_SCHEMAS if s.get("type") == "function"}
    for name in ("ingest_equipment_manual", "search_equipment_manual"):
        assert name in names
        assert name in ti.BUILTIN_TOOL_DESCRIPTIONS
        assert name in al.TOOL_SECTIONS
        assert name in al._DOMAIN_TOOL_MAP["desktop"]
