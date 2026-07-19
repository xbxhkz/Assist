def test_run_workflow_registered_everywhere():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    from src.agent_tools.workflow_tool import RunWorkflowTool
    assert "run_workflow" in TOOL_HANDLERS
    assert "run_workflow" in TOOL_TAGS
    # handler is the tool's execute
    assert TOOL_HANDLERS["run_workflow"].__self__.__class__ is RunWorkflowTool


def test_run_workflow_is_admin_only_and_not_plan_readonly():
    import src.tool_security as ts
    assert "run_workflow" in ts.NON_ADMIN_BLOCKED_TOOLS
    # executes side effects -> must NOT be plan-mode read-only
    assert "run_workflow" not in getattr(ts, "PLAN_MODE_READONLY_TOOLS", set())


def test_run_workflow_has_schema():
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    names = {s["function"]["name"] for s in FUNCTION_TOOL_SCHEMAS if s.get("type") == "function"}
    assert "run_workflow" in names


def test_run_workflow_in_index_and_agent_loop():
    import src.tool_index as ti
    import src.agent_loop as al
    assert "run_workflow" in ti.BUILTIN_TOOL_DESCRIPTIONS
    assert "run_workflow" in al.TOOL_SECTIONS
    assert any("run_workflow" in tools for tools in al._DOMAIN_TOOL_MAP.values())
