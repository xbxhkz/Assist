import src.agent_tools as agent_tools
import src.tool_security as ts
import src.tool_index as ti
import src.agent_loop as al

NET = {"net_info", "discover_hosts", "scan_ports"}


def test_all_in_handlers_and_tags():
    for n in NET:
        assert n in agent_tools.TOOL_HANDLERS and n in agent_tools.TOOL_TAGS


def test_all_admin_blocked_and_plan_readonly():
    for n in NET:
        assert n in ts.NON_ADMIN_BLOCKED_TOOLS
        assert n in ts.PLAN_MODE_READONLY_TOOLS


def test_index_and_prompt_sections():
    for n in NET:
        assert n in ti.BUILTIN_TOOL_DESCRIPTIONS and n in al.TOOL_SECTIONS
