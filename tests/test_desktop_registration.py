DESKTOP = ["launch_app", "find_files", "list_windows", "control_window", "capture_screen"]


def test_all_registered_everywhere():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    from src.agent_loop import TOOL_SECTIONS, _DOMAIN_TOOL_MAP
    from src.tool_index import BUILTIN_TOOL_DESCRIPTIONS
    from src.tool_security import NON_ADMIN_BLOCKED_TOOLS
    names = [(s.get("function") or {}).get("name") for s in FUNCTION_TOOL_SCHEMAS]
    for t in DESKTOP:
        assert t in TOOL_HANDLERS, f"{t} not in TOOL_HANDLERS"
        assert t in TOOL_TAGS, f"{t} not in TOOL_TAGS"
        assert t in names, f"{t} not in FUNCTION_TOOL_SCHEMAS"
        assert t in TOOL_SECTIONS, f"{t} not in TOOL_SECTIONS"
        assert t in _DOMAIN_TOOL_MAP["desktop"], f"{t} not in desktop domain"
        assert t in BUILTIN_TOOL_DESCRIPTIONS, f"{t} not in BUILTIN_TOOL_DESCRIPTIONS"
        assert t in NON_ADMIN_BLOCKED_TOOLS, f"{t} not admin-gated"
