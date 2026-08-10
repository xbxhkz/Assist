"""Source-presence tests for the Tool Call History panel (Mission Control
sub-project 2a) -- mirrors tests/test_mission_control_ui.py's established
style: HTML scaffold present, modal registers correctly, fetch call targets
the right endpoint, esc() applied to tool-call text, node --check syntax gate.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_index_has_tool_call_log_modal_and_entries():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in (
        'id="tool-call-log-modal"',
        'id="rail-tool-calls"',
        'id="tool-tool-calls-btn"',
        'id="tool-call-log-list"',
        'id="tool-call-log-tool-filter"',
        'id="tool-call-log-session-filter-input"',
        'id="tool-call-log-apply-filter"',
        'id="tool-call-log-clear-filter"',
        'id="tool-call-log-more"',
        '/static/js/toolCallLog.js',
    ):
        assert el in html, "missing %s" % el


def test_tool_call_log_js_registers_with_modal_manager():
    src = (ROOT / "static" / "js" / "toolCallLog.js").read_text(encoding="utf-8")
    assert "Modals.register" in src
    assert "'tool-call-log-modal'" in src or '"tool-call-log-modal"' in src


def test_tool_call_log_js_fetches_tool_calls():
    src = (ROOT / "static" / "js" / "toolCallLog.js").read_text(encoding="utf-8")
    assert "/api/tool-calls" in src


def test_tool_call_log_js_escapes_command_and_output():
    src = (ROOT / "static" / "js" / "toolCallLog.js").read_text(encoding="utf-8")
    assert "esc(c.command" in src
    assert "esc((c.output" in src


def test_tool_call_log_js_uses_select_session_for_jump():
    src = (ROOT / "static" / "js" / "toolCallLog.js").read_text(encoding="utf-8")
    assert "sessionModule" in src
    assert "selectSession" in src


def test_tool_call_log_js_syntax():
    result = subprocess.run(
        ["node", "--check", str(ROOT / "static" / "js" / "toolCallLog.js")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
