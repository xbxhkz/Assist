from src.tool_execution import _tool_log_desc


def test_redacts_keyboard_and_set_element_text():
    assert "hunter2" not in _tool_log_desc("keyboard", '{"action":"type","text":"hunter2"}')
    assert "hunter2" not in _tool_log_desc("set_element_text", '{"automation_id":"x","text":"hunter2"}')


def test_keeps_nonsecret_tool_content():
    assert "focused" in _tool_log_desc("list_ui_elements", '{"window_id":"focused"}')
