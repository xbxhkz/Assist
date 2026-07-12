import src.operator.actions as a


def test_parse_act_mutating():
    act = a.parse_action('{"kind":"act","tool":"click_element","args":{"name":"Save"},"rationale":"click save"}')
    assert act.kind == "act" and act.tool == "click_element" and act.args == {"name": "Save"}
    assert a.is_mutating(act) is True


def test_parse_act_readonly_not_mutating():
    act = a.parse_action('{"kind":"act","tool":"list_ui_elements","args":{}}')
    assert act.kind == "act" and a.is_mutating(act) is False


def test_parse_done_wait_ask():
    assert a.parse_action('{"kind":"done","rationale":"finished"}').kind == "done"
    assert a.parse_action('{"kind":"wait"}').kind == "wait"
    assert a.parse_action('{"kind":"ask","rationale":"which file?"}').kind == "ask"


def test_parse_tolerates_json_fence_and_prose():
    act = a.parse_action('Sure!\n```json\n{"kind":"act","tool":"keyboard","args":{"action":"type","text":"hi"}}\n```')
    assert act.kind == "act" and act.tool == "keyboard"


def test_malformed_becomes_ask():
    assert a.parse_action("not json at all").kind == "ask"
    assert a.parse_action("").kind == "ask"
    assert a.parse_action("{bad json").kind == "ask"


def test_unknown_tool_becomes_ask():
    assert a.parse_action('{"kind":"act","tool":"rm_rf","args":{}}').kind == "ask"


def test_unknown_kind_becomes_ask():
    assert a.parse_action('{"kind":"teleport"}').kind == "ask"
