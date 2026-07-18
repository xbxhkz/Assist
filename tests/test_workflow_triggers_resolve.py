from src.workflows.triggers import resolve_trigger_inputs


def _wf(*names):
    return {"nodes": [{"id": n, "type": "input", "config": {"name": n}} for n in names]}


def test_schedule_uses_fixed_inputs_only_and_ignores_unknown():
    wf = _wf("topic")
    assert resolve_trigger_inputs(wf, {"topic": "AI", "nope": "x"}, None) == {"topic": "AI"}


def test_no_inputs_returns_empty():
    assert resolve_trigger_inputs(_wf(), {}, None) == {}
    assert resolve_trigger_inputs(_wf("a"), None, None) == {}


def test_webhook_top_level_body_maps_by_name():
    wf = _wf("topic", "lang")
    assert resolve_trigger_inputs(wf, {}, {"topic": "cats", "extra": 1}) == {"topic": "cats"}


def test_webhook_inputs_wrapper_takes_precedence_over_siblings():
    wf = _wf("topic")
    # when an "inputs" object is present, the top-level siblings are NOT scanned
    ctx = {"inputs": {"topic": "wrapped"}, "topic": "toplevel"}
    assert resolve_trigger_inputs(wf, {}, ctx) == {"topic": "wrapped"}


def test_event_message_injected_into_message_input_when_present():
    assert resolve_trigger_inputs(_wf("message"), {}, {"message": "hello"}) == {"message": "hello"}
    # no 'message' input -> event contributes nothing
    assert resolve_trigger_inputs(_wf("topic"), {"topic": "t"}, {"message": "hello"}) == {"topic": "t"}


def test_context_overrides_fixed():
    wf = _wf("topic")
    assert resolve_trigger_inputs(wf, {"topic": "fixed"}, {"topic": "ctx"}) == {"topic": "ctx"}
