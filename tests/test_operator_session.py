import asyncio
import src.settings as settings
import src.operator.session as s
from src.operator.actions import Action


def _run(**kw):
    return asyncio.run(s.run_operator("goal", **kw))


def _seq(actions):
    it = iter(actions)
    async def decide(goal, history, percept):
        return next(it)
    return decide


async def _perceive_const():
    return "SCREEN"


async def _approve(action):
    return "approve"


async def _ask_noop(q):
    return "answer"


def test_default_operator_bounds():
    assert settings.DEFAULT_SETTINGS.get("operator_max_rounds") == 30
    assert settings.DEFAULT_SETTINGS.get("operator_max_seconds") == 600


def test_require_consent_needs_both(monkeypatch):
    import pytest
    store = {"screen_access_enabled": True, "input_control_enabled": False}
    with pytest.raises(PermissionError):
        s.require_consent(lambda k, d=None: store.get(k, d))
    store["input_control_enabled"] = True
    s.require_consent(lambda k, d=None: store.get(k, d))  # no raise


def test_stops_on_done():
    calls = []
    async def execute(a): calls.append(a.tool); return {"output": "ok"}
    r = _run(perceive=_perceive_const, decide=_seq([Action(kind="done", rationale="fin")]),
             execute=execute, confirm=_approve, ask=_ask_noop)
    assert r["status"] == "done" and calls == []


def test_mutating_requires_confirm_readonly_does_not():
    # Percept changes each round (unlike _perceive_const) so this test exercises
    # confirm-gating in isolation, without tripping the objective stuck guard —
    # a real screen would change after the mutating "mouse" click below.
    screens = iter(["S1", "S2", "S3"])
    async def perceive(): return next(screens)
    confirmed = []
    async def confirm(a): confirmed.append(a.tool); return "approve"
    async def execute(a): return {"output": "ok"}
    r = _run(perceive=perceive,
             decide=_seq([Action(kind="act", tool="list_ui_elements"),      # read-only
                          Action(kind="act", tool="mouse", args={"action": "click", "x": 1, "y": 2}),  # mutating
                          Action(kind="done")]),
             execute=execute, confirm=confirm, ask=_ask_noop)
    assert confirmed == ["mouse"]  # only the mutating action was confirmed
    assert r["status"] == "done"


def test_deny_never_executes():
    executed = []
    async def execute(a): executed.append(a.tool); return {"output": "ok"}
    async def deny(a): return "deny"
    r = _run(perceive=_perceive_const,
             decide=_seq([Action(kind="act", tool="keyboard", args={"action": "type", "text": "x"}),
                          Action(kind="done")]),
             execute=execute, confirm=deny, ask=_ask_noop)
    assert executed == [] and r["status"] == "done"


def test_edit_replaces_args():
    seen = {}
    async def execute(a): seen.update(a.args); return {"output": "ok"}
    async def edit(a): return ("edit", {"action": "type", "text": "EDITED"})
    r = _run(perceive=_perceive_const,
             decide=_seq([Action(kind="act", tool="keyboard", args={"action": "type", "text": "orig"}),
                          Action(kind="done")]),
             execute=execute, confirm=edit, ask=_ask_noop)
    assert seen == {"action": "type", "text": "EDITED"}


def test_stop_ends_session():
    async def stop(a): return "stop"
    async def execute(a): return {"output": "ok"}
    r = _run(perceive=_perceive_const,
             decide=_seq([Action(kind="act", tool="mouse", args={"action": "click", "x": 1, "y": 1})]),
             execute=execute, confirm=stop, ask=_ask_noop)
    assert r["status"] == "stopped"


def test_round_cap():
    async def execute(a): return {"output": "ok"}
    # always returns a read-only act (auto-runs, never done) -> hits the cap
    async def decide(goal, history, percept): return Action(kind="act", tool="list_windows")
    r = asyncio.run(s.run_operator("g", perceive=_perceive_const, decide=decide,
                                   execute=execute, confirm=_approve, ask=_ask_noop, max_rounds=3))
    assert r["status"] == "round_cap" and r["rounds"] == 3


def test_no_progress_after_mutating_act_is_stuck():
    async def execute(a): return {"output": "ok"}
    async def decide(goal, history, percept):
        return Action(kind="act", tool="mouse", args={"action": "click", "x": 1, "y": 1})
    # percept never changes -> after the first mutating act, next round detects no progress
    r = asyncio.run(s.run_operator("g", perceive=_perceive_const, decide=decide,
                                   execute=execute, confirm=_approve, ask=_ask_noop, max_rounds=10))
    assert r["status"] == "stuck"


def test_time_cap():
    clock = {"t": 0.0}
    def now(): clock["t"] += 100.0; return clock["t"]
    async def decide(goal, history, percept): return Action(kind="act", tool="list_windows")
    async def execute(a): return {"output": "ok"}
    r = asyncio.run(s.run_operator("g", perceive=_perceive_const, decide=decide, execute=execute,
                                   confirm=_approve, ask=_ask_noop, max_rounds=100, max_seconds=150, now=now))
    assert r["status"] == "time_cap"


def test_stuck_guard_wins_over_done():
    # After a no-progress mutating act, the objective stuck guard fires BEFORE
    # the model is consulted — a queued "done" claim is never even reached.
    async def execute(a): return {"output": "ok"}
    r = _run(perceive=_perceive_const,
             decide=_seq([Action(kind="act", tool="mouse", args={"action": "click", "x": 1, "y": 1}),
                          Action(kind="done", rationale="claims finished")]),
             execute=execute, confirm=_approve, ask=_ask_noop)
    assert r["status"] == "stuck"


def test_confirm_gate_fails_closed_on_unrecognized_decision():
    executed = []
    async def execute(a): executed.append(a.tool); return {"output": "ok"}
    async def weird(a): return None  # malformed confirm return
    r = _run(perceive=_perceive_const,
             decide=_seq([Action(kind="act", tool="keyboard", args={"action": "type", "text": "x"}),
                          Action(kind="done")]),
             execute=execute, confirm=weird, ask=_ask_noop)
    assert executed == []  # fail closed: never executed without explicit approve


def test_should_stop_true_ends_immediately():
    async def execute(a): return {}
    async def decide(g, h, p): return Action(kind="act", tool="list_windows")
    r = __import__("asyncio").run(s.run_operator(
        "g", perceive=_perceive_const, decide=decide, execute=execute,
        confirm=_approve, ask=_ask_noop, should_stop=lambda: True))
    assert r["status"] == "stopped"


def test_should_stop_checked_each_round():
    n = {"c": 0}
    def should_stop():
        n["c"] += 1
        return n["c"] > 2  # stop on the 3rd round's check
    seq = iter(range(1000))
    async def perceive(): return next(seq)   # changing percept -> never "stuck"
    async def decide(g, h, p): return Action(kind="act", tool="list_windows")  # read-only, auto-runs, never done
    async def execute(a): return {}
    r = __import__("asyncio").run(s.run_operator(
        "g", perceive=perceive, decide=decide, execute=execute,
        confirm=_approve, ask=_ask_noop, should_stop=should_stop, max_rounds=50))
    assert r["status"] == "stopped"
