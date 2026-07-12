"""The bounded operator loop. All I/O is injected as async callables so the
loop unit-tests with fakes (no real screen/model/mouse)."""
import logging
import time

from src.operator.actions import is_mutating

logger = logging.getLogger(__name__)


def require_consent(get_setting):
    """Raise PermissionError unless BOTH screen access and input control are on."""
    if not get_setting("screen_access_enabled", False):
        raise PermissionError("Screen access is off — enable 'Allow screen access' in the sidebar.")
    if not get_setting("input_control_enabled", False):
        raise PermissionError("Input control is off — enable 'Allow input control' in the sidebar.")


async def run_operator(goal, *, perceive, decide, execute, confirm, ask,
                       max_rounds=30, max_seconds=600, now=time.monotonic):
    history = []
    start = now()
    last_act_percept = None  # percept captured just before the previous mutating act
    for rounds in range(1, max_rounds + 1):
        if now() - start > max_seconds:
            return {"status": "time_cap", "rounds": rounds - 1, "history": history}
        percept = await perceive()
        action = await decide(goal, history, percept)
        if action.kind == "done":
            history.append(("done", action.rationale))
            logger.info("operator done after %d round(s)", rounds)
            return {"status": "done", "rounds": rounds, "history": history}
        if last_act_percept is not None and percept == last_act_percept:
            return {"status": "stuck", "rounds": rounds, "history": history}
        last_act_percept = None
        if action.kind == "ask":
            answer = await ask(action.rationale)
            history.append(("ask", action.rationale, answer))
            continue
        if action.kind == "wait":
            history.append(("wait", action.rationale))
            continue
        # kind == "act"
        mutating = is_mutating(action)
        if mutating:
            decision = await confirm(action)
            if decision == "stop":
                logger.info("operator stopped by user at round %d", rounds)
                return {"status": "stopped", "rounds": rounds, "history": history}
            if decision == "deny":
                history.append(("denied", action.tool, action.args))
                continue
            if isinstance(decision, tuple) and decision and decision[0] == "edit":
                action.args = decision[1]
        obs = await execute(action)
        history.append(("act", action.tool, action.args, obs))
        logger.info("operator round %d: %s %s", rounds, action.tool, action.args)
        if mutating:
            last_act_percept = percept
    return {"status": "round_cap", "rounds": max_rounds, "history": history}
