"""Operator action protocol: the model returns exactly one structured Action
per round. The vocabulary reuses existing Desktop Control + Input Automation
tools. Anything malformed or disallowed becomes an `ask` — never a wild action."""
from dataclasses import dataclass, field
import json

MUTATING_TOOLS = {"launch_app", "control_window", "click_element",
                  "set_element_text", "mouse", "keyboard"}
READONLY_TOOLS = {"find_files", "list_windows", "list_ui_elements", "capture_screen"}
OPERATOR_TOOLS = MUTATING_TOOLS | READONLY_TOOLS


@dataclass
class Action:
    kind: str                         # "act" | "wait" | "ask" | "done"
    tool: str | None = None           # for kind == "act"
    args: dict = field(default_factory=dict)
    rationale: str = ""


def is_mutating(action):
    return action.kind == "act" and action.tool in MUTATING_TOOLS


def parse_action(reply):
    """Parse the model reply into ONE Action. The model must emit a JSON object
    {kind, tool?, args?, rationale?}. A genuine model question is kind="ask"
    (pauses for the user). A MALFORMED / unknown-tool / unknown-kind reply is
    kind="invalid" (the loop self-corrects by re-prompting — it never blocks the
    user with a dead question and never executes a wild action)."""
    text = (reply or "").strip()
    try:
        data = json.loads(text[text.index("{"): text.rindex("}") + 1])
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except (ValueError, TypeError):
        return Action(kind="invalid", rationale="could not parse the model's reply as a JSON action")
    kind = str(data.get("kind", "")).strip().lower()
    rationale = str(data.get("rationale", "") or data.get("question", ""))
    if kind in ("wait", "done", "ask"):
        return Action(kind=kind, rationale=rationale)
    if kind == "act":
        tool = str(data.get("tool", "")).strip()
        if tool not in OPERATOR_TOOLS:
            return Action(kind="invalid",
                          rationale=f"tool {tool!r} is not available to the operator; "
                                    f"use one of {sorted(OPERATOR_TOOLS)}")
        args = data.get("args")
        return Action(kind="act", tool=tool,
                      args=args if isinstance(args, dict) else {}, rationale=rationale)
    return Action(kind="invalid", rationale=f"unrecognized action kind {kind!r}; "
                                            "use kind act|wait|ask|done")
