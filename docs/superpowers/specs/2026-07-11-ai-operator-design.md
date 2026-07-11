# AI Operator — Design Spec

**Status:** Approved for planning (2026-07-11)
**Sub-project 5 of 5** in the desktop-agent initiative (the capstone). Depends on shipped #1 Desktop Control (screen capture + VLM) and #2 Input Automation (mouse/keyboard + UI Automation).

## Goal

A permissioned, **goal-directed AI Operator**: the user gives it a task ("fill out this form", "help me configure X", "find and open the Q3 report"), and it runs a **bounded, confirmation-gated loop** — *perceive → decide → confirm → act → recapture* — executing **one user-approved action at a time** until the goal is done or the user stops it. Entirely local. It is an **orchestration layer** over already-shipped capabilities (screen reading, UI Automation, input actuators, the agent model + VLM); the only new code is the control loop, the structured-action protocol, the confirmation gate, and a panel.

## Scope

**In scope:** a goal-directed `OperatorSession` orchestrator with injectable primitives; a structured one-action-per-round decide step; a per-action confirmation gate (Approve/Deny/Edit/Stop); reuse of the existing Desktop Control + Input Automation tools as the action vocabulary; both-consent + admin gating; a panic stop + "Operator active" indicator; bounds (max rounds, wall-clock cap, no-progress detection); backend routes + a plain-IIFE Operator panel; audit logging.

**Out of scope (YAGNI / not this sub-project):** a perpetual background screen-watcher (this is goal-directed, capture only during a session); fully autonomous acting without per-action confirmation (v1 is per-action confirmation; "confirm-the-plan then auto-run" is a deliberate phase-2 upgrade); multi-application macro recording/replay; learning/goal-memory across sessions; any new low-level desktop capability (all actuators/perception already exist).

## Architecture

A new **`src/operator/` package** with a testable orchestrator that reuses the existing tool + perception layers through **injectable primitives** (so the whole loop unit-tests with fakes — no real screen, model, or mouse):

- **`src/operator/session.py`** — `OperatorSession` running the bounded loop with four injected primitives:
  - `perceive() -> Percept` — gather the current structured view: the focused/target window's UIA element tree (via `src.desktop.uia.list_elements`) as `elements`, plus lightweight window context (`src.desktop.windows.list_windows`). **No VLM by default** (on-demand — see below).
  - `decide(goal, history, percept) -> Action` — call the agent model with `{goal, history, percept}` and parse its reply into exactly **one** `Action` (a structured dataclass: `kind`, `tool`, `args`, `rationale`). Enforces the one-action-per-round contract; a malformed/unparseable reply becomes `Action(kind="ask", rationale="could not determine next step")`.
  - `confirm(action) -> Decision` — surface the proposed action + rationale to the user; returns `approve | deny | edit(new_args) | stop`. Only invoked for **mutating** actions.
  - `execute(action) -> Observation` — dispatch the action to the existing tool handler (`TOOL_HANDLERS[action.tool]`), returning its `{output|error}`.
- **`src/operator/actions.py`** — the `Action` dataclass + the **action vocabulary** (which existing tools are operator-usable and which are mutating vs read-only) + the parser/validator for the model's structured reply.
- **`routes/operator_routes.py`** — session lifecycle: `POST /api/operator/start {goal}`, `GET /api/operator/status` (current round, pending action awaiting confirmation, transcript), `POST /api/operator/decision {approve|deny|edit|stop}`, `POST /api/operator/stop`. One session at a time (server-side state); admin-gated.
- **`static/js/operator.js` + panel in `static/index.html`** — a sidebar "AI Operator" entry → modal: goal box + Start; a live transcript of **action cards** (round #, proposed action, rationale, optional screen thumbnail) each with **Approve / Deny / Edit / Stop**; a persistent "Operator active" indicator; a prominent **Stop** (panic) button. Plain IIFE, CSP-safe, mirrors the other panels; polls `/status`.

## The action vocabulary (reuses existing tools)

The model returns **one** structured action per round: `{kind, tool?, args?, rationale}`. Kinds:

- **`act`** — invoke one existing desktop/input tool (`tool` + `args`). The usable set:
  - *Mutating (require confirmation):* `launch_app`, `control_window`, `click_element`, `set_element_text`, `mouse`, `keyboard`.
  - *Read-only (auto-run, no confirmation — they change nothing):* `find_files`, `list_windows`, `list_ui_elements`, `capture_screen`. `capture_screen` is the **on-demand VLM look** the model requests when the UIA tree isn't enough (returns the VLM screen description into the next round's history).
- **`wait`** — let the UI settle (bounded sleep), then re-perceive.
- **`ask`** — pause and ask the user for input/clarification (surfaced in the panel); the loop resumes with the user's answer.
- **`done`** — the goal is complete; ends the session with a summary.

The confirmation gate applies to **`act` with a mutating tool** only. Read-only `act`, `wait`, `ask`, and `done` do not touch the user's input and are not gated (though `ask`/`done` inherently involve the user). New desktop tools added later automatically become operator-usable by extending the vocabulary tables.

## Data flow

```
user goal → OperatorSession.run():
  loop (bounded):
    percept   = perceive()                    # UIA element tree + windows
    action    = decide(goal, history, percept) # model → ONE structured Action
    if action.kind == "done": end(summary)
    if action.kind == "ask":  pause → user answer → history
    if action.kind == "wait": sleep → continue
    if action.kind == "act":
        if mutating(action.tool):
            decision = confirm(action)          # Approve/Deny/Edit/Stop
            if stop: end;  if deny: history += "denied" ; continue
            if edit: action.args = new_args
        obs = execute(action)                   # existing TOOL_HANDLERS[tool]
        history += (action, obs)
    if no_progress(percept): end("stuck — asking user")
```

## Permission / safety model

The Operator is the most powerful capability in the initiative — it chains perception and action in a loop — so it stacks every existing guardrail and adds its own:

- **Both consents required to START:** `screen_access_enabled` AND `input_control_enabled` must be ON, or `start` refuses with a message naming the two sidebar toggles. (These already gate the underlying capture + input tools, so the operator cannot act without them regardless.)
- **Per-action confirmation** on every mutating action — nothing moves the mouse/keyboard or launches/controls anything without an explicit Approve. Deny makes the model reconsider; Edit lets the user adjust args; Stop ends the session.
- **Panic stop:** a prominent Stop button ends the session immediately AND flips `input_control_enabled` **off** (belt-and-suspenders); a persistent "Operator active" indicator shows whenever a session is running.
- **Bounded:** hard **max rounds** (default 30), an overall **wall-clock cap** (default 10 minutes), and **no-progress detection** (if the UIA element tree is unchanged after a mutating action, stop and ask rather than flailing). All three are settings (`operator_max_rounds`, `operator_max_seconds`) with the stated defaults.
- **Admin-gated** (routes use `require_admin`); **single active session** at a time.
- **Auditability:** every round, decision, and executed action logs one INFO line to `app.log`. Screen capture happens **only during an active session** (privacy) and only on-demand.

## Dependency strategy

No new third-party dependency. Reuses: `src.desktop.uia` / `windows` / `capture` (#1, #2), `src.agent_tools` `TOOL_HANDLERS` (the action executors), the running agent model (for `decide`) and the vision model (for on-demand `capture_screen`). The only new packages are `src/operator/` and `routes/operator_routes.py` + the panel. No frozen-build dependency risk (pure Python over existing code).

## Testing strategy (TDD, injected primitives — no real GUI/model/mouse)

- **`actions.py`:** the parser turns a well-formed model reply into the right `Action`; a malformed/empty/non-JSON reply → `Action(kind="ask")`; the mutating/read-only classification is correct for each tool.
- **`session.py` loop** (fake `perceive`/`decide`/`confirm`/`execute`):
  - stops on `done`; stops at the **round cap**; stops on **no-progress** (identical percept after a mutating act).
  - a **mutating** action calls `confirm`; a **read-only** action does NOT; `deny` → the action is never `execute`d and the model is asked to reconsider; `edit` → `execute` receives the edited args; `stop` → session ends immediately.
  - `ask` pauses and resumes with the injected user answer; `wait` re-perceives without executing.
  - `execute` dispatches `act` to `TOOL_HANDLERS[tool]` (fake handler records the call).
- **Consent guard:** `start` refuses unless both `screen_access_enabled` and `input_control_enabled` are set (injected settings).
- **Routes:** start/status/decision/stop drive the session state machine (fake session); admin-gated; single-session enforced.
- **UI guard test:** the sidebar entry + Operator panel elements (`operator-panel`, goal input, transcript, Stop) present; `operator.js` posts `/api/operator/start` and `/api/operator/decision` and reads `/api/operator/status`.
- **Live verification on the packaged exe:** with screen access + input control ON, start a small goal (e.g. "open Notepad and type 'hello world', then save as hello.txt"); confirm each proposed action; verify the operator opens Notepad (`launch_app`), types via `set_element_text`/`keyboard`, drives the Save dialog, and reports `done`; test Deny (model reconsiders) and Stop (session ends + input control flips off); verify it refuses to start when either consent is off.

## Program context

Sub-project 5 of 5 — the capstone; completes the desktop-agent initiative. Shipped: #1 Desktop Control, #2 Input Automation, #3 Network Scanner, #4 Plugin/Connector Hub. This sub-project adds no new low-level capability; it orchestrates the existing ones into a safe, goal-directed operator.
