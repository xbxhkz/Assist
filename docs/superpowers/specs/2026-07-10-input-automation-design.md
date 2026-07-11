# Input Automation — Design Spec

**Status:** Approved for planning (2026-07-10)
**Sub-project 2 of 5** in the desktop-agent initiative (see the Desktop Control spec's "Program context").

## Goal

Give the Assist agent *hands* on the user's Windows machine: **move/click/drag/scroll the mouse, type text and press hotkeys, and drive UI controls by name via Windows UI Automation** — entirely locally, behind an explicit per-session consent toggle. This completes the *eyes* delivered by Desktop Control (sub-project 1), and is the actuator layer the future **AI Operator** (sub-project 5) will drive to actually complete tasks (fill a form, change a setting) rather than only describe them.

## Scope

**In scope:** two desktop backends (raw SendInput actuators + a UI Automation backend), five agent tools (one read + four act), a new session **input-control** consent toggle (sidebar toggle + indicator, reset each launch), admin gating, plan-mode gating, and audit logging.

**Out of scope (later sub-projects / YAGNI):** the continuous Operator watch-loop and its targeting intelligence (sub-project 5 — this spec ships the raw actuators + element-targeting primitives it will call, not the loop that decides *what* to click); network/port/OS scanning (sub-project 3); macro recording/playback; OCR-based targeting; cross-platform (Windows only, consistent with Desktop Control).

## Architecture

Mirror the Desktop Control pattern: OS-specific backends under `src/desktop/` with **injectable primitives so tests never touch a real GUI**, and thin tool wrappers under `src/agent_tools/` that parse args, enforce gates, call the backend, and format the result.

- **`src/desktop/inputraw.py` (new)** — raw actuators over `ctypes.windll.user32.SendInput` (**zero third-party dependency**, exactly like `windows.py`). An injectable `send=` primitive lets tests assert the emitted `INPUT` structs without moving a real cursor:
  - `move(x, y)` — absolute move (coords normalized to the 0–65535 virtual-desktop space, `MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE`).
  - `click(x, y, button="left", double=False)` — move + button down/up (double = two pairs).
  - `drag(x1, y1, x2, y2, button="left")` — down at origin, move, up at destination.
  - `scroll(amount, x=None, y=None)` — `MOUSEEVENTF_WHEEL` (positive = up).
  - `type_text(s)` — per-character `KEYEVENTF_UNICODE` down/up (handles arbitrary text without layout dependence).
  - `press_keys(keys)` — chord/hotkey: resolve each name via a key-name→virtual-key map (`ctrl`, `alt`, `shift`, `win`, `enter`, `tab`, `esc`, `f1`–`f12`, arrows, letters/digits, …), press modifiers down, tap the final key, release in reverse.
- **`src/desktop/uia.py` (new)** — UI Automation over the `IUIAutomation` COM interface (via **`comtypes`**), wrapped so the root automation object is **injectable** (tests pass a fake tree; no real COM):
  - `list_elements(window_id | "focused", interactable_only=True) -> [{name, control_type, automation_id, bounds, patterns}]` — walk the element tree of one top-level window; default to interactable controls (buttons, edits, lists, menu items, …).
  - `find_element(root, *, name=None, automation_id=None, control_type=None, nth=0) -> element | None` — first (or nth) match.
  - `invoke(element)` — `InvokePattern.Invoke()`; fall back to a bounds-center `click` (via an injected clicker) when the element exposes no Invoke pattern.
  - `set_value(element, text)` — `ValuePattern.SetValue(text)`; fall back to focus + `type_text` when no Value pattern.
  - `get_value(element) -> str` — `ValuePattern.CurrentValue` (read).
- **`src/agent_tools/` tool wrappers (new)** — five tool classes with the standard `async execute(content, ctx) -> {output|error, exit_code}` shape, following `desktop_tools.py` (new file `input_tools.py`, or extend `desktop_tools.py` — an implementation detail settled in the plan).

**Registration** (every site the tool system requires, enforced by a guard test exactly like the `open_in_vscode` / Desktop Control precedent):
`src/agent_tools/__init__.py` (`TOOL_HANDLERS`, `TOOL_TAGS`), `src/agent_loop.py` (`TOOL_SECTIONS` prompt blocks + the `desktop` domain map), `src/tool_schemas.py` (function schemas + content marshalling), `src/tool_index.py` (one-line descriptions), `src/tool_execution.py` (direct-dispatch branch), `src/tool_security.py` (`NON_ADMIN_BLOCKED_TOOLS`; the four acting tools stay OUT of `PLAN_MODE_READONLY_TOOLS`, `list_ui_elements` goes IN).

## The five tools

| Tool | Args | Backend | Consent gate | Plan mode |
|---|---|---|---|---|
| `list_ui_elements` | `{"window_id": <int> \| "focused"}` | `uia.list_elements` → `[{name, control_type, automation_id, bounds}]` | **screen_access** (reading) | read-only → allowed |
| `click_element` | `{"window_id"?, "name"?, "automation_id"?, "control_type"?, "nth"?}` | `uia.find_element` + `uia.invoke` | **input_control** (acting) | mutator → blocked |
| `set_element_text` | `{"window_id"?, <target…>, "text": "<str>"}` | `uia.find_element` + `uia.set_value` | **input_control** | mutator → blocked |
| `mouse` | `{"action": "move\|click\|double\|right\|drag\|scroll", "x", "y", "to_x"?, "to_y"?, "amount"?}` | `inputraw.*` | **input_control** | mutator → blocked |
| `keyboard` | `{"action": "type\|hotkey", "text"?, "keys"?}` | `inputraw.type_text` / `press_keys` | **input_control** | mutator → blocked |

Cohesive surface: one read tool, two device tools (`mouse`/`keyboard`), two UIA acting tools. Each acting tool, when its gate is off, refuses with a guidance message naming the sidebar toggle (mirroring `capture_screen`).

## Permission / safety model

The chosen model mirrors Desktop Control's screen-access consent — a *session toggle*, not per-action dialogs (per-action confirmation would defeat "do it for me" automation).

- **New session setting `input_control_enabled`** (default `false`). **Reset to `false` on every app launch** by extending the existing startup reset that clears `screen_access_enabled` (`reset_screen_access` → add/parallel `reset_input_control`, called from the same app-startup hook). The acting tools read it fresh each call.
- **New sidebar toggle "Allow input control" + a persistent "input control ON" indicator**, mirroring the screen-access toggle in the same sidebar section; posted through the existing settings POST.
- **Gating split** (the elegant consistency): *reading* the desktop — `list_ui_elements` (which may include each control's current value via the `uia.get_value` helper) — sits behind **`screen_access_enabled`** because it is screen-reading via the accessibility tree, the same class of capability as `capture_screen`. *Acting* — click/type/drag/scroll/set-text/invoke — sits behind **`input_control_enabled`**.
- **Admin gating:** all five tools in `NON_ADMIN_BLOCKED_TOOLS`.
- **Plan mode:** the four acting tools are mutators, blocked in plan mode; `list_ui_elements` is read-only and stays enabled (like `list_windows` / `find_files` / `capture_screen`).
- **Auditability:** every acting call logs an INFO line (tool + target/coords) to `app.log`.
- **Prompt-level guardrail:** tool descriptions instruct the model to confirm risky or irreversible actions with the user first (consistent with `control_window` close / `bash` / file writes) — the hard guardrail is the consent toggle, admin gate, and plan-mode block, not a per-keystroke dialog.

## Dependency strategy

- **Raw input: `ctypes` SendInput — zero dependency**, exactly like `windows.py`.
- **UI Automation: `comtypes`** (generates the `IUIAutomation` client). This is the one real bundling risk and is flagged as plan-time verification #1 below. The `uia.py` wrapper isolates the dependency behind our own functions so the underlying access method can change without touching the tool contract.
- All SendInput / COM calls stay in-process; nothing spawns `sys.executable` (no fork risk — consistent with the four `sys.executable` fork fixes made earlier).

## Data flow

```
model → tool call → input tool .execute
   → gate checks (admin; acting tools check input_control_enabled;
                  list_ui_elements/read checks screen_access_enabled)
   → src/desktop/inputraw (SendInput)  OR  src/desktop/uia (IUIAutomation)
   → structured result  ──────────────► back to the model
```

## Testing strategy (TDD, injected backends — no real GUI / COM)

- **`inputraw.py`:** each function builds the correct `INPUT` structs / calls the injected `send` with expected flags — `type_text` emits one `KEYEVENTF_UNICODE` down/up pair per character; `click` emits down+up (double = two); `drag` emits down→move→up; `scroll` emits `MOUSEEVENTF_WHEEL` with the right delta sign; absolute coords normalized to 0–65535; the key-name map resolves modifiers + function/arrow keys and `press_keys` releases in reverse order.
- **`uia.py`:** `list_elements` parses a fake automation tree into the dict shape; `find_element` matches by name / automation_id / control_type and honors `nth`; `invoke` calls the fake `InvokePattern` (or the injected clicker when absent); `set_value` uses the fake `ValuePattern` (or focus+type fallback); a missing element yields a clear error.
- **Tools:** gate checks — acting tools refuse with the guidance error when `input_control_enabled` is false; `list_ui_elements` refuses when `screen_access_enabled` is false; admin gating; arg validation; each tool calls the injected backend and formats results.
- **Registration guard test** (like `open_in_vscode` / Desktop Control): the five tools appear in every required registry; the four acting tools are absent from `PLAN_MODE_READONLY_TOOLS`; `list_ui_elements` is present in it; all five are in `NON_ADMIN_BLOCKED_TOOLS`.
- **UI guard tests:** sidebar input-control toggle element + indicator present; settings JS posts `input_control_enabled`; startup resets it to `false`.
- **Live verification on the packaged exe** before ship: toggle input-control on; open Notepad; `list_ui_elements` finds the edit control; `set_element_text` / `keyboard type` enters text; `click_element` clicks a menu; `mouse` drag + scroll behave; toggle off → every acting tool refuses with the guidance message; confirm `list_ui_elements` still refuses when screen access is off.

## Security considerations

This is the user's own machine, operated by the admin user of their own local-first app; the capability is personal-productivity. Input automation is the most powerful desktop capability — hence: default-off, per-session, visibly-indicated consent that resets each launch; admin-only; blocked in plan mode; every action audit-logged; and a prompt-level instruction to confirm irreversible actions. The read/act gating split means turning on *input control* is a distinct, deliberate act separate from screen access.

## Plan-time verifications (flagged, not hidden)

1. **`comtypes` + the generated `IUIAutomation` client bundle and run in the frozen PyInstaller build** (comtypes' runtime code generation can be fragile when frozen). Verify at plan/implementation time; **fallback ladder** if it fails: (a) pin/bundle the generated UIA wrapper module (freeze the comtypes gen cache into the build); (b) a minimal **raw-ctypes** `IUIAutomation` vtable wrapper for just the ~6 calls used (`ElementFromHandle`, `FindAll`/`FindFirst`, `GetCurrentPropertyValue`, `GetCurrentPattern` for Invoke/Value); (c) worst case, ship the raw-input tools in v1 and add the UIA tools in a fast follow-up. The tool contracts above are unchanged in every case.
2. Confirm `SendInput` synthetic events land correctly on the multi-monitor / HiDPI setup (absolute-coordinate normalization uses the full virtual desktop, not a single monitor).
3. Confirm the startup reset hook clears `input_control_enabled` alongside `screen_access_enabled` in the frozen exe (boot-verify), so input control is never silently on across restarts.

## Program context

Sub-project 2 of 5. Depends on #1 (Desktop Control — shipped). Remaining after this: **Network Scanner** (#3, independent) and **AI Operator** (#5 — the continuous screen-watch loop, which consumes this sub-project's actuators + element targeting plus Desktop Control's screen reading and a local VLM). Sub-project #4 (Plugin/Connector Hub) shipped 2026-07-10.
