# AI Shell Execution (PowerShell / cmd) Design

**Goal:** Let the AI run commands in **PowerShell** and **cmd** (the two shells it
currently lacks), gated by a consent toggle: read-only commands auto-run;
state-changing commands prompt the user with Approve / Deny / **Auto-approve all**.
The existing `bash` tool is left untouched.

**Scope:** Agent chat tools only (no new UI terminal panel). One sub-project of a
larger two-part request; multi-GPU is a separate spec.

---

## Background — what exists

- `src/agent_tools/subprocess_tools.py` has `BashTool` (agent tool `bash`) that runs
  `asyncio.create_subprocess_shell(cmd)` — on Windows that is **cmd.exe**, so the
  tool is mislabeled and there is **no PowerShell path**. It runs **ungated** (no
  consent, no confirmation).
- `routes/shell_routes.py` is a separate **admin-only, user-facing** exec endpoint —
  not the AI's tool surface.
- Precedents for "agent stages an action for chat-UI approval": `agent_email_confirm`
  (send_email returns `{pending, pending_id}`, approved via a route) and the **AI
  Operator's blocking confirmation bridge** (`src/operator/` + `routes/operator_routes.py`:
  pending state + an asyncio wake Event + a `/decision` route + an approval card).
- Consent-toggle precedent: `screen_access_enabled` / `input_control_enabled` settings
  drive sidebar switches and gate the desktop tools.

**Decision (reconciling "three named shells" + "keep bash as-is"):** add
**`powershell`** and **`cmd`** as new consent-gated tools; keep the existing ungated
`bash` tool unchanged as the third shell. No second/gated bash variant.

## Architecture

Four units, each testable in isolation:

```
agent calls powershell/cmd tool
        │
        ▼
src/shell_exec/gate.py  ── require_shell_consent(get_setting)  → refuse if toggle off
        │
        ▼
src/shell_exec/classify.py ── classify_command(command, shell) → "read" | "write"
        │  read → run                    │ write ─────────────┐
        ▼                                ▼                    │
src/shell_exec/runner.py            src/shell_exec/approval.py│  (per-session store +
  run_in_shell(command, shell)        await_decision(...)  ◀──┘   asyncio wake bridge)
  → {output, exit_code}                │  approve → run
                                       │  deny → {denied}
                                       │  auto_all → set session flag, run
                                       ▼
                          routes/shell_approval_routes.py  POST /api/shell/decision
                          static/js chat approval card  [Approve][Deny][Auto-approve all]
```

### 1. Consent gate — `src/shell_exec/gate.py` + setting

- New **global** setting `shell_exec_enabled` (default `False`), added to `src/settings.py`
  exactly like `screen_access_enabled` / `input_control_enabled` — including the
  **startup reset to `False`** (settings.py:354-367), so shell access is re-consented each
  launch, not persisted on.
- `require_shell_consent(get_setting)` raises `PermissionError` when off. Tool handlers
  catch it and return `{"error": "Shell execution is off — enable 'Allow shell commands'
  in the sidebar.", "exit_code": 1}` so the AI can tell the user.
- Sidebar switch "Allow shell commands" mirrors the existing consent switches.

### 2. Classifier — `src/shell_exec/classify.py`

`classify_command(command, shell) -> "read" | "write"`. **Conservative and fail-safe:**

- Returns `"read"` **only if** the command is a *single simple invocation*: it contains
  **none** of the shell-control/compose characters `| > < ; & \n` backtick `$(` `&&`
  `||`, **and** its leading command token is on the per-shell read-only allowlist.
- Everything else → `"write"` (including any pipe/redirect/chain and any unrecognized
  leading token). A read command hidden inside a pipe/chain (`Get-Content x | Remove-Item`)
  is therefore treated as `"write"` — intended.
- Allowlists (case-insensitive), curated and small:
  - PowerShell: `get-*` (verb prefix), `test-path`, `select-object`/`select-string`,
    `measure-object`, `where-object`, `resolve-path`, `write-output`, `format-*`.
  - cmd: `dir`, `type`, `where`, `echo`, `ver`, `whoami`, `hostname`, `set` (no args → print).
  - (bash is out of scope — existing tool unchanged.)
- Pure function, no I/O — trivially unit-tested.

### 3. Runner — `src/shell_exec/runner.py`

`run_in_shell(command, shell, ctx) -> dict` builds the right argv and reuses the existing
`_run_subprocess_streaming` (timeout, progress, output-cap) from `subprocess_tools.py`:

- `powershell` → `["powershell", "-NoProfile", "-NonInteractive", "-Command", command]`
  (prefer `pwsh` when present, else Windows PowerShell).
- `cmd` → `["cmd", "/c", command]`.
- Runs at `agent_cwd()` with the agent `subproc_env`, like the existing tools. Returns
  `{output, exit_code}` (same shape as `bash`).

### 4. Approval bridge — `src/shell_exec/approval.py` + route + UI

Mirrors the AI Operator's confirmation bridge, keyed per chat session (the tool `ctx`
already carries `session_id` — `src/tool_execution.py:538,560`):

- A per-session store keyed by `session_id`: `auto_approve_all: bool` and a pending map
  `{pending_id → {command, shell, wake: asyncio.Event, decision}}`.
- `await_decision(session_id, command, shell, *, timeout=300)`:
  1. If `auto_approve_all` for the session → return `"approve"` immediately.
  2. Else stage a pending entry (surfaced to the chat UI as an approval card) and
     `await wake` (bounded by timeout).
  3. On timeout → `"deny"` (fail-closed).
- `POST /api/shell/decision {session_id, pending_id, decision}` where decision ∈
  `approve | deny | auto_approve_all`. `auto_approve_all` sets the session flag **and**
  approves the current one. Route is `async def` so `Event.set()` runs on the loop thread
  (operator lesson). Admin/consent checks as per the existing agent routes.
- Chat approval card (static/js, CSP-safe like `operator.js`): shows the exact command +
  shell, buttons **[Approve] [Deny] [Auto-approve all]**. All dynamic text via
  `textContent` (no XSS).
- Session scope: the store resets when a new chat session starts or `shell_exec_enabled`
  is turned off. Reads always auto-run while the toggle is on.

### 5. Tool handlers — `src/agent_tools/shell_tools.py`

`PowerShellTool` and `CmdTool`, registered in `src/agent_tools/__init__.py` as `powershell`
and `cmd`, with schemas in `src/tool_schemas.py` + `src/tool_index.py`. Each:

```
require_shell_consent(get_setting)          # → refuse if off
kind = classify_command(command, shell)
if kind == "write":
    decision = await await_decision(session_id, command, shell)
    if decision == "deny": return {"output": "Command denied by the user.", "exit_code": 1}
return await run_in_shell(command, shell, ctx)
```

The AI's tool descriptions state: PowerShell/cmd need "Allow shell commands" on; reads run
automatically, state-changing commands wait for the user's approval.

## Error handling

- Toggle off → structured refuse (above); the AI surfaces it, doesn't retry blindly.
- Approval timeout / deny → `exit_code 1` with a clear message; the AI reports it and stops.
- Runner failure (shell missing, non-zero exit) → same `{output/exit_code}` shape as `bash`,
  with stderr folded in.

## Testing

- **classify** (`tests/test_shell_classify.py`): PS `Get-ChildItem`→read; `Remove-Item`→write;
  `Get-Content x | Remove-Item`→write (pipe); `dir`→read; `del x`→write; redirect `echo x > f`
  →write; unknown token→write; case-insensitivity.
- **gate** (`tests/test_shell_gate.py`): off→`PermissionError`; on→passes.
- **runner** (`tests/test_shell_runner.py`): argv shape per shell (inject a fake spawn);
  output/exit_code passthrough.
- **approval bridge** (`tests/test_shell_approval.py`, injected fakes/Events): read never
  stages; write stages then approve→"approve"; deny→"deny"; auto_approve_all→sets flag +
  approves, and a subsequent write auto-approves without staging; timeout→deny.
- **tool handlers** (`tests/test_shell_tools.py`): off→refuse; read→runs without staging;
  write+deny→"denied"; write+approve→runs.
- **Live-verify (frozen app):** enable the toggle; AI runs `Get-Process` (auto), then a
  state-changing PowerShell command (card appears → Approve → runs); Auto-approve-all makes
  the next state-changer run without a card; toggle off → AI refuses.

## Non-goals

- No new UI terminal panel (agent tools only).
- The existing `bash` tool is unchanged (stays ungated).
- No bash classification/gating (bash out of scope this sub-project).
- Multi-GPU (separate spec).
- No sandboxing/allowlist of *which* programs may run beyond the read/write gate — the
  toggle + per-command approval is the control.
