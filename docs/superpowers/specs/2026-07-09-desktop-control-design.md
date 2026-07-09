# Desktop Control — Design Spec

**Status:** Approved for planning (2026-07-09)
**Sub-project 1 of 5** in the desktop-agent initiative (see "Program context").

## Goal

Give the Assist agent the ability to **launch applications, search the whole PC for files, list and control windows, and capture the screen** on the user's Windows machine — entirely locally. This is the foundation the later "AI Operator" mode builds on: it gives the agent *eyes* (screen capture → the existing vision pipeline) and non-input desktop awareness. It deliberately does **not** move the mouse or type — input automation is a separate sub-project.

## Program context (the 5 sub-projects)

The user requested a broad desktop-agent capability. It decomposes into five independently-shippable sub-projects, each with its own spec → plan → implementation cycle:

1. **Desktop Control** ← *this spec*: launch apps, global file search, window list/control, screen capture.
2. **Input Automation**: mouse/keyboard (`SendInput`) with confirmation guardrails. Depends on #1.
3. **Network Scanner**: LAN device discovery, port scan, OS fingerprint (the user's own network). Independent.
4. **Plugin / Connector UI**: a first-class screen over the existing MCP subsystem. Mostly UX.
5. **AI Operator mode**: permissioned continuous screen-watch loop + a local vision-language model + observe→suggest→act-with-confirmation. Depends on #1, #2, and a local VLM.

Already-existing capabilities that the original request named and that are therefore **not** in any sub-project: task **scheduling** (`src/task_scheduler.py`, Tasks panel, `manage_tasks`), and the **plugin engine** itself (MCP — sub-project 4 only adds UI over it).

## Scope

**In scope (this spec):** five read-or-launch-only desktop tools + a session screen-access consent toggle.

**Out of scope (explicitly, later sub-projects):** mouse/keyboard input; network/port/OS scanning; the plugin/connector UI; the continuous Operator loop and vision-language model sourcing. `capture_screen` produces an image for the *existing* vision path; it does not add a new model.

## Architecture

Mirror the pattern used for the recently-shipped `open_in_vscode` tool.

- **`src/desktop/` (new package)** — OS-specific backends, each a small pure-ish module with injectable primitives so tests never touch a real GUI:
  - `apps.py` — `resolve_app(name, *, start_menu_index=..., which=..., registry=...) -> AppTarget|None`; enumerate Start Menu `.lnk` shortcuts (system + user Start Menu dirs), App Paths registry (`HKLM/HKCU\...\App Paths`), `PATH`, and UWP apps (`shell:AppsFolder`). `launch(target)`.
  - `windows.py` — `list_windows(enum=...) -> list[WindowInfo]` and `control_window(window, action, user32=...)` over `ctypes.windll.user32` (EnumWindows, GetWindowText, GetWindowThreadProcessId, ShowWindow, SetForegroundWindow). No third-party dependency.
  - `capture.py` — `capture(region, grabber=...) -> bytes` (PNG) via `mss`; `region` ∈ {full, monitor:N, window:id}. Returns PNG bytes; the tool layer base64-encodes into a data URI.
  - `filesearch.py` — `search(query, *, roots, ext=None, max_results=200, searcher=..., walker=...) -> list[FileHit]`. Primary path queries the Windows Search index; on any failure (indexing disabled, provider missing) falls back to a bounded `os.walk` over `roots`. Sensitive basenames/patterns (reuse `src.tool_execution._SENSITIVE_*`) are filtered from results.
- **`src/agent_tools/desktop_tools.py` (new)** — five tool classes (`LaunchAppTool`, `FindFilesTool`, `ListWindowsTool`, `ControlWindowTool`, `CaptureScreenTool`), each `.execute(content, ctx)` returning the standard `{output|error, exit_code}` shape. Thin wrappers: parse args, enforce gates, call the backend, format the result.

**Registration** (every site the tool system requires, per the `open_in_vscode` precedent — a guard test enforces the set):
`src/agent_tools/__init__.py` (`TOOL_HANDLERS`, `TOOL_TAGS`), `src/agent_loop.py` (`TOOL_SECTIONS` prompt blocks + the `files`/new `desktop` domain map), `src/tool_schemas.py` (function schema + content marshalling), `src/tool_index.py` (one-line descriptions), `src/tool_execution.py` (direct-dispatch branch), `src/tool_security.py` (`NON_ADMIN_BLOCKED_TOOLS`; none in `PLAN_MODE_READONLY_TOOLS` except `find_files`/`list_windows`/`capture_screen` which are read-only and may stay enabled in plan mode).

## The five tools

1. **`launch_app`** — `{"name": "<app or file/url>"}`. Resolves via `resolve_app`; if `name` is an existing path or URL, opens it with the default handler (`os.startfile`). Returns what was launched (resolved target path). Admin-gated. Launch is detached, `CREATE_NO_WINDOW`, never waits.
2. **`find_files`** — `{"query": "<name/glob>", "ext": "py", "all_drives": false, "max_results": 200}`. Index-first, scoped-walk fallback. Default roots: the user profile + any Folder-access grants (`tool_path_extra_roots`); `all_drives:true` widens to fixed drives. Returns `[{path, size, modified}]`, sensitive paths filtered. Read-only; opening a hit still goes through the confined `read_file`.
3. **`list_windows`** — `{}`. Returns visible top-level windows `[{id, title, process, pid, state}]`.
4. **`control_window`** — `{"id": <int> | "title": "<match>", "action": "focus|minimize|maximize|restore|close"}`. `close` is mildly destructive: the tool description instructs the model to confirm with the user first; the tool itself still performs the requested action (consistent with how `bash`/file writes are handled — the guardrail is prompt-level, not a second dialog).
5. **`capture_screen`** — `{"target": "full|monitor:1|window:<id>"}`. **Gated by the session screen toggle.** When off: `{"error": "Screen access is off. Ask the user to enable 'Allow screen access' in the sidebar.", "exit_code": 1}`. When on: capture → PNG → base64 data URI → returned as an `image_url` content block so the existing vision pipeline (`src/llm_core.py`, which already converts `image_url` data URIs for Ollama/Anthropic/OpenAI) feeds it to the vision model. Requires `vision_enabled` + a configured vision model; if none, returns a clear "no vision model configured" error naming the setting.

## Permission model

- **Admin gating:** all five tools added to `NON_ADMIN_BLOCKED_TOOLS`, exactly like the existing file/shell tools.
- **Screen-access session toggle (new):** setting `screen_access_enabled` (default `false`), exposed via the existing settings POST and a **sidebar toggle + persistent "screen access on" indicator**. It is a *session* switch: reset to `false` on every app launch (a startup hook clears it), so screen capture is never silently available across restarts. `CaptureScreenTool` reads it fresh each call.
- **Contents stay confined:** `find_files` returns only metadata (name/path/size/date). Reading a file's contents still routes through `_resolve_tool_path` confinement (workspace / allowlist / Folder-access grants). Sensitive directories and key-file patterns are filtered from search results even when their parent is granted.
- **Auditability:** every `launch_app`, `control_window`, and `capture_screen` call logs an INFO line (tool + target) to `app.log`.

## Dependency strategy

**Minimal-deps hybrid** (chosen over a pywin32 backbone to avoid heavy frozen-build coupling, right after resolving four `sys.executable` fork bugs):

- Window list/control: `ctypes.windll.user32` — **zero dependency**.
- Screen capture: **`mss`** — small, pure-ish, PyInstaller-friendly. Verified to bundle at plan time.
- App launch: stdlib (`os.startfile`, `subprocess`), `winreg`, Start-Menu `.lnk` parsing (stdlib `struct`/`win32`-free shortcut target read, or a tiny parser).
- File search: a thin Windows Search query via the ADO/`Search.CollatorDSO` OLE DB provider. **Exact library (`adodbapi` vs. a minimal `comtypes`/`win32com` call) is pinned and bundle-verified during planning.** The bounded-walk fallback means the feature works even if the index path fails entirely — so the dependency is non-critical.
- All subprocess/COM calls use `CREATE_NO_WINDOW`; nothing spawns `sys.executable` (no fork risk).

If the thin Windows-Search query proves fragile in the frozen build at implementation time, fall back to walk-only for v1 and revisit indexing later — the tool contract (`find_files`) is unchanged either way.

## Data flow

```
model → tool call → desktop_tools.<Tool>.execute
   → gate checks (admin; capture also checks screen_access_enabled)
   → src/desktop/<backend>  (ctypes / mss / winreg / search)
   → structured result  ──────────────► back to the model
   capture_screen: PNG → base64 data URI → image_url block → existing vision pipeline
```

## Testing strategy (TDD, injected backends — no real GUI/registry)

- `apps.py`: `resolve_app` picks the right target across a fake Start-Menu map / registry / PATH; unknown name → None; a real path/URL → default-handler launch.
- `filesearch.py`: index path used when the fake searcher returns hits; walk fallback when it raises; sensitive paths filtered; `all_drives` widens roots; `max_results` honored.
- `windows.py`: `list_windows` parses a fake `EnumWindows` callback set; `control_window` dispatches the correct `ShowWindow`/`SetForegroundWindow` for each action against a fake `user32`.
- `capture.py` / `CaptureScreenTool`: refuses with the guidance error when `screen_access_enabled` is false; calls the fake grabber and emits a data-URI `image_url` block when true; clear error when no vision model configured.
- Registration guard test (like `open_in_vscode`): the five tools appear in every required registry, and none is missing from `NON_ADMIN_BLOCKED_TOOLS`.
- UI guard tests: sidebar toggle element + indicator present; settings JS posts `screen_access_enabled`; startup resets it to false.
- **Live verification on the packaged exe** before ship: launch Notepad by name, `find_files` a known file, list/focus a window, and (toggle on) capture the screen into a chat and confirm the vision model describes it.

## Security considerations

This is the user's own machine, operated by the admin user of their own local-first app; the capability is personal-productivity/defensive. Guardrails: admin-only tools; screen capture behind an explicit, per-session, visibly-indicated consent toggle that defaults off; file *contents* never broadened beyond the existing confinement; sensitive dirs/key files filtered from search; full audit logging. Input automation (the tool that could act destructively) is intentionally deferred to its own sub-project with its own guardrails.

## Plan-time verifications (flagged, not hidden)

1. Pin and bundle-verify the Windows Search query library; confirm the walk fallback triggers cleanly when indexing is disabled.
2. Confirm `mss` bundles in the PyInstaller build and captures on the multi-monitor/HiDPI setup.
3. Confirm `.lnk` target resolution without pywin32 (or accept a tiny dependency), and UWP `shell:AppsFolder` launch.
4. Confirm the `image_url` data-URI block from `capture_screen` flows through `src/llm_core.py` to the served/served-or-cloud vision model unchanged.
