# Plugin / Connector Hub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A first-class admin "Plugins" sidebar screen that unifies built-in MCP servers, user MCP servers, and HTTP connectors into one list with status, enable/disable, add-from-catalog, and troubleshooting.

**Architecture:** Client-side aggregation (Approach A): a plain-IIFE `static/js/plugins.js` modal fetches the existing `/api/mcp/servers` and `/api/auth/integrations` plus a new small `/api/mcp/builtins`, and merges them into one list. The only new backend is a per-built-in enable/disable — a `disabled_builtin_mcp` setting honored at boot, and list/toggle endpoints that live-apply via the MCP manager.

**Tech Stack:** FastAPI, pytest (`--import-mode=importlib`), vanilla JS (plain IIFE), the existing MCP manager + OAuth + integrations routes.

## Global Constraints

- All pytest runs use `--import-mode=importlib` (a global `ultralytics` package shadows `tests/`).
- Admin-gated: every new route uses `require_admin(request)` (MCP servers spawn local processes). Matches existing `/api/mcp/*` and `/api/auth/integrations/*`.
- Reuse, don't reimplement: MCP CRUD/OAuth (`routes/mcp_routes.py`), connectors (`routes/auth_routes.py` `/api/auth/integrations*`), and the preset catalogs (`MCP_PRESETS` in `static/js/admin.js`, `INTEGRATION_PRESETS` via `/api/auth/integrations/presets`) are consumed as-is.
- Frontend: plain-IIFE script like `static/js/help.js`/`localModels.js`; listeners via `addEventListener` (CSP forbids inline handlers); no `type="module"`.
- Built-ins: toggle on/off, never delete. User MCP servers + connectors: full CRUD.
- The existing Admin → Tools MCP tab stays functional — do not modify or remove it.
- Verified manager API (`src/mcp_manager.py`): `get_server_status(id)->dict{status,error,tool_count,...}`, `is_builtin(id)->bool`, `async disconnect_server(id)`, `async _reconnect_builtin(id)->bool`. Built-in ids/names: `_BUILTIN_SERVERS` in `src/builtin_mcp.py` = `{image_gen, memory, rag, email}`.

---

### Task 1: `disabled_builtin_mcp` setting + boot-skip

**Files:**
- Modify: `src/settings.py` (add to `DEFAULT_SETTINGS`)
- Modify: `src/builtin_mcp.py` (`register_builtin_servers` skips disabled ids)
- Test: `tests/test_builtin_toggle.py`

**Interfaces:**
- Produces: setting key `disabled_builtin_mcp` (list of built-in ids, default `[]`); `register_builtin_servers` does not register any id in that list.

- [ ] **Step 1 — failing test** `tests/test_builtin_toggle.py`:

```python
import asyncio
import src.settings as settings
import src.builtin_mcp as bm


def test_default_disabled_builtin_is_empty():
    assert settings.DEFAULT_SETTINGS.get("disabled_builtin_mcp") == []


def test_register_skips_disabled_builtins(monkeypatch):
    """A built-in listed in disabled_builtin_mcp must never connect at boot."""
    monkeypatch.setattr(bm, "MCP_DISABLED", False, raising=False)
    monkeypatch.setattr(bm, "get_setting",
                        lambda k, d=None: ["rag", "email"] if k == "disabled_builtin_mcp" else d,
                        raising=False)
    connected = []

    class FakeMgr:
        async def connect_server(self, server_id, name, transport, command, args, env):
            connected.append(server_id)
            return True

    # Neutralize the NPX-server registration path so only builtins are exercised.
    monkeypatch.setattr(bm, "_find_npx", lambda: None, raising=False)
    monkeypatch.setattr(bm, "os", bm.os)  # keep os
    asyncio.run(bm.register_builtin_servers(FakeMgr()))
    assert "rag" not in connected and "email" not in connected
    assert "memory" in connected and "image_gen" in connected
```

- [ ] **Step 2 — run, expect FAIL:** `python -m pytest tests/test_builtin_toggle.py --import-mode=importlib -q`
- [ ] **Step 3 — implement.** In `src/settings.py` `DEFAULT_SETTINGS`, next to `"screen_access_enabled": False,` add:

```python
    "disabled_builtin_mcp": [],
```

In `src/builtin_mcp.py`, add a lazy import of `get_setting` at the top of `register_builtin_servers` and skip disabled ids where the per-server loop connects them. The function has a loop over `_BUILTIN_SERVERS.items()` (each dispatched via `_spawn_bg(_connect_python_server(server_id, script_path, name))`). Guard it:

```python
    from src.settings import get_setting
    _disabled = set(get_setting("disabled_builtin_mcp", []) or [])
```

and inside the loop, before spawning:

```python
        if server_id in _disabled:
            logger.info(f"Built-in MCP server disabled by setting: {name}")
            continue
```

(If the test's monkeypatch of `bm.get_setting` requires the name to exist at module scope, add `from src.settings import get_setting` as a module-level import in `builtin_mcp.py` so `monkeypatch.setattr(bm, "get_setting", ...)` binds; then inside the function call `get_setting(...)` directly.)

- [ ] **Step 4 — run, expect PASS.** **Step 5 — commit** `feat(plugins): disabled_builtin_mcp setting + boot-skip`.

---

### Task 2: `/api/mcp/builtins` list + toggle endpoints

**Files:**
- Modify: `routes/mcp_routes.py` (inside `setup_mcp_routes(mcp_manager)`)
- Test: `tests/test_mcp_builtins_routes.py`

**Interfaces:**
- Consumes: Task 1 setting; manager `get_server_status`, `disconnect_server`, `_reconnect_builtin`.
- Produces: `GET /api/mcp/builtins` → `{"builtins": [{id, name, status, enabled, tool_count, error}]}`; `POST /api/mcp/builtins/{server_id}/toggle` body `{enabled: bool}` → `{ok, id, enabled}`.

- [ ] **Step 1 — failing test** `tests/test_mcp_builtins_routes.py`:

```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.mcp_routes as mr
from core.middleware import require_admin


class FakeMgr:
    def __init__(self):
        self.disconnected = []
        self.reconnected = []
    def get_server_status(self, sid):
        return {"status": "connected", "tool_count": 1, "error": None}
    def is_builtin(self, sid):
        return sid in {"image_gen", "memory", "rag", "email"}
    async def disconnect_server(self, sid):
        self.disconnected.append(sid)
    async def _reconnect_builtin(self, sid):
        self.reconnected.append(sid)
        return True


@pytest.fixture
def client(monkeypatch):
    fake = FakeMgr()
    # Setting store the endpoints read/write.
    store = {"disabled_builtin_mcp": []}
    monkeypatch.setattr(mr, "get_setting",
                        lambda k, d=None: store.get(k, d), raising=False)
    def _save(d):
        store.update(d)
    monkeypatch.setattr(mr, "set_setting", _save, raising=False)
    app = FastAPI()
    app.include_router(mr.setup_mcp_routes(fake))
    app.dependency_overrides = {}
    # require_admin is called inside handlers; stub it to pass.
    monkeypatch.setattr(mr, "require_admin", lambda request: None)
    return TestClient(app), fake, store


def test_list_builtins(client):
    c, _fake, _store = client
    r = c.get("/api/mcp/builtins")
    assert r.status_code == 200
    ids = {b["id"]: b for b in r.json()["builtins"]}
    assert set(ids) == {"image_gen", "memory", "rag", "email"}
    assert ids["memory"]["enabled"] is True
    assert ids["memory"]["status"] == "connected"


def test_toggle_off_disconnects_and_persists(client):
    c, fake, store = client
    r = c.post("/api/mcp/builtins/rag/toggle", json={"enabled": False})
    assert r.status_code == 200 and r.json()["enabled"] is False
    assert "rag" in store["disabled_builtin_mcp"]
    assert "rag" in fake.disconnected


def test_toggle_on_reconnects_and_persists(client):
    c, fake, store = client
    store["disabled_builtin_mcp"] = ["rag"]
    r = c.post("/api/mcp/builtins/rag/toggle", json={"enabled": True})
    assert r.status_code == 200 and r.json()["enabled"] is True
    assert "rag" not in store["disabled_builtin_mcp"]
    assert "rag" in fake.reconnected


def test_toggle_rejects_unknown_id(client):
    c, *_ = client
    assert c.post("/api/mcp/builtins/nope/toggle", json={"enabled": False}).status_code == 400
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** inside `setup_mcp_routes(mcp_manager)` in `routes/mcp_routes.py` (add near the other `@router` handlers). Add imports at top of the file: `from src.settings import get_setting, save_settings, load_settings` and `from src.builtin_mcp import _BUILTIN_SERVERS`. Provide a `set_setting` helper the test patches — define a module-level:

```python
def set_setting(update: dict):
    s = load_settings()
    s.update(update)
    save_settings(s)
```

Then the routes:

```python
    @router.get("/builtins")
    def list_builtins(request: Request):
        require_admin(request)
        disabled = set(get_setting("disabled_builtin_mcp", []) or [])
        out = []
        for sid, (_script, name) in _BUILTIN_SERVERS.items():
            st = mcp_manager.get_server_status(sid)
            out.append({
                "id": sid, "name": name,
                "status": "disabled" if sid in disabled else st.get("status", "disconnected"),
                "enabled": sid not in disabled,
                "tool_count": st.get("tool_count", 0),
                "error": st.get("error"),
            })
        return {"builtins": out}

    @router.post("/builtins/{server_id}/toggle")
    async def toggle_builtin(server_id: str, request: Request, body: dict = Body(...)):
        require_admin(request)
        if server_id not in _BUILTIN_SERVERS:
            raise HTTPException(status_code=400, detail="unknown built-in server")
        enabled = bool(body.get("enabled"))
        disabled = set(get_setting("disabled_builtin_mcp", []) or [])
        if enabled:
            disabled.discard(server_id)
            set_setting({"disabled_builtin_mcp": sorted(disabled)})
            try:
                await mcp_manager._reconnect_builtin(server_id)
            except Exception as e:
                logger.warning(f"builtin reconnect failed for {server_id}: {e}")
        else:
            disabled.add(server_id)
            set_setting({"disabled_builtin_mcp": sorted(disabled)})
            try:
                await mcp_manager.disconnect_server(server_id)
            except Exception as e:
                logger.warning(f"builtin disconnect failed for {server_id}: {e}")
        return {"ok": True, "id": server_id, "enabled": enabled}
```

Ensure `Body` and `HTTPException` are imported (the file already imports from `fastapi`). Ensure `logger` exists in the module (it does — used elsewhere).

- [ ] **Step 4 — run, expect PASS.** Also `python -c "import routes.mcp_routes"`.
- [ ] **Step 5 — commit** `feat(plugins): /api/mcp/builtins list + toggle endpoints`.

---

### Task 3: Sidebar "Plugins" entry + modal + `plugins.js` aggregation/render

**Files:**
- Modify: `static/index.html` (sidebar entry near `tool-help-btn`; a `plugins-modal`; script include)
- Create: `static/js/plugins.js`
- Test: `tests/test_plugins_ui.py`

**Interfaces:**
- Consumes: `GET /api/mcp/servers`, `GET /api/mcp/builtins`, `GET /api/auth/integrations`.
- Produces: a rendered unified list (`plugins-list`) with type badges (Built-in / MCP / Connector), status, and per-row action buttons.

- [ ] **Step 1 — failing UI guard** `tests/test_plugins_ui.py`:

```python
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
def _read(p): return (ROOT / p).read_text(encoding="utf-8")

def test_index_has_plugins_entry_and_modal():
    html = _read("static/index.html")
    for el in ('id="tool-plugins-btn"', 'id="plugins-modal"', 'id="plugins-list"'):
        assert el in html, f"{el} missing from index.html"
    assert 'src="/static/js/plugins.js"' in html

def test_plugins_js_fetches_all_three_sources():
    js = _read("static/js/plugins.js")
    for ep in ("/api/mcp/servers", "/api/mcp/builtins", "/api/auth/integrations"):
        assert ep in js, f"{ep} not fetched in plugins.js"
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement.** In `static/index.html`: add a sidebar item mirroring the Help entry (search `id="tool-help-btn"`), e.g. below it:

```html
        <div class="list-item" id="tool-plugins-btn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;opacity:0.5;"><path d="M6 3v6M6 15v6M18 3v6M18 15v6M6 9h12v6H6z"/></svg>
          <span class="grow">Plugins</span>
        </div>
```

Add a modal shell before the Settings modal (mirror the Help modal structure):

```html
  <div id="plugins-modal" class="modal hidden">
    <div class="modal-content admin-modal-content" role="dialog" aria-label="Plugins">
      <div class="modal-header">
        <h4>Plugins &amp; Connectors</h4>
        <button class="close-btn" id="close-plugins-modal" aria-label="Close">&#x2716;</button>
      </div>
      <div style="display:flex;gap:6px;margin-bottom:8px;">
        <button id="plugins-add-btn">Add…</button>
        <button id="plugins-refresh-btn">Refresh</button>
      </div>
      <div id="plugins-list"></div>
      <div id="plugins-msg" style="font-size:12px;opacity:0.8;margin-top:6px;"></div>
      <div style="font-size:11px;opacity:0.5;margin-top:8px;">Troubleshooting: data/logs/mcp-servers.log · data/logs/app.log</div>
    </div>
  </div>
```

Add `<script src="/static/js/plugins.js"></script>` beside the other includes. Create `static/js/plugins.js` (plain IIFE mirroring `help.js`): on `tool-plugins-btn` click, open the modal and `refresh()`. `refresh()` does `Promise.all` of the three fetches (`/api/mcp/builtins`, `/api/mcp/servers`, `/api/auth/integrations`), normalizes each into `{id, name, type, status, tools, error}` (type = `builtin` / `mcp` / `connector`), and renders rows into `plugins-list` — each row: a type badge, name, status dot, tool count (mcp/builtin), and a placeholder actions container (filled in Task 4). Include `credentials: 'same-origin'` on all fetches. Wire `close-plugins-modal` + backdrop click to close, and `plugins-refresh-btn` to `refresh()`.

- [ ] **Step 4 — run, expect PASS.** **Step 5 — commit** `feat(plugins): sidebar entry + modal + unified list render`.

---

### Task 4: Per-row actions + Add-from-catalog

**Files:**
- Modify: `static/js/plugins.js`
- Test: `tests/test_plugins_ui.py` (extend)

**Interfaces:**
- Consumes: existing endpoints — MCP `POST /api/mcp/servers/{id}/reconnect`, `DELETE /api/mcp/servers/{id}`, `GET /api/mcp/servers/{id}/tools`, OAuth authorize (`/api/mcp/oauth/authorize/{id}`), `POST /api/mcp/servers` (add); built-in `POST /api/mcp/builtins/{id}/toggle`; connectors `POST /api/auth/integrations/{id}/test`, `DELETE /api/auth/integrations/{id}`, `POST /api/auth/integrations` (add), `GET /api/auth/integrations/presets`.

- [ ] **Step 1 — extend the guard test** in `tests/test_plugins_ui.py`:

```python
def test_plugins_js_wires_actions():
    js = _read("static/js/plugins.js")
    for ref in ("/reconnect", "/api/mcp/builtins/", "/toggle",
                "/api/auth/integrations/", "/test", "/api/mcp/servers/",
                "/api/auth/integrations/presets"):
        assert ref in js, f"action {ref} not wired in plugins.js"
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** the per-row action buttons in `plugins.js` `renderRow`, dispatched by `type`:
  - `builtin`: an enable/disable toggle → `POST /api/mcp/builtins/${id}/toggle {enabled}` then `refresh()`.
  - `mcp`: **Reconnect** → `POST /api/mcp/servers/${id}/reconnect`; **Tools** → `GET /api/mcp/servers/${id}/tools` (show count/names in `plugins-msg`); **Authorize** (only when the row's `needs_oauth`) → open `/api/mcp/oauth/authorize/${id}`; **Delete** (confirm) → `DELETE /api/mcp/servers/${id}`; each followed by `refresh()`.
  - `connector`: **Test** → `POST /api/auth/integrations/${id}/test` (show result in `plugins-msg`); **Delete** (confirm) → `DELETE /api/auth/integrations/${id}`; then `refresh()`.
  Add the **Add…** flow (`plugins-add-btn`): a small picker with four choices — MCP preset (read the `MCP_PRESETS` array already defined in `static/js/admin.js`; if not importable, fetch is not needed — reference the same preset shape and post to `/api/mcp/servers`), custom MCP, connector preset (`GET /api/auth/integrations/presets` → prefilled → `POST /api/auth/integrations`), custom connector. For v1, "Add" may open the existing add forms by delegating to the admin flow if reachable; otherwise implement a minimal form that POSTs the required fields. All actions surface errors into `plugins-msg` rather than silently failing.
- [ ] **Step 4 — run, expect PASS.** **Step 5 — commit** `feat(plugins): per-row actions + add-from-catalog`.

---

### Task 5: Package + live-verify

- [ ] **Step 1 — full affected suite:** `python -m pytest tests/test_builtin_toggle.py tests/test_mcp_builtins_routes.py tests/test_plugins_ui.py --import-mode=importlib -q` → all green.
- [ ] **Step 2 — build:** `.\build-installer.ps1` (full clean — `-Fast` has dropped bundled deps before). Confirm compile succeeds.
- [ ] **Step 3 — boot-verify** the packaged exe against an isolated `ODYSSEUS_DATA_DIR` + `ODYSSEUS_INTERNAL_TOKEN`: `GET /api/mcp/builtins` returns the four built-ins with status; `GET /api/mcp/servers` and `GET /api/auth/integrations` respond.
- [ ] **Step 4 — user manual test:** open **Plugins** from the sidebar — see the four built-ins (with status), any user MCP servers, and any connectors in one list. Toggle **RAG** off → confirm it shows disabled and its tools drop; toggle on → reconnects. Add an MCP preset (e.g. a simple one) and a connector preset; delete them. Confirm the Admin → Tools MCP tab still works unchanged.
- [ ] **Step 5 — commit** the installer: `git add -f installer/Output/Assist-Setup.exe && git commit -m "build: Assist-Setup.exe with Plugins hub"`.

## Self-Review

- **Spec coverage:** unified sidebar screen (T3), client-side aggregation of the three sources (T3), per-type actions (T4), built-in enable/disable setting + boot-skip (T1) + list/toggle endpoints (T2), add-from-catalog reusing presets/OAuth (T4), troubleshooting log pointer (T3 modal), admin tab untouched (global constraint), package + live-verify (T5). All spec sections covered.
- **Placeholders:** none. The four spec plan-time verifications are resolved in the Global Constraints (built-ins absent from `/api/mcp/servers` → new `/api/mcp/builtins`; manager methods `disconnect_server`/`_reconnect_builtin`/`get_server_status`/`is_builtin` confirmed; connector routes confirmed; sidebar/CSP conventions from `help.js`). T4's "Add" allows a minimal-form fallback but specifies the exact endpoints and preset sources — not a TODO.
- **Type consistency:** `disabled_builtin_mcp` (list) used identically in T1/T2; normalized row `{id,name,type,status,tools,error}` consistent T3→T4; endpoint paths match the verified backend (`/api/mcp/builtins`, `/api/mcp/builtins/{id}/toggle`, `/api/mcp/servers/{id}/reconnect`, `/api/auth/integrations/{id}/test`). `set_setting`/`get_setting` names consistent in T2 and its test.
