# Plugin / Connector Hub — Design Spec

**Status:** Approved for planning (2026-07-10)
**Sub-project 4 of 5** in the desktop-agent initiative (see the Desktop Control spec's program context).

## Goal

Give Assist a **first-class, discoverable "Plugins" screen** that unifies its three existing extension mechanisms — built-in MCP servers, user-added MCP servers, and HTTP connectors — into one list where the admin can see status, enable/disable, add from a catalog, and troubleshoot. It elevates the currently-buried Admin → Tools MCP management into a sidebar destination and surfaces the built-in servers (Memory, RAG, Email, Image Generation) for the first time.

This is a **UX-consolidation project over existing backends**, not a new engine. The MCP CRUD + OAuth path (recently hardened during the fork-bomb/stderr fixes) and the HTTP-connector routes are reused untouched; the only new server code is a small per-built-in enable/disable.

## Scope

**In scope:** one admin-only Plugins screen (new sidebar entry) that aggregates the three extension types client-side; per-type actions over existing endpoints; add-from-catalog reusing existing presets; a new built-in enable/disable capability; inline troubleshooting.

**Out of scope (YAGNI / later):** a unified server-side `/api/plugins` endpoint (rejected — Approach B); merging the three mechanisms into one backend registry (Approach C); removing the existing Admin → Tools MCP tab (left functional to keep risk low); a remote plugin marketplace beyond the presets already in the codebase.

## The three extension mechanisms (current state)

1. **MCP servers** — external `npx`/remote processes. Backend `routes/mcp_routes.py` under `/api/mcp`: `GET /servers`, `POST /servers`, `POST /servers/{id}/reconnect`, `DELETE /servers/{id}`, `GET /tools`, `GET /servers/{id}/tools`, and OAuth `authorize/callback/exchange`. Catalog: `MCP_PRESETS` in `static/js/admin.js` (Gmail, Notion, Linear, Playwright, Todoist, Calendar, …). Currently managed in the **Admin panel → Tools tab** (admin-only).
2. **Built-in MCP servers** — `_BUILTIN_SERVERS` in `src/builtin_mcp.py`: `image_gen`, `memory`, `rag`, `email`. Auto-registered at boot through the same `mcp_manager` (so they appear in the MCP manager's server list). Today only a global `ODYSSEUS_DISABLE_MCP` env switch exists — no per-server toggle.
3. **HTTP connectors (integrations)** — `src/integrations.py` (`INTEGRATION_PRESETS`: Miniflux, Gitea, …), used via the `api_call` tool. Backend routes in `routes/auth_routes.py`: list / update / delete / test integration.

## Architecture (Approach A — client-side aggregation)

- **New sidebar entry "Plugins"** (admin-only), opening a full modal like the Local Models screen. Plain-IIFE JS module `static/js/plugins.js` (mirrors `help.js` / `localModels.js`), included via a `<script>` tag.
- The page fetches **two existing endpoints** and merges them in the browser into one normalized list:
  - `GET /api/mcp/servers` → MCP servers **including built-ins** (built-in rows identified by `id ∈ _BUILTIN_SERVERS` / the `Built-in:` name prefix).
  - `GET /api/integrations` → HTTP connectors.
- **Normalized row:** `{ id, name, type: "builtin"|"mcp"|"connector", status: "connected"|"error"|"disabled", tools?: number, error?: string }`.
- **Plan-time verification #1:** confirm `GET /api/mcp/servers` actually returns the built-in servers with a usable status. If it does not, add a tiny `GET /api/mcp/builtins` that lists `_BUILTIN_SERVERS` with each one's live connection status from `mcp_manager`. Either way the client sees built-ins with status.

## Per-row actions (by type)

- **Built-in** (`type: "builtin"`): status + **enable/disable toggle** only. No edit/delete (it's part of the app).
- **MCP server** (`type: "mcp"`): status + tool count; **Reconnect** (`POST /servers/{id}/reconnect`), **View tools** (`GET /servers/{id}/tools`), **Authorize** (OAuth flow, when the server declares OAuth), **Edit** (re-`POST /servers`), **Delete** (`DELETE /servers/{id}`).
- **Connector** (`type: "connector"`): status; **Test** (existing test route), **Edit** (update route), **Delete** (delete route).

## Backend additions (the only new server code)

1. **`disabled_builtin_mcp` setting** — a list of built-in server ids (default `[]`) in `DEFAULT_SETTINGS`. `register_builtin_servers` (`src/builtin_mcp.py`) skips any id in this list at boot, so a disabled built-in never registers. Injected/tested with the existing settings helpers.
2. **Toggle endpoint** — `POST /api/mcp/builtins/{server_id}/toggle {enabled: bool}` (admin-guarded): updates `disabled_builtin_mcp`, and live-applies by disconnecting (`mcp_manager.disconnect_server`) or reconnecting (`mcp_manager.reconnect` — the frozen-safe path fixed earlier) the built-in, so the toggle takes effect without a restart. Returns the new state.

Everything else (MCP add/reconnect/delete/tools/OAuth, connector list/update/delete/test) is an existing endpoint the UI calls directly.

## Add-from-catalog

An **Add** button opens a picker with four paths:
- **MCP preset** — from `MCP_PRESETS` (Gmail, Notion, …) → prefilled add form → `POST /servers` → OAuth authorize if the preset declares it.
- **Custom MCP** — command/args/env form → `POST /servers`.
- **Connector preset** — from `INTEGRATION_PRESETS` (Miniflux, Gitea, …) → prefilled → existing integration-create path.
- **Custom connector** — base URL + auth header form → integration-create.

The picker reuses the existing preset data and add/OAuth flows verbatim; it does not reimplement them.

## Error handling / troubleshooting

Each row surfaces its `status` and, on error, the backend `error` string, with **Reconnect** (MCP/built-in) or **Test** (connector) inline. A footer note points to `data/logs/mcp-servers.log` (the built-in MCP server log added during the fork-bomb fix) and `data/logs/app.log`. A failed toggle/reconnect shows the error rather than silently no-oping.

## Testing strategy (TDD)

- **Built-in toggle backend** (injected fakes, no real processes): `disabled_builtin_mcp` defaults `[]`; `register_builtin_servers` skips disabled ids (fake manager records which servers it connected); the toggle endpoint updates the setting and calls disconnect/reconnect on the (fake) manager; re-enabling reconnects.
- **Aggregation/normalization** (if any is extracted to a testable JS-independent helper, otherwise covered by UI guards): built-in rows are flagged from `/api/mcp/servers` by id/prefix; connector rows from `/api/integrations`.
- **UI guard tests** (text-guard, mirroring `test_localmodels_ui.py` / `test_vision_settings_ui.py`): sidebar entry present (`id="tool-plugins-btn"`), modal + list elements (`plugins-modal`, `plugins-list`), and the two fetch calls (`/api/mcp/servers`, `/api/integrations`) + the toggle call present in `plugins.js`.
- **Live verification** on the packaged exe: open Plugins, see built-ins + any MCP servers + connectors with status; toggle a built-in (e.g. RAG) off and confirm it disconnects and its tools disappear, then on; add an MCP preset and a connector preset.

## Plan-time verifications (flagged, not hidden)

1. Confirm `GET /api/mcp/servers` includes the built-in servers with status; if not, add `GET /api/mcp/builtins`.
2. Confirm `mcp_manager` exposes a `disconnect_server` (or equivalent) for the live-disable path; if only reconnect exists, the toggle disables by removing from the setting + a disconnect helper (add a minimal one if missing).
3. Confirm the connectors list/test/update/delete route paths and shapes in `routes/auth_routes.py` for the exact client calls.
4. Confirm the sidebar insertion point and modal/CSP conventions (plain-IIFE script, `addEventListener`, no inline handlers) against `help.js` / `localModels.js`.

## Program context

Sub-project 4 of 5. Remaining after this: **AI Operator mode** (continuous screen-watch loop, builds on Desktop Control's screen reading), **Input automation** (mouse/keyboard), and **Network/device scanning**. Each gets its own spec → plan → implementation cycle.
