# Live Equipment Data (Read Foundation) — Design

**Goal:** A `read_equipment` builtin tool that reads live values on demand from a
Modbus TCP or OPC UA endpoint — two protocol adapters behind one agent tool,
read-only and safety-guarded — giving the agent live equipment data.

**Scope:** Sub-project 1 of the Industrial Assistant's live-data initiative (the
read *foundation*). Non-goals here: continuous monitoring / threshold alarms
(sub-project 2, which reuses these adapters via the scheduler), writes to any
device, OPC UA certificate security, Modbus RTU/serial, MQTT, engineering-unit
scaling, and a UI panel.

---

## Background — what this builds on

- **No native industrial protocols today** — the app has a Plugin/Connector Hub
  (`src/integrations.py`), MCP, and `api_call`, but no Modbus/OPC UA/MQTT. This adds
  the first two as an agent tool.
- **Builtin-tool pattern** — a `SomeTool().execute(content, ctx)` async handler
  registered at ~7 surfaces (same as `diagnose_equipment`/`run_workflow`).
- **The network-scanner family** (`discover_hosts`/`scan_ports`, `src/desktop/netscan.py`)
  is the security precedent: admin-gated + a shipped `_require_private` guard
  (strict `ipaddress`, RFC1918 + loopback/link-local only). `read_equipment` reuses
  that guard and matches netscan's gating.
- **Dependencies (user-accepted, vet pending):** `pymodbus` (BSD-3-Clause, permissive)
  and `asyncua` (OPC UA; **LGPL-3.0** — the user accepts this for their distribution).
  Both are pure-Python and PyPI-available on Python 3.14. **The automated Sonatype
  security vet could not run in the build session (MCP auth unavailable); the user
  will run it manually in an interactive session before release.** Both are locally
  testable (each ships a server), so this is buildable/testable headlessly — NOT a
  hardware-blocked feasibility gate like EXL2/ControlNet.

## Architecture

One agent tool over two thin adapter modules with a common shape (connect → read →
return → disconnect):

- **`src/industrial/modbus_client.py`** — wraps `pymodbus`' async TCP client:
  `async read_modbus(host, port, unit, reg_type, address, count, data_type) ->
  {"values": [...], "raw": [int, ...]}`. `reg_type` ∈ `holding|input|coil|discrete`.
- **`src/industrial/opcua_client.py`** — wraps `asyncua`:
  `async read_opcua(endpoint, node_ids, *, timeout) -> {node_id: value}`. Connects
  anonymously (`SecurityPolicy None` — fine for an admin on a private LAN).
- **`src/agent_tools/industrial_live.py`** — `ReadEquipmentTool().execute` +
  `read_equipment(content, ctx, *, modbus_read=None, opcua_read=None)`. The two
  adapter callables are **injectable** so the tool's logic is unit-testable without a
  server. A `protocol` arg selects the adapter.

**Read-only, permanently.** The adapters call only read functions; no code path
writes a register or node. Writing to a PLC can move real machinery — a permanent
safety boundary, stated in the tool description.

## Tool contract

`content` is JSON; the tool branches on `protocol`, validates that protocol's
required args (missing/invalid → `{"error": …}`), applies the security guards, then
reads. Never raises.

**Common:** `protocol` (`modbus`|`opcua`, required), `timeout` (optional, default 5 s).

**Modbus:** `host` (required), `port` (default 502), `unit` (slave id, default 1),
`reg_type` (`holding|input|coil|discrete`, default `holding`), `address` (int,
required), `count` (default 1), `data_type` (default `uint16`; one of
`uint16|int16|uint32|int32|float32` — the 32-bit types span two registers,
big-endian). Coils/discretes return booleans.

**OPC UA:** `endpoint` (required, `opc.tcp://host:port`), `nodes` (required, a list of
node-id strings, e.g. `["ns=2;i=2","ns=2;s=Temp"]`).

**Returns:**
- Modbus → `{"output": {"values": [decoded…], "raw": [registers…], "address":…, "reg_type":…}}`
- OPC UA → `{"output": {"values": {node_id: value, …}}}`
- Failure (bad args, guard rejection, connect timeout, read error) → `{"error": …}`.

Raw registers are returned alongside decoded values so nothing is hidden. Scaling to
engineering units (register ÷ 10 = Hz) is deferred to sub-project 2 (per-tag config).

## Security

- **Admin-only** — in `NON_ADMIN_BLOCKED_TOOLS`, matching the network-scanner family
  (only an admin can point the agent at a network endpoint). Plan-mode treatment
  matches netscan (resolved during planning by reading netscan's lists).
- **Private-network guard** — reuse the scanner's shipped `_require_private`: the
  Modbus `host`, or the host parsed from the OPC UA `endpoint` URL, must resolve to a
  private/loopback/link-local address, else `{"error": "read_equipment: refuses a
  non-private endpoint"}`. Keeps the agent off the public internet; industrial gear is
  on private LANs. A public opt-in is future work.
- **Read-only + minimal footprint** — one connect/read/disconnect per call, a short
  default timeout; the description notes the admin should confirm the target device
  tolerates the connection.

## Error handling

The handler never raises into the agent loop. All of: non-JSON / non-object `content`,
a non-string/absent required arg, an unknown `protocol`, a non-private endpoint, a
connect timeout or refused connection, a read error, or an adapter that raises → each
returns `{"error": …}`. Hostile/wrong-shape args (a non-str `host`, a non-list
`nodes`, a non-int `address`) are shape-guarded, not just value-checked (the lesson
from `run_workflow`/`diagnose_equipment`).

## Testing

- **Task 1 — live feasibility gate (GO/NO-GO):** install `pymodbus` + `asyncua`,
  import both on Python 3.14, and prove a real **local client↔server round-trip for
  each** (in-process `pymodbus` server with a known datastore → read via the adapter;
  in-process `asyncua` server with a known node → read). If either won't install/
  import/round-trip on 3.14, escalate before building the rest. (Expected GO — both are
  pure-Python.)
- **Adapter tests (headless, against local servers — no hardware):** Modbus — assert
  reads + each `data_type` decode (uint16/int16/uint32/int32/float32) and coil booleans;
  OPC UA — assert a node read. These are real integration tests against localhost servers.
- **Tool tests (adapters injected/mocked):** arg validation per protocol; protocol
  branching; the **private-network guard** (a public IP → error); never-raises on every
  hostile/failure path; the result shape.
- **Manual check (owed by the user):** connecting to a *real* PLC — the user's device,
  tags, and network; plus the Sonatype dep vet. The automated tests prove the protocol
  plumbing against local servers.

## Registration & bundling

- Register `read_equipment` at every builtin-tool surface (handler/tags, schema,
  `NON_ADMIN_BLOCKED_TOOLS` + netscan's plan-mode treatment, `BUILTIN_TOOL_DESCRIPTIONS`,
  `TOOL_SECTIONS` + `_DOMAIN_TOOL_MAP`), with a parity test — matching the netscan family.
- Add `pymodbus` + `asyncua` to `requirements.txt` and `Assist.spec` (`asyncua` ships
  nodeset/schema data files → `collect_all("asyncua")`); a frozen boot-check confirms
  both import in the bundle (like the `diagnose_equipment` verification).

## Non-goals (this sub-project)

- Writes to any device (permanent safety boundary).
- Continuous monitoring / threshold alarms / notifications (sub-project 2, riding the
  scheduler + reusing these adapters).
- OPC UA certificate/security policies (v1 connects anonymously).
- Modbus RTU/serial (v1 is Modbus **TCP** only); MQTT (separate protocol).
- Engineering-unit scaling config (lives with monitoring in sub-project 2).
- A UI panel (the agent-tool path is v1).
