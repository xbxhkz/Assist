# Live Equipment Data (Read Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `read_equipment` builtin tool that reads live values on demand from a Modbus TCP or OPC UA endpoint — two protocol adapters behind one read-only, admin-gated tool.

**Architecture:** Two thin async adapters (`pymodbus`, `asyncua`) each proven against a local server; a tool that branches on `protocol`, validates + shape-guards untrusted args, applies the netscan private-network guard, and calls the injected adapter. Read-only. **Feasibility already proven GO** — local client↔server round-trips pass on Python 3.14 for both protocols.

**Tech Stack:** Python 3.14, `pymodbus` 3.14 (Modbus TCP), `asyncua` 2.0.1 (OPC UA). Reuses `src/desktop/netscan._require_private`.

## Global Constraints

- One tool `read_equipment`. Handler `ReadEquipmentTool().execute(content: str, ctx: dict) -> dict` → module-level `async read_equipment(content, ctx, *, modbus_read=None, opcua_read=None)`. `content` is JSON `{"protocol": "modbus"|"opcua", …per-protocol…}`; `ctx` carries `owner`.
- **READ-ONLY, permanently** — the adapters call only read functions; no code path writes a register or node.
- **Admin-only + plan-mode-readonly + private-network guard**, matching the network-scanner family (`discover_hosts`/`scan_ports`): in `NON_ADMIN_BLOCKED_TOOLS` AND `PLAN_MODE_READONLY_TOOLS`, domain `_DOMAIN_TOOL_MAP["network"]`. Reuse `src/desktop/netscan._require_private(ip)` (raises `ValueError` unless the IP is private) — resolve the host first, then guard.
- **The handler NEVER raises** — non-JSON/non-object content, a missing/wrong-shape arg, an unknown protocol, a non-private endpoint, a resolve/connect/read failure, or an adapter that raises → each returns `{"error": …}`. Args are model/attacker JSON: **shape-guard** them (a non-str `host`, non-int `address`, non-list `nodes`), not just value-check.
- **Adapters are injectable** (`modbus_read`/`opcua_read`); the tool's unit tests inject fakes. The adapters' own tests run against **local in-process servers** (no hardware).
- **Deps:** `pymodbus`, `asyncua` — already installed; added to `requirements.txt` here and bundled in `Assist.spec` (`asyncua` ships nodeset data → `collect_all`). **Pending the user's manual Sonatype vet; asyncua is LGPL-3.0 (user-accepted).**
- pytest `--import-mode=importlib`. Stage specific files (never `git add -A`; note `installer/Output/Assist-Setup.exe` shows modified — do NOT stage it). Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Commit directly to `dev`.

---

### Task 1: Modbus TCP read adapter (+ deps)

**Files:**
- Create: `src/industrial/__init__.py` (empty), `src/industrial/modbus_client.py`
- Modify: `requirements.txt` (add `pymodbus` + `asyncua`)
- Test: `tests/test_modbus_client.py`

**Interfaces:**
- Produces: `async read_modbus(host, port=502, unit=1, reg_type="holding", address=0, count=1, data_type="uint16", timeout=5.0) -> {"values": list, "raw": list}`; `_decode(regs, data_type) -> list`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_modbus_client.py`:

```python
import asyncio
import socket
import struct

import pytest

pytest.importorskip("pymodbus")
from src.industrial.modbus_client import read_modbus, _decode


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _decode_test():
    assert _decode([10, 11, 12], "uint16") == [10, 11, 12]
    assert _decode([0xFFFF], "int16") == [-1]
    hi, lo = struct.unpack(">HH", struct.pack(">f", 42.5))
    assert _decode([hi, lo], "float32") == [pytest.approx(42.5)]
    hi, lo = struct.unpack(">HH", struct.pack(">I", 100000))
    assert _decode([hi, lo], "uint32") == [100000]


def test_decode_all_types():
    _decode_test()


async def _serve(port, hr_values):
    from pymodbus.server import StartAsyncTcpServer
    from pymodbus.datastore import (ModbusServerContext, ModbusDeviceContext,
                                    ModbusSequentialDataBlock)
    dev = ModbusDeviceContext(hr=ModbusSequentialDataBlock(1, hr_values))  # 1-based store
    ctx = ModbusServerContext(devices=dev, single=True)
    return asyncio.create_task(StartAsyncTcpServer(context=ctx, address=("127.0.0.1", port)))


def test_read_holding_uint16_and_float32_roundtrip():
    async def go():
        port = _free_port()
        # registers: [0]=10,[1]=11,[2]=12 ; a float32 42.5 at [4],[5]
        hi, lo = struct.unpack(">HH", struct.pack(">f", 42.5))
        regs = [10, 11, 12, 0, hi, lo]
        srv = await _serve(port, regs)
        await asyncio.sleep(0.6)
        try:
            r16 = await read_modbus("127.0.0.1", port=port, address=0, count=3)
            rf = await read_modbus("127.0.0.1", port=port, address=4, count=1, data_type="float32")
            return r16, rf
        finally:
            srv.cancel()
            try:
                await srv
            except BaseException:
                pass
    r16, rf = asyncio.run(go())
    assert r16["values"] == [10, 11, 12] and r16["raw"] == [10, 11, 12]
    assert rf["values"][0] == pytest.approx(42.5)


def test_connection_refused_raises():
    # nothing listening on this port -> read_modbus raises (the tool layer catches it)
    with pytest.raises(Exception):
        asyncio.run(read_modbus("127.0.0.1", port=_free_port(), address=0, count=1, timeout=1.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_modbus_client.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.industrial.modbus_client'`).

- [ ] **Step 3: Write the implementation**

Create `src/industrial/__init__.py` (empty). Create `src/industrial/modbus_client.py`:

```python
"""Thin async Modbus TCP read adapter over pymodbus (3.14). READ-ONLY — calls only
read functions, never writes. Returns {"values": decoded, "raw": registers/bits}.
Raises on connect/read failure; the tool layer catches and returns {"error": ...}."""
import struct

_REG_READERS = {
    "holding": "read_holding_registers",
    "input": "read_input_registers",
    "coil": "read_coils",
    "discrete": "read_discrete_inputs",
}
_32BIT = {"uint32", "int32", "float32"}


def _decode(regs, data_type):
    dt = (data_type or "uint16").lower()
    if dt == "int16":
        return [struct.unpack(">h", struct.pack(">H", r & 0xFFFF))[0] for r in regs]
    if dt in _32BIT:
        out = []
        for i in range(0, len(regs) - 1, 2):
            raw4 = struct.pack(">HH", regs[i] & 0xFFFF, regs[i + 1] & 0xFFFF)
            if dt == "uint32":
                out.append(struct.unpack(">I", raw4)[0])
            elif dt == "int32":
                out.append(struct.unpack(">i", raw4)[0])
            else:
                out.append(struct.unpack(">f", raw4)[0])
        return out
    return [r & 0xFFFF for r in regs]          # uint16 (and fallback)


async def read_modbus(host, port=502, unit=1, reg_type="holding", address=0, count=1,
                      data_type="uint16", timeout=5.0):
    from pymodbus.client import AsyncModbusTcpClient
    reader = _REG_READERS.get(reg_type, "read_holding_registers")
    client = AsyncModbusTcpClient(host, port=port, timeout=timeout)
    try:
        if not await client.connect():
            raise ConnectionError(f"could not connect to {host}:{port}")
        reg_count = count * 2 if data_type in _32BIT else count
        rr = await getattr(client, reader)(address, count=reg_count, device_id=unit)
        if rr.isError():
            raise IOError(f"modbus read error: {rr}")
        if reg_type in ("coil", "discrete"):
            bits = list(rr.bits)[:count]
            return {"values": [bool(b) for b in bits], "raw": [int(bool(b)) for b in bits]}
        raw = list(rr.registers)
        return {"values": _decode(raw, data_type), "raw": raw}
    finally:
        client.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_modbus_client.py --import-mode=importlib -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Add the dependencies**

Append to `requirements.txt` (keep the file's existing style):
```
pymodbus>=3.14.0
asyncua>=2.0.1
```

- [ ] **Step 6: Commit**

```bash
git add src/industrial/__init__.py src/industrial/modbus_client.py tests/test_modbus_client.py requirements.txt
git commit -m "feat(industrial): Modbus TCP read adapter (read-only) + deps

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: OPC UA read adapter

**Files:**
- Create: `src/industrial/opcua_client.py`
- Test: `tests/test_opcua_client.py`

**Interfaces:**
- Produces: `async read_opcua(endpoint, node_ids, *, timeout=5.0) -> {node_id: value}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_opcua_client.py`:

```python
import asyncio
import socket

import pytest

pytest.importorskip("asyncua")
from src.industrial.opcua_client import read_opcua


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def test_read_opcua_node_roundtrip():
    async def go():
        from asyncua import Server
        port = _free_port()
        url = f"opc.tcp://127.0.0.1:{port}/freeopcua/server/"
        server = Server()
        await server.init()
        server.set_endpoint(url)
        idx = await server.register_namespace("test")
        obj = await server.nodes.objects.add_object(idx, "Dev")
        var = await obj.add_variable(idx, "Temp", 42.5)
        node_id = var.nodeid.to_string()
        await server.start()
        try:
            return node_id, await read_opcua(url, [node_id])
        finally:
            await server.stop()
    node_id, values = asyncio.run(go())
    assert values == {node_id: 42.5}


def test_read_opcua_connection_failure_raises():
    with pytest.raises(Exception):
        asyncio.run(read_opcua(f"opc.tcp://127.0.0.1:{_free_port()}/x/", ["ns=2;i=2"], timeout=1.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_opcua_client.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.industrial.opcua_client'`).

- [ ] **Step 3: Write the implementation**

Create `src/industrial/opcua_client.py`:

```python
"""Thin async OPC UA read adapter over asyncua. READ-ONLY, anonymous connection
(SecurityPolicy None — for an admin on a private LAN). Returns {node_id: value}.
Raises on connect/read failure; the tool layer catches and returns {"error": ...}."""


async def read_opcua(endpoint, node_ids, *, timeout=5.0):
    from asyncua import Client
    client = Client(url=endpoint, timeout=timeout)
    await client.connect()
    try:
        out = {}
        for nid in node_ids:
            out[nid] = await client.get_node(nid).read_value()
        return out
    finally:
        await client.disconnect()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_opcua_client.py --import-mode=importlib -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/industrial/opcua_client.py tests/test_opcua_client.py
git commit -m "feat(industrial): OPC UA read adapter (read-only, anonymous)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: The `read_equipment` tool

**Files:**
- Create: `src/agent_tools/industrial_live.py`
- Test: `tests/test_read_equipment.py`

**Interfaces:**
- Consumes: `src.industrial.modbus_client.read_modbus`, `src.industrial.opcua_client.read_opcua` (defaults); `src.desktop.netscan._require_private`.
- Produces: `class ReadEquipmentTool` with `async execute(self, content, ctx) -> dict`; module-level `async read_equipment(content, ctx, *, modbus_read=None, opcua_read=None) -> dict`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_read_equipment.py`:

```python
import asyncio
import json

import src.agent_tools.industrial_live as il


def _run(coro):
    return asyncio.run(coro)


def _mods():
    seen = {}
    async def modbus_read(host, port=502, unit=1, reg_type="holding", address=0, count=1,
                          data_type="uint16", timeout=5.0):
        seen["modbus"] = dict(host=host, port=port, unit=unit, reg_type=reg_type,
                              address=address, count=count, data_type=data_type)
        return {"values": [42], "raw": [42]}
    async def opcua_read(endpoint, node_ids, *, timeout=5.0):
        seen["opcua"] = dict(endpoint=endpoint, node_ids=node_ids)
        return {nid: 1.0 for nid in node_ids}
    return modbus_read, opcua_read, seen


def _exec(content, ctx=None, **kw):
    m, o, seen = _mods()
    kw.setdefault("modbus_read", m); kw.setdefault("opcua_read", o)
    out = _run(il.read_equipment(content, ctx or {}, **kw))
    return out, seen


def test_modbus_happy_path_private_host(monkeypatch):
    # force the private guard to pass regardless of DNS
    monkeypatch.setattr(il, "_guard_host", lambda h: None)
    out, seen = _exec(json.dumps({"protocol": "modbus", "host": "192.168.1.50",
                                  "address": 10, "count": 1, "data_type": "uint16"}),
                      {"owner": "admin"})
    assert out["output"]["values"] == [42] and out["output"]["reg_type"] == "holding"
    assert seen["modbus"]["address"] == 10


def test_opcua_happy_path(monkeypatch):
    monkeypatch.setattr(il, "_guard_host", lambda h: None)
    out, seen = _exec(json.dumps({"protocol": "opcua",
                                  "endpoint": "opc.tcp://192.168.1.50:4840/",
                                  "nodes": ["ns=2;i=2"]}), {"owner": "admin"})
    assert out["output"]["values"] == {"ns=2;i=2": 1.0}
    assert seen["opcua"]["node_ids"] == ["ns=2;i=2"]


def test_unknown_protocol_is_error():
    out, _ = _exec(json.dumps({"protocol": "mqtt"}))
    assert "error" in out and "protocol" in out["error"]


def test_bad_json_is_error():
    out, _ = _exec("not json")
    assert "error" in out


def test_missing_and_wrong_shape_args_are_errors(monkeypatch):
    monkeypatch.setattr(il, "_guard_host", lambda h: None)
    assert "error" in _exec(json.dumps({"protocol": "modbus", "address": 1}))[0]         # no host
    assert "error" in _exec(json.dumps({"protocol": "modbus", "host": 5, "address": 1}))[0]  # non-str host
    assert "error" in _exec(json.dumps({"protocol": "modbus", "host": "x", "address": "a"}))[0]  # non-int addr
    assert "error" in _exec(json.dumps({"protocol": "opcua", "endpoint": "opc.tcp://x/"}))[0]    # no nodes
    assert "error" in _exec(json.dumps({"protocol": "opcua", "endpoint": "opc.tcp://x/",
                                        "nodes": "ns=2;i=2"}))[0]                          # nodes not a list


def test_private_guard_rejects_public_ip():
    # the REAL guard: a public IP must be refused (no monkeypatch)
    out, _ = _exec(json.dumps({"protocol": "modbus", "host": "8.8.8.8", "address": 1}))
    assert "error" in out and "private" in out["error"].lower()


def test_never_raises_when_adapter_raises(monkeypatch):
    monkeypatch.setattr(il, "_guard_host", lambda h: None)
    async def boom(*a, **k):
        raise RuntimeError("device down")
    out, _ = _exec(json.dumps({"protocol": "modbus", "host": "192.168.1.9", "address": 1}),
                   modbus_read=boom)
    assert "error" in out and "device down" in out["error"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_read_equipment.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.agent_tools.industrial_live'`).

- [ ] **Step 3: Write the implementation**

Create `src/agent_tools/industrial_live.py`:

```python
"""The `read_equipment` builtin tool: read live values on demand from a Modbus TCP
or OPC UA endpoint. READ-ONLY. Admin-gated + private-network guarded (reuses the
network-scanner's _require_private). The handler NEVER raises — every failure
returns {"error": ...}. Adapters are injectable so the tool logic is unit-testable."""
import json
import socket
from urllib.parse import urlparse

_REG_TYPES = ("holding", "input", "coil", "discrete")
_DATA_TYPES = ("uint16", "int16", "uint32", "int32", "float32")


def _guard_host(host):
    """Return an error string, or None if `host` resolves to a private address."""
    from src.desktop.netscan import _require_private
    if not isinstance(host, str) or not host:
        return "read_equipment: a host/endpoint is required"
    try:
        ip = socket.gethostbyname(host)
    except (OSError, UnicodeError) as e:
        return f"read_equipment: cannot resolve host {host!r}: {e}"
    try:
        _require_private(ip)
    except ValueError:
        return f"read_equipment: refuses a non-private endpoint ({host})"
    return None


async def _do_modbus(args, modbus_read):
    host = args.get("host")
    if not isinstance(host, str) or not host:
        return {"error": "read_equipment: modbus requires a string 'host'"}
    address = args.get("address")
    if not isinstance(address, int) or isinstance(address, bool):
        return {"error": "read_equipment: modbus requires an integer 'address'"}
    reg_type = args.get("reg_type", "holding")
    if reg_type not in _REG_TYPES:
        return {"error": f"read_equipment: reg_type must be one of {_REG_TYPES}"}
    data_type = args.get("data_type", "uint16")
    if data_type not in _DATA_TYPES:
        return {"error": f"read_equipment: data_type must be one of {_DATA_TYPES}"}
    port, unit, count = args.get("port", 502), args.get("unit", 1), args.get("count", 1)
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (port, unit, count)):
        return {"error": "read_equipment: port/unit/count must be integers"}
    guard = _guard_host(host)
    if guard:
        return {"error": guard}
    try:
        res = await modbus_read(host, port=port, unit=unit, reg_type=reg_type,
                                address=address, count=count, data_type=data_type,
                                timeout=float(args.get("timeout", 5.0)))
    except Exception as e:
        return {"error": f"read_equipment: modbus read failed: {e}"}
    return {"output": {"values": res.get("values"), "raw": res.get("raw"),
                       "address": address, "reg_type": reg_type}}


async def _do_opcua(args, opcua_read):
    endpoint = args.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint:
        return {"error": "read_equipment: opcua requires a string 'endpoint'"}
    nodes = args.get("nodes")
    if not isinstance(nodes, list) or not nodes or not all(isinstance(n, str) for n in nodes):
        return {"error": "read_equipment: opcua requires a non-empty 'nodes' list of strings"}
    host = urlparse(endpoint).hostname
    guard = _guard_host(host)
    if guard:
        return {"error": guard}
    try:
        values = await opcua_read(endpoint, nodes, timeout=float(args.get("timeout", 5.0)))
    except Exception as e:
        return {"error": f"read_equipment: opcua read failed: {e}"}
    return {"output": {"values": values}}


async def read_equipment(content, ctx, *, modbus_read=None, opcua_read=None):
    if modbus_read is None:
        from src.industrial.modbus_client import read_modbus as modbus_read
    if opcua_read is None:
        from src.industrial.opcua_client import read_opcua as opcua_read
    try:
        args = json.loads(content) if content and content.strip() else {}
    except (ValueError, TypeError):
        return {"error": "read_equipment: arguments must be valid JSON"}
    if not isinstance(args, dict):
        return {"error": "read_equipment: arguments must be a JSON object"}
    protocol = args.get("protocol")
    if protocol == "modbus":
        return await _do_modbus(args, modbus_read)
    if protocol == "opcua":
        return await _do_opcua(args, opcua_read)
    return {"error": "read_equipment: protocol must be 'modbus' or 'opcua'"}


class ReadEquipmentTool:
    async def execute(self, content, ctx):
        return await read_equipment(content, ctx)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_read_equipment.py --import-mode=importlib -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/agent_tools/industrial_live.py tests/test_read_equipment.py
git commit -m "feat(industrial): read_equipment tool (modbus/opcua, guarded, never-raises)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Register the tool + bundle the deps

**Files:**
- Modify: `src/agent_tools/__init__.py`, `src/tool_schemas.py`, `src/tool_security.py`, `src/tool_index.py`, `src/agent_loop.py`, `Assist.spec`
- Test: `tests/test_read_equipment_registration.py`

**Interfaces:**
- Consumes (Task 3): `ReadEquipmentTool` from `src.agent_tools.industrial_live`.
- Produces: `read_equipment` at every builtin-tool surface; gated exactly like `discover_hosts`/`scan_ports`; `asyncua` bundled in the frozen build.

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_read_equipment_registration.py`:

```python
def test_registered_handler_and_tag():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    from src.agent_tools.industrial_live import ReadEquipmentTool
    assert "read_equipment" in TOOL_HANDLERS
    assert "read_equipment" in TOOL_TAGS
    assert TOOL_HANDLERS["read_equipment"].__self__.__class__ is ReadEquipmentTool


def test_gating_matches_netscan():
    import src.tool_security as ts
    # a read-only network tool: admin-blocked AND plan-mode readonly (like scan_ports)
    assert "read_equipment" in ts.NON_ADMIN_BLOCKED_TOOLS
    assert "read_equipment" in ts.PLAN_MODE_READONLY_TOOLS
    assert "scan_ports" in ts.NON_ADMIN_BLOCKED_TOOLS and "scan_ports" in ts.PLAN_MODE_READONLY_TOOLS


def test_schema_index_and_agent_loop():
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    import src.tool_index as ti
    import src.agent_loop as al
    names = {s["function"]["name"] for s in FUNCTION_TOOL_SCHEMAS if s.get("type") == "function"}
    assert "read_equipment" in names
    assert "read_equipment" in ti.BUILTIN_TOOL_DESCRIPTIONS
    assert "read_equipment" in al.TOOL_SECTIONS
    assert "read_equipment" in al._DOMAIN_TOOL_MAP["network"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_read_equipment_registration.py --import-mode=importlib -q`
Expected: FAIL (not registered yet).

- [ ] **Step 3: Register at each surface**

Read each file's current structure first; add `read_equipment` following the `discover_hosts`/`scan_ports` sibling pattern.

1. **`src/agent_tools/__init__.py`:** add `from .industrial_live import ReadEquipmentTool` with the other `from .` imports; `"read_equipment": ReadEquipmentTool().execute,` in `TOOL_HANDLERS`; `"read_equipment"` in `TOOL_TAGS`.

2. **`src/tool_schemas.py`:** add to `FUNCTION_TOOL_SCHEMAS`:
```python
    {
        "type": "function",
        "function": {
            "name": "read_equipment",
            "description": "Read live values from an industrial device on the private network — Modbus TCP registers/coils or OPC UA nodes. READ-ONLY (never writes). Admin-only; private/local addresses only. Use for on-demand equipment readings (a drive's output frequency, a meter value, a PLC tag).",
            "parameters": {
                "type": "object",
                "properties": {
                    "protocol": {"type": "string", "enum": ["modbus", "opcua"], "description": "modbus (TCP) or opcua"},
                    "host": {"type": "string", "description": "Modbus: device host/IP (private network)."},
                    "port": {"type": "integer", "description": "Modbus TCP port (default 502)."},
                    "unit": {"type": "integer", "description": "Modbus unit/slave id (default 1)."},
                    "reg_type": {"type": "string", "enum": ["holding", "input", "coil", "discrete"], "description": "Modbus register type (default holding)."},
                    "address": {"type": "integer", "description": "Modbus start address."},
                    "count": {"type": "integer", "description": "Modbus number of values (default 1)."},
                    "data_type": {"type": "string", "enum": ["uint16", "int16", "uint32", "int32", "float32"], "description": "Modbus decode (default uint16; 32-bit types span 2 registers)."},
                    "endpoint": {"type": "string", "description": "OPC UA endpoint, e.g. opc.tcp://192.168.1.50:4840"},
                    "nodes": {"type": "array", "items": {"type": "string"}, "description": "OPC UA node ids to read, e.g. [\"ns=2;i=2\"]."},
                    "timeout": {"type": "number", "description": "Connection timeout seconds (default 5)."}
                },
                "required": ["protocol"]
            }
        }
    },
```

3. **`src/tool_security.py`:** add `"read_equipment"` to `NON_ADMIN_BLOCKED_TOOLS` (near `"scan_ports"`, ~line 66) AND `PLAN_MODE_READONLY_TOOLS` (near `"scan_ports"`, ~line 147).

4. **`src/tool_index.py`:** add to `BUILTIN_TOOL_DESCRIPTIONS`: `"read_equipment": "Read live values from an industrial device (Modbus TCP or OPC UA) on the private network — read-only.",`.

5. **`src/agent_loop.py`:**
   - `TOOL_SECTIONS` — add (model the `scan_ports` section, ~line 472):
     ```python
     "read_equipment": "- ```read_equipment``` — Read live values from an industrial device (READ-ONLY, private network, admin). Args (JSON): modbus → {\"protocol\":\"modbus\",\"host\":\"192.168.1.50\",\"reg_type\":\"holding\",\"address\":40001,\"count\":1,\"data_type\":\"float32\"}; opcua → {\"protocol\":\"opcua\",\"endpoint\":\"opc.tcp://192.168.1.50:4840\",\"nodes\":[\"ns=2;i=2\"]}.",
     ```
   - `_DOMAIN_TOOL_MAP["network"]` (~line 309) — add `"read_equipment"` to the `"network"` set alongside `"scan_ports"`.

- [ ] **Step 4: Bundle asyncua in the frozen build**

In `Assist.spec`, add `asyncua` to the `collect_all` loop (it ships nodeset XML/schema data files that must be bundled). Find the existing `collect_all(...)` usage and add `"asyncua"` to the list of packages it iterates (mirror how e.g. `faster_whisper`/`ctranslate2` are collected). `pymodbus` is pure code and needs no data collection (PyInstaller's import analysis picks it up), but add it to the spec's `hiddenimports` if the frozen import check (Step 6) fails to find it.

- [ ] **Step 5: Run the parity test + import smoke**

Run: `python -m pytest tests/test_read_equipment_registration.py --import-mode=importlib -q` (Expected: PASS, 3 passed) and `python -c "import app"` (no error).

- [ ] **Step 6: Run the adapter + tool suites (no regression)**

Run: `python -m pytest tests/test_modbus_client.py tests/test_opcua_client.py tests/test_read_equipment.py --import-mode=importlib -q`
Expected: PASS (12 passed).

- [ ] **Step 7: Commit**

```bash
git add src/agent_tools/__init__.py src/tool_schemas.py src/tool_security.py src/tool_index.py src/agent_loop.py Assist.spec tests/test_read_equipment_registration.py
git commit -m "feat(industrial): register read_equipment (netscan gating) + bundle asyncua

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **All tasks are TDD.** The adapter tests (T1/T2) run **real local in-process servers** — no hardware. The tool test (T3) injects fake adapters. T4 asserts registration + `import app`.
- **Feasibility is already proven GO** (both round-trips pass on Python 3.14); if an adapter test flakes on server startup timing, raise the `asyncio.sleep(0.6)` warm-up, don't change the adapter.
- **READ-ONLY is a permanent safety boundary** — no task adds a write path. The private-network guard (admin + `_require_private`) and admin-only gating are security requirements; do not weaken them.
- **Verify the real module-level names** before editing in T4 (`FUNCTION_TOOL_SCHEMAS`, `BUILTIN_TOOL_DESCRIPTIONS`, `NON_ADMIN_BLOCKED_TOOLS`, `PLAN_MODE_READONLY_TOOLS`, `TOOL_SECTIONS`, `_DOMAIN_TOOL_MAP` — all verified at plan time); if one moved, fix registration AND assertion, never weaken an assertion.
- **Owed by the user (not automated):** the Sonatype dep vet (interactive session) and connecting to a *real* PLC (their device/tags/network). A frozen boot-check (after a rebuild) confirms `pymodbus`/`asyncua` import in the bundle.
- Scope: the read foundation only. Do NOT build writes, continuous monitoring/alarms, OPC UA certs, Modbus RTU, MQTT, scaling config, or a UI panel (all sub-project 2 or explicit non-goals).
