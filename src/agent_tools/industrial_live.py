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
