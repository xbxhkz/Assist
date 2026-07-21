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


def test_never_raises_on_null_byte_host():
    # embedded null in host makes socket.gethostbyname raise (before the adapter, in the REAL
    # _guard_host) -- the tool must still return an error, not propagate. chr(0) builds the null at
    # runtime so the SOURCE file stays free of a literal null byte.
    out, _ = _exec(json.dumps({"protocol": "modbus", "host": "192.168.1.5" + chr(0), "address": 1}))
    assert "error" in out


def test_never_raises_on_malformed_ipv6_endpoint():
    # a malformed IPv6 opc.tcp URL makes urlparse().hostname raise ValueError (before _guard_host);
    # the tool must catch it and return an error.
    out, _ = _exec(json.dumps({"protocol": "opcua", "endpoint": "opc.tcp://[::1",
                               "nodes": ["ns=2;i=2"]}))
    assert "error" in out
