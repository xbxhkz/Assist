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
