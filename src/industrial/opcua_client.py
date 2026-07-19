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
