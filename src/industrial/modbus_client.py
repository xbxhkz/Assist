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
