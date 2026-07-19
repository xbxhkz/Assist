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
