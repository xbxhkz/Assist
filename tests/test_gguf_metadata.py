import struct

from src.gguf_meta import read_gguf_metadata

_STRING_T, _UINT32_T, _UINT64_T = 8, 4, 10


def _kv_string(key: bytes, val: bytes) -> bytes:
    return (struct.pack("<Q", len(key)) + key
            + struct.pack("<I", _STRING_T)
            + struct.pack("<Q", len(val)) + val)


def _kv_uint32(key: bytes, n: int) -> bytes:
    return struct.pack("<Q", len(key)) + key + struct.pack("<I", _UINT32_T) + struct.pack("<I", n)


def _kv_uint64(key: bytes, n: int) -> bytes:
    return struct.pack("<Q", len(key)) + key + struct.pack("<I", _UINT64_T) + struct.pack("<Q", n)


def _build_gguf(entries: list) -> bytes:
    body = b"".join(entries)
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(entries)) + body


def test_reads_arch_and_numeric_fields(tmp_path):
    entries = [
        _kv_string(b"general.architecture", b"llama"),
        _kv_uint32(b"llama.context_length", 8192),
        _kv_uint32(b"llama.block_count", 32),
        _kv_uint32(b"llama.attention.head_count", 32),
        _kv_uint32(b"llama.attention.head_count_kv", 8),
        _kv_uint64(b"llama.embedding_length", 4096),
    ]
    p = tmp_path / "m.gguf"
    p.write_bytes(_build_gguf(entries))
    meta = read_gguf_metadata(str(p))
    assert meta == {
        "architecture": "llama", "context_length": 8192, "block_count": 32,
        "head_count": 32, "head_count_kv": 8, "embedding_length": 4096,
    }


def test_arch_after_numeric_keys_still_resolves(tmp_path):
    # order-independence: numeric keys before the arch string
    entries = [
        _kv_uint32(b"llama.context_length", 4096),
        _kv_string(b"general.architecture", b"llama"),
    ]
    p = tmp_path / "m2.gguf"
    p.write_bytes(_build_gguf(entries))
    assert read_gguf_metadata(str(p))["context_length"] == 4096


def test_non_gguf_returns_empty(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"NOTGGUF" + b"\x00" * 40)
    assert read_gguf_metadata(str(p)) == {}


def test_truncated_returns_empty(tmp_path):
    p = tmp_path / "t.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<IQ", 3, 0))  # header cut short
    assert read_gguf_metadata(str(p)) == {}


def test_missing_file_returns_empty():
    assert read_gguf_metadata("C:/no/such/file.gguf") == {}
