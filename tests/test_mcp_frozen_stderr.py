"""MCP stdio spawn must survive the frozen windowed build.

PyInstaller's windowed mode replaces sys.stderr with a NullWriter that has no
fileno(); the mcp SDK's stdio_client defaults errlog=sys.stderr and hands it
to subprocess spawn, which crashes ('NullWriter' object has no attribute
'fileno') — every built-in MCP server failed on every boot since 07-01.
"""
import sys

import src.mcp_manager as mm


class _NullWriter:
    """Mimics PyInstaller's console-less stderr stub."""
    def write(self, *_a, **_k):
        pass
    def flush(self):
        pass


def test_safe_errlog_none_when_stderr_is_real():
    # Under pytest sys.stderr is a real (or fileno-capable) stream.
    if not hasattr(sys.stderr, "fileno"):
        return  # environment already odd; covered by the other test
    try:
        sys.stderr.fileno()
    except Exception:
        return
    assert mm._safe_errlog() is None


def test_safe_errlog_returns_real_file_when_stderr_unusable(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "stderr", _NullWriter())
    monkeypatch.setattr(mm, "DATA_DIR", str(tmp_path), raising=False)
    f = mm._safe_errlog()
    try:
        assert f is not None
        assert isinstance(f.fileno(), int)  # usable by subprocess spawn
    finally:
        if f is not None:
            f.close()
