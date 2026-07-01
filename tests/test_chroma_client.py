"""Regression tests for the ChromaDB singleton client (issue #326).

Covers the fast-fail preflight (so an unreachable ChromaDB doesn't block
startup for the full OS connection timeout) and the rule that a failed
connection must not poison the cached singleton.
"""
import socket
import time

import pytest

import src.chroma_client as cc


def _free_port() -> int:
    """Bind to port 0, grab the assigned port, release it — nothing listens."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_port_open_false_for_closed_port_and_is_fast():
    port = _free_port()
    t0 = time.monotonic()
    assert cc._port_open("127.0.0.1", port, timeout=1.0) is False
    # The whole point: we fail fast, nowhere near the 30-60s OS timeout.
    assert time.monotonic() - t0 < 5.0


def test_port_open_true_for_listening_socket():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()
    try:
        assert cc._port_open(host, port, timeout=1.0) is True
    finally:
        srv.close()


def test_get_chroma_client_does_not_cache_when_unreachable(monkeypatch):
    pytest.importorskip("chromadb")
    cc.reset_client()
    monkeypatch.setenv("CHROMADB_HOST", "127.0.0.1")
    monkeypatch.setenv("CHROMADB_PORT", str(_free_port()))
    with pytest.raises(RuntimeError):
        cc.get_chroma_client()
    # A failed connection must leave the singleton unset so a later call
    # (once ChromaDB is up) can succeed.
    assert cc._client is None


def test_get_chroma_client_uses_embedded_when_no_host(monkeypatch, tmp_path):
    pytest.importorskip("chromadb")
    import chromadb
    cc.reset_client()
    monkeypatch.delenv("CHROMADB_HOST", raising=False)
    monkeypatch.setenv("CHROMADB_PATH", str(tmp_path / "chroma"))
    client = cc.get_chroma_client()
    # Embedded client is a ClientAPI backed by a local path, NOT an HTTP client.
    assert isinstance(client, chromadb.api.ClientAPI)
    # It works fully offline: create a collection and read it back.
    # Name must satisfy chromadb's collection-name validation (>=3 chars).
    col = client.get_or_create_collection("tc1")
    assert col.name == "tc1"
    # The persistent directory was created under the configured path.
    assert (tmp_path / "chroma").is_dir()
    cc.reset_client()


def test_get_chroma_client_uses_http_when_host_set(monkeypatch):
    pytest.importorskip("chromadb")
    cc.reset_client()
    monkeypatch.setenv("CHROMADB_HOST", "127.0.0.1")
    monkeypatch.setenv("CHROMADB_PORT", str(_free_port()))
    # Host is set but nothing is listening → HTTP path still fast-fails, proving
    # the embedded branch was NOT taken.
    with pytest.raises(RuntimeError):
        cc.get_chroma_client()
    assert cc._client is None
    cc.reset_client()
