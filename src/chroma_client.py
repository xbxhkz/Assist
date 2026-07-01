"""
chroma_client.py

Singleton ChromaDB client. Embedded (on-disk PersistentClient) by default;
connects to a standalone ChromaDB service when CHROMADB_HOST is set.
"""

import os
import socket
import logging

logger = logging.getLogger(__name__)

_client = None

# A short connect probe so an unreachable ChromaDB fails fast instead of
# blocking on the OS connection timeout (~30-60s, WinError 10060 on Windows),
# which otherwise stalls app startup. Tunable via CHROMADB_CONNECT_TIMEOUT.
_CONNECT_TIMEOUT = float(os.getenv("CHROMADB_CONNECT_TIMEOUT", "2.0"))


def _port_open(host: str, port: int, timeout: float = None) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout or _CONNECT_TIMEOUT):
            return True
    except OSError:
        return False


def get_chroma_client():
    """Get or create the singleton ChromaDB client.

    With no external service configured (``CHROMADB_HOST`` unset/empty) this
    returns an *embedded* ``PersistentClient`` writing under ``CHROMADB_PATH``
    (default ``<DATA_DIR>/chroma``) — no server, no socket. When ``CHROMADB_HOST``
    is set it behaves as before: a fast-failing ``HttpClient`` to a standalone
    ChromaDB service (the Docker path).

    Raises RuntimeError with a clear install hint if the `chromadb` package is
    not installed.
    """
    global _client
    if _client is not None:
        return _client

    try:
        import chromadb
    except ImportError as e:
        raise RuntimeError(
            "ChromaDB integration is not installed. Install it with: "
            "pip install chromadb"
        ) from e

    host = (os.getenv("CHROMADB_HOST") or "").strip()

    if not host:
        # Embedded mode: local on-disk store, no standalone service required.
        from src.constants import DATA_DIR
        path = (os.getenv("CHROMADB_PATH") or "").strip() or os.path.join(DATA_DIR, "chroma")
        os.makedirs(path, exist_ok=True)
        client = chromadb.PersistentClient(path=path)
        client.heartbeat()
        _client = client
        logger.info(f"ChromaDB embedded (persistent) at: {path}")
        return _client

    # HTTP mode: talk to a standalone ChromaDB service (Docker/self-hosted).
    port = int(os.getenv("CHROMADB_PORT", "8100"))
    if not _port_open(host, port):
        raise RuntimeError(
            f"ChromaDB is not reachable at {host}:{port}. Start the ChromaDB "
            f"service (e.g. `docker compose up chromadb`) or set CHROMADB_HOST / "
            f"CHROMADB_PORT to point at a running instance."
        )
    client = chromadb.HttpClient(host=host, port=port)
    # Health check before caching — if the port is open but the service isn't
    # healthy yet (e.g. still starting), don't poison the singleton with a dead
    # client; leave _client unset so the next call retries.
    client.heartbeat()
    _client = client
    logger.info(f"ChromaDB connected: {host}:{port}")
    return _client


def reset_client():
    """Reset the singleton (e.g. after config change)."""
    global _client
    _client = None
