"""Pure, unit-testable helpers for the native desktop launcher.

No GUI and no top-level `webview` import so this module is safe to import in
any environment (including the headless Docker/server path). The launcher
(`launcher.py`) wires these together and owns the pywebview window.
"""
import os
import socket
import time
import urllib.request


def _port_bindable(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def choose_port(preferred: int = 7000) -> int:
    """Return `preferred` if bindable on loopback, else an OS-assigned free port."""
    return preferred if _port_bindable(preferred) else _free_port()


def local_origin(port: int) -> str:
    """The loopback origin/URL the window and CORS checks use."""
    return f"http://127.0.0.1:{port}"


def augment_allowed_origins(origin: str, existing: str | None = None) -> str:
    """Return an ALLOWED_ORIGINS value that includes `origin` exactly once.

    CORS origin matching is scheme+host+port, so the window's ported origin
    (http://127.0.0.1:<port>) must be added to the unported defaults.
    """
    if existing is None:
        existing = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://127.0.0.1")
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    if origin not in parts:
        parts.append(origin)
    return ",".join(parts)


def make_uvicorn_server(app, host: str, port: int, log_level: str = "info"):
    """Build a uvicorn.Server we can run on a thread and stop via should_exit."""
    import uvicorn
    config = uvicorn.Config(app=app, host=host, port=port, log_level=log_level)
    return uvicorn.Server(config)


def _default_opener(url: str) -> int:
    with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310 (loopback only)
        return resp.getcode()


def wait_for_server_ready(url: str, timeout: float = 45.0, interval: float = 0.25,
                          opener=None, sleep=time.sleep, now=time.monotonic) -> bool:
    """Poll `url` until it answers with a non-5xx status or the deadline passes.

    `opener(url) -> int` and `sleep`/`now` are injectable for tests.
    """
    opener = opener or _default_opener
    deadline = now() + timeout
    while now() < deadline:
        try:
            code = opener(url)
            if 200 <= code < 500:
                return True
        except Exception:
            pass
        sleep(interval)
    return False
