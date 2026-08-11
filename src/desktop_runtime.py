"""Pure, unit-testable helpers for the native desktop launcher.

No GUI and no top-level `webview` import so this module is safe to import in
any environment (including the headless Docker/server path). The launcher
(`launcher.py`) wires these together and owns the pywebview window.
"""
import logging
import os
import socket
import sys
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


def run_server_logging_errors(server, logger=None) -> None:
    """Run a uvicorn Server, logging (not swallowing) any exception it raises.

    Meant as the target of the background thread that runs the server. A
    windowed (console-less) frozen build has no visible stdout/stderr, so an
    exception here (e.g. a port-bind race between choose_port()'s check and
    the server's real bind) would otherwise vanish with zero trace, making a
    genuine crash indistinguishable from a slow cold start once
    wait_for_server_ready times out.
    """
    log = logger or logging.getLogger(__name__)
    try:
        server.run()
    except Exception:
        log.exception("Background server thread failed")


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


def bundled_fastembed_cache() -> str | None:
    """Path to the embedding-model cache bundled into the frozen app, if present.

    Returns None in normal (non-frozen) runs so the app uses its default cache.
    """
    if not getattr(sys, "frozen", False):
        return None
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return None
    path = os.path.join(base, "fastembed_cache")
    return path if os.path.isdir(path) else None


# Evergreen WebView2 Runtime registration key (per Microsoft docs).
_WEBVIEW2_KEY = (r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients"
                 r"\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}")


def _read_webview2_pv() -> str | None:
    """Read the installed WebView2 runtime version ('pv') from the registry."""
    try:
        import winreg
    except ImportError:
        return None  # non-Windows
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(hive, _WEBVIEW2_KEY) as key:
                pv, _ = winreg.QueryValueEx(key, "pv")
                if pv:
                    return pv
        except OSError:
            continue
    return None


def webview2_runtime_available(read_pv=None) -> bool:
    """True when a real Edge WebView2 runtime version is registered."""
    read_pv = read_pv or _read_webview2_pv
    pv = read_pv()
    return bool(pv) and pv != "0.0.0.0"
