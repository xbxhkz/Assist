# launcher.py
"""Native desktop entrypoint for Assist.

Runs the FastAPI backend (uvicorn) on a daemon thread, then opens a native
WebView2 window (pywebview) on the main thread pointing at the local server.
Closing the window stops the server and exits. In dev (`python launcher.py`)
it behaves the same; in a frozen build it also loads the bundled embedding
model offline.
"""
import src.brand_compat  # noqa: F401  -- bridge ASSIST_*/ODYSSEUS_* env at startup

import os
import sys
import threading


# Windowed (no-console) builds have no real stdout/stderr; guard library
# code that calls .write()/.isatty() so it can't crash the process.
class NullWriter:
    def write(self, text):
        pass

    def flush(self):
        pass

    def isatty(self):
        return False


if sys.stdout is None:
    sys.stdout = NullWriter()
if sys.stderr is None:
    sys.stderr = NullWriter()


def _show_error(message: str) -> None:
    """Show a blocking native error box (Windows), else print."""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, message, "Assist", 0x10)
    except Exception:
        print(message)


def main() -> None:
    from src.desktop_runtime import (
        choose_port, local_origin, augment_allowed_origins,
        make_uvicorn_server, wait_for_server_ready,
        bundled_fastembed_cache, webview2_runtime_available,
    )

    # The desktop app is intentionally loopback-only: the window, CORS origin,
    # and health poll below all assume 127.0.0.1, so the bind host must not be
    # configurable (binding 0.0.0.0 would expose the backend to the network).
    host = "127.0.0.1"
    port = choose_port(int(os.getenv("APP_PORT", "7000")))
    origin = local_origin(port)

    # These env vars are read at import time by app.py / src.constants, so set
    # them BEFORE importing the app.
    os.environ["ALLOWED_ORIGINS"] = augment_allowed_origins(origin)
    cache = bundled_fastembed_cache()
    if cache and not os.getenv("FASTEMBED_CACHE_PATH"):
        os.environ["FASTEMBED_CACHE_PATH"] = cache

    from app import app

    server = make_uvicorn_server(app, host, port)
    threading.Thread(target=server.run, daemon=True).start()

    if not wait_for_server_ready(origin + "/api/health", timeout=45.0):
        _show_error("Assist could not start its background service in time.")
        server.should_exit = True
        return

    if not webview2_runtime_available():
        _show_error(
            "Assist needs the Microsoft Edge WebView2 Runtime.\n\n"
            "Install it from:\n"
            "https://developer.microsoft.com/microsoft-edge/webview2/"
        )
        server.should_exit = True
        return

    import webview

    class _JsApi:
        """Native bridges callable from the page as window.pywebview.api.<m>()."""
        def __init__(self):
            self._window = None

        def pick_gguf(self):
            """Open a native file dialog and return the chosen .gguf absolute
            path, or '' if cancelled. Used by the Local Models "Browse" button
            to link a model that lives anywhere on disk."""
            try:
                paths = self._window.create_file_dialog(
                    webview.OPEN_DIALOG, allow_multiple=False,
                    file_types=('GGUF model (*.gguf)', 'All files (*.*)'))
                return paths[0] if paths else ''
            except Exception:
                return ''

    api = _JsApi()
    api._window = webview.create_window("Assist", origin, js_api=api)
    webview.start()  # blocks until the window is closed

    server.should_exit = True


if __name__ == "__main__":
    main()
