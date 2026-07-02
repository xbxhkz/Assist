"""Streaming GGUF downloader (Phase 3b).

One download at a time, into MODELS_DIR, atomic .part -> .gguf on success.
The transfer (http_stream) and the thread (spawn) are injectable so progress,
cancel, and atomic rename are unit-testable without network or real threads.
"""
import os
import threading
from contextlib import contextmanager

from src.constants import MODELS_DIR


class _Cancelled(Exception):
    pass


def _safe_filename(filename):
    """Return a safe .gguf basename, or None if unsafe."""
    if not filename or "/" in filename or "\\" in filename:
        return None
    if os.path.basename(filename) != filename:
        return None
    if not filename.lower().endswith(".gguf"):
        return None
    return filename


@contextmanager
def _default_http_stream(url, headers):
    import httpx
    with httpx.stream("GET", url, headers=headers, timeout=None,
                      follow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0) or None
        yield total, r.iter_bytes()


class DownloadManager:
    def __init__(self, http_stream=_default_http_stream, spawn=None,
                 dest_dir=None, headers_provider=None):
        self._http_stream = http_stream
        self._dest_dir = dest_dir or MODELS_DIR
        self._headers_provider = headers_provider or (lambda: {})
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread = None
        self._active = False
        self._state = None  # {filename, bytes, total, error, done}

        if spawn is not None:
            self._spawn = spawn
        else:
            def _thread_spawn(fn):
                self._thread = threading.Thread(target=fn, daemon=True)
                self._thread.start()
            self._spawn = _thread_spawn

    def start(self, url, filename):
        with self._lock:
            if self._active:
                raise RuntimeError("a download is already in progress")
            safe = _safe_filename(filename)
            if not safe:
                raise ValueError("invalid filename (must be a plain .gguf name)")
            self._cancel.clear()
            self._state = {"filename": safe, "bytes": 0, "total": None,
                           "error": None, "done": False}
            self._active = True

        def job():
            try:
                self._transfer(url, safe)
            finally:
                self._active = False

        self._spawn(job)
        return self.status()

    def _transfer(self, url, filename):
        os.makedirs(self._dest_dir, exist_ok=True)
        part = os.path.join(self._dest_dir, filename + ".part")
        final = os.path.join(self._dest_dir, filename)
        try:
            with self._http_stream(url, self._headers_provider()) as (total, chunks):
                self._state["total"] = total
                with open(part, "wb") as f:
                    for chunk in chunks:
                        if self._cancel.is_set():
                            raise _Cancelled()
                        f.write(chunk)
                        self._state["bytes"] += len(chunk)
            os.replace(part, final)
            self._state["done"] = True
        except _Cancelled:
            self._cleanup(part)
        except Exception as e:  # network, disk, HTTP status
            self._state["error"] = str(e)
            self._cleanup(part)

    def _cleanup(self, part):
        try:
            if os.path.exists(part):
                os.remove(part)
        except Exception:
            pass

    def cancel(self):
        self._cancel.set()
        t = self._thread
        if t is not None:
            try:
                t.join(timeout=10)
            except Exception:
                pass
        return self.status()

    def wait(self, timeout=5):
        t = self._thread
        if t is not None:
            t.join(timeout)

    def status(self):
        s = self._state
        if s is None:
            return {"downloading": False, "filename": None, "bytes": 0,
                    "total": None, "pct": None, "error": None}
        pct = round(100.0 * s["bytes"] / s["total"], 1) if s["total"] else None
        return {"downloading": bool(self._active and not s["done"]),
                "filename": s["filename"], "bytes": s["bytes"],
                "total": s["total"], "pct": pct, "error": s["error"]}


_download_manager = None


def get_download_manager():
    """Process-wide singleton, wired with HF auth headers for gated files."""
    global _download_manager
    if _download_manager is None:
        from src.localmodels.catalog import _hf_headers
        _download_manager = DownloadManager(headers_provider=_hf_headers)
    return _download_manager
