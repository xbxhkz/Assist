"""Stateful manager for a single native local model (Phase 3a).

Owns at most one llama-server subprocess at a time. All side-effecting
dependencies (process spawn, port choice, readiness probe, endpoint register/
unregister, binary resolution, clock) are injectable so the state machine is
unit-testable without a real binary, port, or database.

llama-server's stdout/stderr are captured to a log file (not discarded), and
the readiness wait is process-aware: if the server exits during startup — e.g.
a model whose architecture the bundled llama.cpp doesn't support — the manager
fails fast and surfaces the captured output instead of waiting out the full
timeout with no signal.
"""
import os
import subprocess
import threading
import time
import urllib.request

from src.constants import MODELS_DIR
from src.desktop_runtime import choose_port
from src.localmodels.runtime import (
    resolve_llama_binary, build_serve_argv, local_endpoint_url, list_gguf_models,
)


def _default_log_path() -> str:
    """Where llama-server's captured output is written (next to app logs)."""
    return os.path.join(os.path.dirname(MODELS_DIR), "logs", "llama-server.log")


def _default_probe(url: str) -> bool:
    """Single-shot readiness check: True if `url` answers with a non-5xx status."""
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310 (loopback)
            return 200 <= resp.getcode() < 500
    except Exception:
        return False


def _poll(proc):
    """Return the process's exit code, or None if still running / unknown."""
    poll = getattr(proc, "poll", None)
    if poll is None:
        return None
    try:
        return poll()
    except Exception:
        return None


def _read_log_tail(path: str, n: int = 4000) -> str:
    """Return the last `n` bytes of `path` as text (for error messages)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", "replace").strip()
    except Exception:
        return ""


class LocalModelManager:
    def __init__(self, spawn=None, port_chooser=None, probe=None,
                 register_endpoint=None, unregister_endpoint=None,
                 resolve_binary=resolve_llama_binary, log_path=None,
                 sleep=None, now=None, ready_timeout=45.0, probe_interval=0.5):
        self._log_path = log_path or _default_log_path()
        self._spawn = spawn or self._default_spawn
        self._port_chooser = port_chooser or (lambda: choose_port(8100))
        self._probe = probe or _default_probe
        self._sleep = sleep or time.sleep
        self._now = now or time.monotonic
        self._ready_timeout = ready_timeout
        self._probe_interval = probe_interval
        self._register = register_endpoint
        self._unregister = unregister_endpoint
        self._resolve_binary = resolve_binary
        self._lock = threading.Lock()
        self._proc = None
        self._logf = None
        self._state = None  # {"model_path", "port", "endpoint_id", "pid"}

    def _default_spawn(self, argv):
        """Launch llama-server, capturing stdout+stderr to the log file."""
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        self._logf = open(self._log_path, "wb")
        return subprocess.Popen(argv, stdout=self._logf,
                                stderr=subprocess.STDOUT)

    def _await_ready(self, url: str, proc) -> bool:
        """Poll `url` until ready, bailing the instant the process exits."""
        deadline = self._now() + self._ready_timeout
        while self._now() < deadline:
            if _poll(proc) is not None:
                return False  # llama-server exited on startup
            if self._probe(url):
                return True
            self._sleep(self._probe_interval)
        return False

    def start(self, model_path: str) -> dict:
        with self._lock:
            if self._proc is not None:
                self._stop_locked()
            binary = self._resolve_binary()
            port = self._port_chooser()
            proc = self._spawn(build_serve_argv(binary, model_path, port))
            url = local_endpoint_url(port)
            if not self._await_ready(url + "/models", proc):
                exited = _poll(proc) is not None
                tail = _read_log_tail(self._log_path)
                self._terminate(proc)
                self._close_log()
                reason = ("exited on startup (the bundled llama.cpp may not "
                          "support this model's architecture)"
                          if exited else "did not become ready in time")
                msg = f"llama-server {reason}."
                if tail:
                    msg += "\n\n--- llama-server output (tail) ---\n" + tail
                raise RuntimeError(msg)
            endpoint_id = None
            if self._register:
                endpoint_id = self._register(name=os.path.basename(model_path),
                                             base_url=url)
            self._proc = proc
            self._state = {"model_path": model_path, "port": port,
                           "endpoint_id": endpoint_id,
                           "pid": getattr(proc, "pid", None)}
            return self.status()

    def stop(self) -> dict:
        with self._lock:
            return self._stop_locked()

    @staticmethod
    def _terminate(proc):
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        except Exception:
            pass

    def _close_log(self):
        if self._logf is not None:
            try:
                self._logf.close()
            except Exception:
                pass
            self._logf = None

    def _stop_locked(self) -> dict:
        if self._proc is not None:
            proc = self._proc
            self._proc = None
            self._terminate(proc)
            self._close_log()
        if self._state and self._state.get("endpoint_id") and self._unregister:
            try:
                self._unregister(self._state["endpoint_id"])
            except Exception:
                pass
        self._state = None
        return self.status()

    def status(self) -> dict:
        if self._state is None:
            return {"running": False, "model": None, "port": None,
                    "endpoint_id": None}
        return {"running": True,
                "model": os.path.basename(self._state["model_path"]),
                "port": self._state["port"],
                "endpoint_id": self._state["endpoint_id"]}

    def list_models(self) -> list:
        return list_gguf_models(MODELS_DIR)

    def delete_model(self, filename: str) -> dict:
        """Delete a downloaded .gguf from MODELS_DIR; stop it first if serving."""
        base = os.path.basename(filename or "")
        if not base or base != filename or not base.lower().endswith(".gguf"):
            raise ValueError("invalid filename (must be a plain .gguf name)")
        with self._lock:
            if self._state and os.path.basename(
                    self._state.get("model_path") or "") == base:
                self._stop_locked()
            real_dir = os.path.realpath(MODELS_DIR)
            real = os.path.realpath(os.path.join(MODELS_DIR, base))
            try:
                inside = os.path.commonpath([real, real_dir]) == real_dir
            except ValueError:
                inside = False
            if inside and os.path.isfile(real):
                try:
                    os.remove(real)
                except Exception:
                    pass
        return self.status()


_manager = None


def get_manager() -> "LocalModelManager":
    """Process-wide singleton wired with the real endpoint store."""
    global _manager
    if _manager is None:
        from src.localmodels.store import (
            register_local_endpoint, unregister_local_endpoint,
        )
        _manager = LocalModelManager(
            register_endpoint=register_local_endpoint,
            unregister_endpoint=unregister_local_endpoint,
        )
    return _manager
