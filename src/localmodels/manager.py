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
import json
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


def _last_model_file() -> str:
    """State file remembering the last successfully-served model, so it can be
    auto-served on the next launch."""
    return os.path.join(os.path.dirname(MODELS_DIR), "last_model.json")


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


def _default_force_kill(pid) -> None:
    """Forcefully kill a whole process tree (Windows: taskkill /F /T) as a last
    resort when terminate()/kill() don't reap a stuck child — e.g. llama-server
    mid-mmap of a huge model — so it can't orphan and hold gigabytes of RAM."""
    if not pid:
        return
    try:
        from core.platform_compat import kill_process_tree
        kill_process_tree(pid)
    except Exception:
        pass


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
                 sleep=None, now=None, ready_timeout=45.0, probe_interval=0.5,
                 sec_per_gb=12.0, force_kill=None):
        self._log_path = log_path or _default_log_path()
        self._spawn = spawn or self._default_spawn
        self._port_chooser = port_chooser or (lambda: choose_port(8100))
        self._probe = probe or _default_probe
        self._sleep = sleep or time.sleep
        self._now = now or time.monotonic
        self._ready_timeout = ready_timeout
        self._sec_per_gb = sec_per_gb
        self._probe_interval = probe_interval
        self._register = register_endpoint
        self._unregister = unregister_endpoint
        self._resolve_binary = resolve_binary
        self._force_kill = force_kill or _default_force_kill
        self._lock = threading.Lock()
        self._proc = None
        self._logf = None
        self._state = None  # {"model_path", "port", "endpoint_id", "pid"}

    def _default_spawn(self, argv):
        """Launch llama-server, capturing stdout+stderr to the log file.

        CREATE_NO_WINDOW keeps the child's console hidden — without it, a
        windowed (no-console) build pops a visible console window for
        llama-server. No-op on POSIX (the flag is 0 there)."""
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        self._logf = open(self._log_path, "wb")
        return subprocess.Popen(argv, stdout=self._logf,
                                stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def _timeout_for_bytes(self, nbytes: int) -> float:
        """Readiness timeout scaled by model size — a 48GB model can take
        minutes to mmap off disk, so a flat cap would abandon a healthy load.
        The base `ready_timeout` is the floor for small models."""
        return max(self._ready_timeout, (nbytes / 1e9) * self._sec_per_gb)

    def _ready_timeout_for(self, model_path: str) -> float:
        try:
            return self._timeout_for_bytes(os.path.getsize(model_path))
        except OSError:
            return self._ready_timeout

    def _await_ready(self, url: str, proc, timeout: float) -> bool:
        """Poll `url` until ready, bailing the instant the process exits."""
        deadline = self._now() + timeout
        while self._now() < deadline:
            if _poll(proc) is not None:
                return False  # llama-server exited on startup
            if self._probe(url):
                return True
            self._sleep(self._probe_interval)
        return False

    def start(self, model_path: str, device: str = "cpu") -> dict:
        with self._lock:
            if self._proc is not None:
                self._stop_locked()
            binary = self._resolve_binary(device=device)
            port = self._port_chooser()
            proc = self._spawn(build_serve_argv(binary, model_path, port,
                                                device=device))
            url = local_endpoint_url(port)
            timeout = self._ready_timeout_for(model_path)
            if not self._await_ready(url + "/models", proc, timeout):
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
                           "endpoint_id": endpoint_id, "device": device,
                           "pid": getattr(proc, "pid", None)}
            self._persist_last_model(model_path, device)
            return self.status()

    def _persist_last_model(self, model_path: str, device: str = "cpu") -> None:
        """Remember this model (and device) to auto-serve next launch."""
        try:
            f = _last_model_file()
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, "w", encoding="utf-8") as fh:
                json.dump({"model_path": model_path, "device": device}, fh)
        except Exception:
            pass

    def stop(self) -> dict:
        with self._lock:
            return self._stop_locked()

    def _terminate(self, proc):
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass
        except Exception:
            pass
        # Last resort: if it still hasn't died (a huge model mmap can make
        # terminate/kill unreliable), force-kill the tree so it can't orphan.
        if _poll(proc) is None:
            self._force_kill(getattr(proc, "pid", None))

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
                    "endpoint_id": None, "device": None}
        return {"running": True,
                "model": os.path.basename(self._state["model_path"]),
                "port": self._state["port"],
                "endpoint_id": self._state["endpoint_id"],
                "device": self._state.get("device", "cpu")}

    def list_models(self) -> list:
        """Downloaded models (in MODELS_DIR) plus linked/external ones, each
        tagged with an `external` flag; de-duplicated by realpath."""
        from src.localmodels.external import list_external_models
        downloaded = list_gguf_models(MODELS_DIR)
        for m in downloaded:
            m["external"] = False
        seen = {os.path.realpath(m["path"]) for m in downloaded}
        linked = [e for e in list_external_models()
                  if os.path.realpath(e["path"]) not in seen]
        return downloaded + linked

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


def autoserve_last_model(manager=None, model_file=None):
    """Re-serve the last successfully-served model (best-effort).

    Reads the persisted path and starts it. No-ops if there is no record, the
    model file is gone, a model is already running, or serving fails — so it is
    safe to fire-and-forget in a background thread at startup. `manager` and
    `model_file` are injectable for tests.
    """
    model_file = model_file or _last_model_file()
    try:
        with open(model_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    path = (data or {}).get("model_path")
    if not path or not os.path.isfile(path):
        return None
    device = (data or {}).get("device") or "cpu"
    mgr = manager or get_manager()
    try:
        if mgr.status().get("running"):
            return None
        return mgr.start(path, device=device)
    except Exception:
        return None
