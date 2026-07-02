"""Stateful manager for a single native local model (Phase 3a).

Owns at most one llama-server subprocess at a time. All side-effecting
dependencies (process spawn, port choice, readiness poll, endpoint register/
unregister, binary resolution) are injectable so the state machine is
unit-testable without a real binary, port, or database.
"""
import os
import subprocess
import threading

from src.constants import MODELS_DIR
from src.desktop_runtime import choose_port, wait_for_server_ready
from src.localmodels.runtime import (
    resolve_llama_binary, build_serve_argv, local_endpoint_url, list_gguf_models,
)


def _default_spawn(argv):
    return subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


class LocalModelManager:
    def __init__(self, spawn=_default_spawn, port_chooser=None, readiness=None,
                 register_endpoint=None, unregister_endpoint=None,
                 resolve_binary=resolve_llama_binary):
        self._spawn = spawn
        self._port_chooser = port_chooser or (lambda: choose_port(8100))
        self._readiness = readiness or wait_for_server_ready
        self._register = register_endpoint
        self._unregister = unregister_endpoint
        self._resolve_binary = resolve_binary
        self._lock = threading.Lock()
        self._proc = None
        self._state = None  # {"model_path", "port", "endpoint_id", "pid"}

    def start(self, model_path: str) -> dict:
        with self._lock:
            if self._proc is not None:
                self._stop_locked()
            binary = self._resolve_binary()
            port = self._port_chooser()
            proc = self._spawn(build_serve_argv(binary, model_path, port))
            url = local_endpoint_url(port)
            if not self._readiness(url + "/models"):
                try:
                    proc.terminate()
                except Exception:
                    pass
                raise RuntimeError("llama-server did not become ready in time")
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

    def _stop_locked(self) -> dict:
        if self._proc is not None:
            proc = self._proc
            self._proc = None
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
