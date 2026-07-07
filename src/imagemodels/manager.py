"""Stateful manager for a single native local image model (sd-server).

Owns at most one sd-server subprocess. All side-effecting dependencies are
injectable so the state machine is unit-testable without a real binary, port,
GPU, or database. Mirrors src/localmodels/manager.py (readiness that bails on
early exit, size-scaled timeout, robust process-tree kill, hidden console,
captured log) — adapted for image models (start takes the 4 FLUX files + device).
"""
import json
import os
import subprocess
import threading
import time
import urllib.request

from src.constants import IMAGE_MODELS_DIR
from src.desktop_runtime import choose_port
from src.imagemodels.runtime import (
    resolve_sd_binary, build_serve_argv, local_image_endpoint_url,
    list_gguf_image_models,
)


def _default_log_path() -> str:
    return os.path.join(os.path.dirname(IMAGE_MODELS_DIR), "logs", "sd-server.log")


def _default_probe(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310 (loopback)
            return 200 <= resp.getcode() < 500
    except Exception:
        return False


def _poll(proc):
    poll = getattr(proc, "poll", None)
    if poll is None:
        return None
    try:
        return poll()
    except Exception:
        return None


def _read_log_tail(path: str, n: int = 4000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", "replace").strip()
    except Exception:
        return ""


def _default_force_kill(pid) -> None:
    if not pid:
        return
    try:
        from core.platform_compat import kill_process_tree
        kill_process_tree(pid)
    except Exception:
        pass


class ImageModelManager:
    def __init__(self, spawn=None, port_chooser=None, probe=None,
                 register_endpoint=None, unregister_endpoint=None,
                 resolve_binary=resolve_sd_binary, log_path=None,
                 sleep=None, now=None, ready_timeout=45.0, probe_interval=0.5,
                 sec_per_gb=12.0, force_kill=None):
        self._log_path = log_path or _default_log_path()
        self._spawn = spawn or self._default_spawn
        self._port_chooser = port_chooser or (lambda: choose_port(8200))
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
        self._state = None  # {"model_path", "port", "endpoint_id", "pid", "device"}

    def _default_spawn(self, argv):
        """Launch sd-server, hidden console, output captured to the log file."""
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        self._logf = open(self._log_path, "wb")
        return subprocess.Popen(argv, stdout=self._logf, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def _timeout_for_bytes(self, nbytes: int) -> float:
        return max(self._ready_timeout, (nbytes / 1e9) * self._sec_per_gb)

    def _ready_timeout_for(self, model_path: str) -> float:
        try:
            return self._timeout_for_bytes(os.path.getsize(model_path))
        except OSError:
            return self._ready_timeout

    def _await_ready(self, url: str, proc, timeout: float) -> bool:
        deadline = self._now() + timeout
        while self._now() < deadline:
            if _poll(proc) is not None:
                return False  # sd-server exited on startup
            if self._probe(url):
                return True
            self._sleep(self._probe_interval)
        return False

    def start(self, files: dict, device: str = "cpu") -> dict:
        with self._lock:
            if self._proc is not None:
                self._stop_locked()
            binary = self._resolve_binary(device)
            port = self._port_chooser()
            threads = os.cpu_count() or 4
            proc = self._spawn(build_serve_argv(binary, files, port, device=device,
                                                threads=threads))
            url = local_image_endpoint_url(port)
            timeout = self._ready_timeout_for(files["diffusion_model"])
            if not self._await_ready(url + "/models", proc, timeout):
                exited = _poll(proc) is not None
                tail = _read_log_tail(self._log_path)
                self._terminate(proc)
                self._close_log()
                reason = ("exited on startup (the bundled sd.cpp may not support "
                          "this model's architecture)"
                          if exited else "did not become ready in time")
                msg = f"sd-server {reason}."
                if tail:
                    msg += "\n\n--- sd-server output (tail) ---\n" + tail
                raise RuntimeError(msg)
            endpoint_id = None
            if self._register:
                endpoint_id = self._register(
                    name=os.path.basename(files["diffusion_model"]), base_url=url)
            self._proc = proc
            self._state = {"model_path": files["diffusion_model"], "port": port,
                           "endpoint_id": endpoint_id,
                           "pid": getattr(proc, "pid", None), "device": device}
            return self.status()

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
                "device": self._state["device"]}

    def list_models(self) -> list:
        return list_gguf_image_models(IMAGE_MODELS_DIR)


_manager = None


def get_manager() -> "ImageModelManager":
    global _manager
    if _manager is None:
        from src.imagemodels.store import (
            register_image_endpoint, unregister_image_endpoint,
        )
        _manager = ImageModelManager(
            register_endpoint=register_image_endpoint,
            unregister_endpoint=unregister_image_endpoint,
        )
    return _manager
