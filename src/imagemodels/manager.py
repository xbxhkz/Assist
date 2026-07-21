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
import urllib.error
import urllib.request

from src.constants import IMAGE_MODELS_DIR, MODELS_DIR
from src.desktop_runtime import choose_port
from src.imagemodels.runtime import (
    resolve_sd_binary, build_serve_argv, local_image_endpoint_url,
    list_gguf_image_models, list_image_arch_ggufs,
)


VRAM_MARGIN_GB = 1.0  # free VRAM left as headroom for sd-server compute buffers


def _default_vram_probe():
    """Live free VRAM (GB) or None. Lazy import keeps hwfit off the hot path."""
    try:
        from services.hwfit.hardware import free_vram_gb
        return free_vram_gb()
    except Exception:
        return None


def _default_log_path() -> str:
    return os.path.join(os.path.dirname(IMAGE_MODELS_DIR), "logs", "sd-server.log")


def _default_probe(url: str) -> bool:
    """True once sd-server is accepting connections. Any HTTP response — even a
    404/405 (sd-server may not serve GET /v1/models) — means it's listening;
    only a refused/failed connection means not-ready yet."""
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310 (loopback)
            return 200 <= resp.getcode() < 500
    except urllib.error.HTTPError:
        return True
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
                 sec_per_gb=12.0, force_kill=None, vram_probe=None):
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
        self._vram_probe = vram_probe or _default_vram_probe
        self._lock = threading.Lock()
        self._proc = None
        self._logf = None
        self._state = None  # {"model_path", "port", "endpoint_id", "pid", "device"}

    def _default_spawn(self, argv):
        """Launch sd-server, hidden console, output captured to the log file."""
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        # Keep the previous serve's output as .prev — fresh "wb" truncation
        # kept destroying exactly the log needed to debug the last failure.
        try:
            if os.path.exists(self._log_path):
                os.replace(self._log_path, self._log_path + ".prev")
        except OSError:
            pass
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

    def start(self, files: dict, device: str = "cpu", steps=None) -> dict:
        with self._lock:
            if self._proc is not None:
                self._stop_locked()
            # A bare diffusion model uses "diffusion_model"; an all-in-one
            # SD/SDXL checkpoint uses "checkpoint" — the primary file is either.
            model_path = files.get("diffusion_model") or files.get("checkpoint")
            threads = os.cpu_count() or 4
            # Ordered attempt ladder. A GPU request fills VRAM (Tier 1), then
            # falls back to all-RAM offload (Tier 2), then a pure-CPU serve
            # (Tier 3). No GPU tiers for an explicit CPU request.
            if device == "gpu":
                free = self._vram_probe()
                budget = max(1.0, free - VRAM_MARGIN_GB) if free else None
                attempts = ([("gpu", budget)] if budget else []) + \
                    [("gpu", None), ("cpu", None)]
            else:
                attempts = [("cpu", None)]

            last_msg = "sd-server did not start."
            for attempt_device, max_vram_gb in attempts:
                binary = self._resolve_binary(attempt_device)
                port = self._port_chooser()
                proc = self._spawn(build_serve_argv(
                    binary, files, port, device=attempt_device,
                    threads=threads, steps=steps, max_vram_gb=max_vram_gb))
                url = local_image_endpoint_url(port)
                timeout = self._ready_timeout_for(model_path)
                if self._await_ready(url + "/models", proc, timeout):
                    endpoint_id = None
                    if self._register:
                        endpoint_id = self._register(
                            name=os.path.basename(model_path), base_url=url)
                    self._proc = proc
                    self._state = {"model_path": model_path, "port": port,
                                   "endpoint_id": endpoint_id,
                                   "pid": getattr(proc, "pid", None),
                                   "device": attempt_device}
                    return self.status()
                exited = _poll(proc) is not None
                tail = _read_log_tail(self._log_path)
                self._terminate(proc)
                self._close_log()
                reason = ("exited on startup (the bundled sd.cpp may not support "
                          "this model's architecture)"
                          if exited else "did not become ready in time")
                last_msg = f"sd-server {reason}."
                if tail:
                    last_msg += "\n\n--- sd-server output (tail) ---\n" + tail
            raise RuntimeError(last_msg)

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
        """Ggufs placed in image-models/ (listed as-is) plus diffusion-
        architecture ggufs from the shared download dir, de-duplicated."""
        out = list_gguf_image_models(IMAGE_MODELS_DIR)
        seen = {os.path.realpath(m["path"]) for m in out}
        out += [m for m in list_image_arch_ggufs(MODELS_DIR)
                if os.path.realpath(m["path"]) not in seen]
        return out


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


def ensure_image_served(owner=None, *, settings=None, manager=None,
                        resolver=None, lister=None) -> dict:
    """Ensure a local image model is serving so do_generate_image can resolve one.

    Mirrors ensure_vision_served. Returns {"model": <served id>|None,
    "error": <str>|None} and NEVER raises. Behavior:
      - if the manager is already serving a local image model, reuse it (no swap);
      - else, if the configured default `image_model` names a LOCAL model, resolve
        its encoders and serve it on the GPU (falling back to CPU via start());
      - else (external/unset/unmatched default), return {"model": None} so the
        existing external/auto-detect path in do_generate_image runs unchanged.
    """
    import os
    from src.imagemodels.serve_resolve import resolve_image_files
    from src.imagemodels.encoders import MissingEncoderError

    mgr = manager or get_manager()
    try:
        st = mgr.status()
        if st.get("running"):
            return {"model": st.get("model"), "error": None}
    except Exception:
        pass  # fall through and try to serve the default

    try:
        if settings is None:
            from src.settings import load_settings
            settings = load_settings()
        default = str(settings.get("image_model") or "").strip()
    except Exception:
        default = ""
    if not default:
        return {"model": None, "error": None}

    try:
        models = (lister or mgr.list_models)()
    except Exception:
        models = []
    match = None
    dl = default.lower()
    for m in models:
        name = str(m.get("name") or "").lower()
        pbase = os.path.basename(str(m.get("path") or "")).lower()
        if dl in (name, pbase):
            match = m
            break
    if not match:
        return {"model": None, "error": None}  # external / not a local model

    try:
        files = (resolver or resolve_image_files)(match["path"])
    except MissingEncoderError as e:
        return {"model": None, "error": getattr(e, "hint", str(e))}
    except Exception as e:
        return {"model": None, "error": f"could not resolve image model files: {e}"}

    try:
        status = mgr.start(files, device="gpu")
        return {"model": status.get("model"), "error": None}
    except Exception as e:
        return {"model": None, "error": f"could not serve image model: {e}"}
