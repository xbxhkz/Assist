# src/image_training/manager.py
"""Orchestrate the SDXL image-LoRA training sidecar from the Py3.14 app.
Never imports the training stack; never raises into the route. One run
at a time. Mirrors src/training/manager.py's TrainingManager shape,
scoped to the single family/toolchain combo the feasibility spike
proved (SDXL via diffusers)."""
import json
import os
import re
import subprocess
import threading

from src.image_training.runtime import resolve_image_sidecar_script


def _default_spawn(argv):
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1,
                            env=env,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _safe_output_name(name) -> str:
    try:
        base = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name or "").strip()).strip("-.")
    except Exception:  # noqa: BLE001
        base = ""
    return base or "lora"


def _new_run_id():
    import datetime
    return datetime.datetime.now().strftime("imgrun-%Y%m%d-%H%M%S")


class ImageTrainingManager:
    def __init__(self, env=None, spawn=None, dataset_store=None, runs_dir=None, loras_dir=None):
        if env is None:
            from src.image_training.env import ImageTrainingEnv
            env = ImageTrainingEnv()
        if dataset_store is None:
            from src.image_dataset_tools.store import get_image_dataset_store
            dataset_store = get_image_dataset_store()
        if runs_dir is None:
            from src.constants import DATA_DIR
            runs_dir = os.path.join(DATA_DIR, "training", "image_training_runs")
        if loras_dir is None:
            from src.imagemodels.loras import loras_dir as _loras_dir
            loras_dir = _loras_dir
        self._env = env
        self._spawn = spawn or _default_spawn
        self._dataset_store = dataset_store
        self._runs_dir = runs_dir
        self._loras_dir = loras_dir
        self._proc = None
        self._starting = False
        self._state = {"status": "idle", "last_step": None, "loss": None,
                       "vram_gb": None, "peak_vram_gb": None, "error": None, "lora_path": None}
        self._lock = threading.Lock()

    def start(self, config) -> dict:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return {"error": "a training run is already in progress"}
            if self._starting:
                return {"error": "a training run is already starting"}
            self._starting = True
        try:
            errs = config.validate()
            if errs:
                return {"error": "; ".join(errs)}
            try:
                ds = self._dataset_store.load(config.dataset_name)
            except Exception as e:  # noqa: BLE001
                return {"error": f"dataset: {e}"}
            if not isinstance(ds, dict) or ds.get("error"):
                return {"error": f"dataset: {(ds or {}).get('error', 'not found')}"}
            imgs = ds.get("images")
            if not isinstance(imgs, list) or not imgs:
                return {"error": "dataset has no images"}
            trigger = ds.get("trigger_word") or ""
            path = ds.get("path")
            if not isinstance(path, str) or not path:
                return {"error": "dataset has no path"}

            ready = self._env.ensure_ready()
            if not ready.get("ready"):
                return {"error": f"image training env not ready: {ready.get('error')}"}

            items = []
            for img in imgs:
                if not isinstance(img, dict):
                    continue
                fn = img.get("filename")
                if not isinstance(fn, str) or not fn:
                    continue
                caption = img.get("caption") if isinstance(img.get("caption"), str) else ""
                text = f"{trigger}, {caption}" if trigger and caption else (trigger or caption)
                items.append({"image": os.path.join(path, fn), "caption": text})
            if not items:
                return {"error": "dataset has no usable images"}

            run_id = _new_run_id()
            run_dir = os.path.join(self._runs_dir, run_id)
            filename = _safe_output_name(config.output_name) + ".safetensors"
            lora_path = os.path.join(self._loras_dir(), filename)
            try:
                os.makedirs(run_dir, exist_ok=True)
                cfg = config.to_dict()
                cfg["images"] = items
                cfg["lora_path"] = lora_path
                cfg["run_dir"] = run_dir
                cfg_path = os.path.join(run_dir, "config.json")
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f)
                argv = [self._env.venv_python(), resolve_image_sidecar_script(), "--config", cfg_path]
                with self._lock:
                    self._state = {"status": "running", "last_step": None, "loss": None,
                                   "vram_gb": None, "peak_vram_gb": None, "error": None,
                                   "lora_path": lora_path, "run_id": run_id}
                    self._proc = self._spawn(argv)
                    proc = self._proc
            except Exception as e:  # noqa: BLE001
                self._state["status"] = "error"
                self._state["error"] = f"could not start training: {e}"
                return {"error": self._state["error"]}
            threading.Thread(target=self._pump, args=(proc,), daemon=True).start()
            return {"started": True, "run_id": run_id}
        except Exception as e:  # noqa: BLE001
            return {"error": f"could not start training: {e}"}
        finally:
            with self._lock:
                self._starting = False

    def _pump(self, proc):
        tail = []
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                tail.append(line)
                tail[:] = tail[-40:]
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if self._proc is not proc:
                    continue  # a newer run replaced us -- stop touching shared state
                kind = ev.get("event")
                if kind == "step":
                    self._state.update(status="running", last_step=ev.get("step"),
                                       loss=ev.get("loss"), vram_gb=ev.get("vram_gb"))
                elif kind == "done":
                    self._state.update(status="done", peak_vram_gb=ev.get("peak_vram_gb"),
                                       lora_path=ev.get("lora_path", self._state.get("lora_path")))
                elif kind == "error":
                    self._state.update(status="error", error=ev.get("message"))
        except Exception:
            pass
        finally:
            if self._proc is proc:
                rc = proc.poll()
                if rc not in (0, None) and self._state["status"] not in ("done", "error", "stopped"):
                    self._state.update(status="error",
                                       error="training process exited: " + "".join(tail)[-500:])

    def status(self) -> dict:
        return dict(self._state)

    def stop(self) -> dict:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=5)
                except Exception:
                    pass
                self._state["status"] = "stopped"
                return {"stopped": True}
            return {"stopped": False}

    def env_status(self) -> dict:
        try:
            return {"status": self._env.status()}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def setup_env(self) -> dict:
        try:
            return self._env.ensure_ready()
        except Exception as e:
            return {"ready": False, "error": str(e)}


_manager = None


def get_image_training_manager():
    global _manager
    if _manager is None:
        _manager = ImageTrainingManager()
    return _manager
