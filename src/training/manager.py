"""Orchestrate the training sidecar from the Py3.14 app. Never imports the
training stack; never raises into the route. One run at a time."""
import json
import os
import subprocess
import threading

from src.training.runtime import resolve_sidecar_script


def _default_spawn(argv):
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1,
                            env=env,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _default_free_vram():
    try:
        from services.hwfit.hardware import free_vram_gb
        return free_vram_gb()
    except Exception:
        return None


class TrainingManager:
    def __init__(self, env=None, spawn=None, free_vram=None, adapters_dir=None):
        if env is None:
            from src.training.env import TrainingEnv
            env = TrainingEnv()
        if adapters_dir is None:
            from src.constants import DATA_DIR
            adapters_dir = os.path.join(DATA_DIR, "training", "adapters")
        self._env = env
        self._spawn = spawn or _default_spawn
        self._free_vram = free_vram or _default_free_vram
        self._adapters_dir = adapters_dir
        self._proc = None
        self._starting = False
        self._state = {"status": "idle", "last_step": None, "loss": None,
                       "vram_gb": None, "peak_vram_gb": None, "error": None, "output_dir": None}
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
                from src.training.dataset import load_jsonl
                rows = load_jsonl(config.dataset_path)
            except Exception as e:
                return {"error": f"dataset: {e}"}
            ready = self._env.ensure_ready()
            if not ready.get("ready"):
                return {"error": f"training env not ready: {ready.get('error')}"}

            # VRAM soft-gate (warn only — the user chose the model). The injectable
            # free_vram is called defensively: a failing probe must not break start().
            from src.training.config import parse_params_b, fit_level
            warning = None
            try:
                level = fit_level(parse_params_b(config.base_model), self._free_vram())
            except Exception:
                level = "unknown"
            if level == "too_big":
                warning = "model may exceed available VRAM; it may fail with out-of-memory"

            run_id = _new_run_id()
            out_dir = os.path.join(self._adapters_dir, run_id)
            try:
                os.makedirs(out_dir, exist_ok=True)
                ds_path = os.path.join(out_dir, "dataset.jsonl")
                with open(ds_path, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                cfg = config.to_dict()
                cfg["dataset_path"] = ds_path
                cfg["output_dir"] = out_dir
                cfg_path = os.path.join(out_dir, "config.json")
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f)
                argv = [self._env.venv_python(), resolve_sidecar_script(), "--config", cfg_path]
                with self._lock:
                    self._state = {"status": "running", "last_step": None, "loss": None,
                                   "vram_gb": None, "peak_vram_gb": None, "error": warning,
                                   "output_dir": out_dir, "run_id": run_id}
                    self._proc = self._spawn(argv)
                    proc = self._proc
            except Exception as e:
                self._state["status"] = "error"
                self._state["error"] = f"could not start training: {e}"
                return {"error": self._state["error"]}
            threading.Thread(target=self._pump, args=(proc,), daemon=True).start()
            return {"started": True, "run_id": run_id, "warning": warning}
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
                    continue  # a newer run replaced us — stop touching shared state
                kind = ev.get("event")
                if kind == "step":
                    self._state.update(status="running", last_step=ev.get("step"),
                                       loss=ev.get("loss"), vram_gb=ev.get("vram_gb"))
                elif kind == "done":
                    self._state.update(status="done", peak_vram_gb=ev.get("peak_vram_gb"),
                                       output_dir=ev.get("output_dir", self._state.get("output_dir")))
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

    def list_adapters(self) -> list:
        out = []
        if not os.path.isdir(self._adapters_dir):
            return out
        for name in sorted(os.listdir(self._adapters_dir), reverse=True):
            d = os.path.join(self._adapters_dir, name)
            if not os.path.isdir(d):
                continue
            has = os.path.isfile(os.path.join(d, "adapter_model.safetensors"))
            cfg = {}
            rc = os.path.join(d, "run_config.json")
            if os.path.isfile(rc):
                try:
                    with open(rc, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                except Exception:
                    cfg = {}
            out.append({"run_id": name, "complete": has,
                        "base_model": cfg.get("base_model"), "path": d})
        return out


def _new_run_id():
    import datetime
    return datetime.datetime.now().strftime("run-%Y%m%d-%H%M%S")


_manager = None


def get_training_manager():
    global _manager
    if _manager is None:
        _manager = TrainingManager()
    return _manager
