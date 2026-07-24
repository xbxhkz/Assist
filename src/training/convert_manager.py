"""Orchestrate the LoRA->GGUF conversion sidecar from the Py3.14 app. Never
imports the conversion stack; never raises. Blocking (call via asyncio.to_thread)."""
import json
import os
import subprocess

from src.training.runtime import resolve_convert_script


def _default_spawn(argv):
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return p.returncode, (p.stdout or "") + (p.stderr or "")


class AdapterConverter:
    def __init__(self, env=None, spawn=None):
        if env is None:
            from src.training.env import TrainingEnv
            env = TrainingEnv()
        self._env = env
        self._spawn = spawn or _default_spawn

    def convert(self, adapter_dir, base=None) -> dict:
        try:
            if not (isinstance(adapter_dir, str) and os.path.isdir(adapter_dir)):
                return {"error": "adapter directory not found"}
            ready = self._env.ensure_ready()
            if not ready.get("ready"):
                return {"error": f"training env not ready: {ready.get('error')}"}
            argv = [self._env.venv_python(), resolve_convert_script(), "--adapter", adapter_dir]
            if base:
                argv += ["--base", base]
            rc, out = self._spawn(argv)
            ev = {}
            for line in (out or "").splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        ev = json.loads(line)
                    except Exception:
                        pass
            if ev.get("event") == "done" and ev.get("adapter_gguf"):
                return {"ok": True, "adapter_gguf": ev["adapter_gguf"]}
            if ev.get("event") == "error":
                return {"error": ev.get("message", "conversion failed")}
            gguf = os.path.join(adapter_dir, "adapter.gguf")
            if rc == 0 and os.path.isfile(gguf):
                return {"ok": True, "adapter_gguf": gguf}
            return {"error": "conversion failed: " + (out or "")[-500:]}
        except Exception as e:  # noqa: BLE001
            return {"error": f"conversion error: {e}"}


_converter = None


def get_adapter_converter():
    global _converter
    if _converter is None:
        _converter = AdapterConverter()
    return _converter
