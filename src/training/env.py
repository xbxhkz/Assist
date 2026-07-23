"""On-demand Python 3.11 CUDA training venv, set up via the vendored `uv`.

The main app (Py3.14) can't run torch; this builds a side venv with the CUDA
training stack the spike proved. Never raises — ensure_ready returns a status
dict. The `run` callable is injectable for tests."""
import os
import subprocess

PY_VERSION = "3.11"
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
STACK = ["transformers", "peft", "bitsandbytes", "accelerate", "datasets", "trl", "gguf==0.19.0"]


def _default_run(argv):
    p = subprocess.run(argv, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return p.returncode, (p.stdout or "") + (p.stderr or "")


class TrainingEnv:
    def __init__(self, base_dir=None, uv_binary=None, run=None):
        if base_dir is None:
            from src.constants import DATA_DIR
            base_dir = os.path.join(DATA_DIR, "training")
        self._base = base_dir
        self._venv = os.path.join(base_dir, "venv")
        self._uv = uv_binary
        self._run = run or _default_run

    def _uv_bin(self):
        if self._uv is None:
            from src.training.runtime import resolve_uv_binary
            self._uv = resolve_uv_binary()
        return self._uv

    def venv_python(self) -> str:
        sub = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
        return os.path.join(self._venv, *sub)

    def _marker(self) -> str:
        return os.path.join(self._venv, ".assist_training_ready")

    def status(self) -> str:
        return "ready" if (os.path.isfile(self.venv_python())
                           and os.path.isfile(self._marker())) else "not_installed"

    def ensure_ready(self, progress=None) -> dict:
        """Idempotently build the training venv. Returns {"ready","error"}. Never raises."""
        if self.status() == "ready":
            return {"ready": True, "error": None}
        try:
            uv = self._uv_bin()
            py = self.venv_python()
            steps = [
                [uv, "python", "install", PY_VERSION],
                [uv, "venv", "--python", PY_VERSION, self._venv],
                [uv, "pip", "install", "--python", py, "torch", "--index-url", TORCH_INDEX],
                [uv, "pip", "install", "--python", py] + STACK,
            ]
            os.makedirs(self._base, exist_ok=True)
            for argv in steps:
                if progress:
                    try:
                        progress({"event": "install", "cmd": " ".join(argv[:3])})
                    except Exception:
                        pass
                rc, out = self._run(argv)
                if rc != 0:
                    return {"ready": False, "error": f"uv step failed ({argv[1]}): {out[-500:]}"}
            if not os.path.isfile(py):
                return {"ready": False, "error": "venv python missing after setup"}
            open(self._marker(), "w").close()
            return {"ready": True, "error": None}
        except Exception as e:  # noqa: BLE001
            return {"ready": False, "error": f"training env setup failed: {e}"}
