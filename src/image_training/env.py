"""On-demand extension of the existing training sidecar venv with
`diffusers`, for SDXL image-LoRA training. Reuses the SAME Py3.11 CUDA
venv the text-LoRA trainer provisions (torch/peft/bitsandbytes/accelerate
already installed there) instead of standing up a second multi-gigabyte
CUDA venv -- diffusers is the only package this needs on top. Never
raises."""
import os

STACK = ["diffusers"]


def _default_run(argv):
    import subprocess
    p = subprocess.run(argv, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return p.returncode, (p.stdout or "") + (p.stderr or "")


class ImageTrainingEnv:
    def __init__(self, training_env=None, uv_binary=None, run=None):
        if training_env is None:
            from src.training.env import TrainingEnv
            training_env = TrainingEnv()
        self._env = training_env
        self._uv = uv_binary
        self._run = run or _default_run

    def venv_python(self) -> str:
        return self._env.venv_python()

    def _marker(self) -> str:
        venv_dir = os.path.dirname(os.path.dirname(self.venv_python()))
        return os.path.join(venv_dir, ".assist_image_training_ready")

    def status(self) -> str:
        if self._env.status() != "ready":
            return "not_installed"
        return "ready" if os.path.isfile(self._marker()) else "not_installed"

    def _uv_bin(self):
        if self._uv is None:
            from src.training.runtime import resolve_uv_binary
            self._uv = resolve_uv_binary()
        return self._uv

    def ensure_ready(self, progress=None) -> dict:
        """Idempotently ensure the base training venv exists AND has
        `diffusers` installed. Returns {"ready", "error"}. Never raises."""
        base = self._env.ensure_ready(progress=progress)
        if not base.get("ready"):
            return base
        if self.status() == "ready":
            return {"ready": True, "error": None}
        try:
            uv = self._uv_bin()
            py = self.venv_python()
            if progress:
                try:
                    progress({"event": "install", "cmd": "pip diffusers"})
                except Exception:
                    pass
            rc, out = self._run([uv, "pip", "install", "--python", py] + STACK)
            if rc != 0:
                return {"ready": False, "error": f"diffusers install failed: {out[-500:]}"}
            os.makedirs(os.path.dirname(self._marker()), exist_ok=True)
            open(self._marker(), "w").close()
            return {"ready": True, "error": None}
        except Exception as e:  # noqa: BLE001
            return {"ready": False, "error": f"image training env setup failed: {e}"}
