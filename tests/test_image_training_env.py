import os
from src.image_training.env import ImageTrainingEnv


class FakeBaseEnv:
    def __init__(self, ready=True, venv_dir=None):
        self._ready = ready
        self._venv_dir = venv_dir

    def ensure_ready(self, progress=None):
        return {"ready": self._ready, "error": None if self._ready else "base env not ready"}

    def venv_python(self):
        return os.path.join(self._venv_dir, "Scripts", "python.exe")

    def status(self):
        return "ready" if self._ready else "not_installed"


def test_status_not_installed_when_base_env_not_ready(tmp_path):
    env = ImageTrainingEnv(training_env=FakeBaseEnv(ready=False, venv_dir=str(tmp_path)))
    assert env.status() == "not_installed"


def test_status_not_installed_when_base_ready_but_diffusers_marker_missing(tmp_path):
    env = ImageTrainingEnv(training_env=FakeBaseEnv(ready=True, venv_dir=str(tmp_path)))
    assert env.status() == "not_installed"


def test_ensure_ready_installs_diffusers_and_marks_ready(tmp_path):
    calls = []

    def fake_run(argv):
        calls.append(argv)
        return (0, "ok")

    env = ImageTrainingEnv(training_env=FakeBaseEnv(ready=True, venv_dir=str(tmp_path)),
                           uv_binary="uv.exe", run=fake_run)
    out = env.ensure_ready()
    assert out == {"ready": True, "error": None}
    assert calls and calls[0][:2] == ["uv.exe", "pip"] and "diffusers" in calls[0]
    assert env.status() == "ready"


def test_ensure_ready_idempotent_skips_when_already_ready(tmp_path):
    env = ImageTrainingEnv(training_env=FakeBaseEnv(ready=True, venv_dir=str(tmp_path)),
                           uv_binary="uv.exe", run=lambda argv: (0, ""))
    env.ensure_ready()
    calls = []
    env2 = ImageTrainingEnv(training_env=FakeBaseEnv(ready=True, venv_dir=str(tmp_path)),
                            uv_binary="uv.exe", run=lambda argv: (calls.append(argv), (0, ""))[1])
    out = env2.ensure_ready()
    assert out == {"ready": True, "error": None}
    assert calls == []


def test_ensure_ready_propagates_base_env_error(tmp_path):
    env = ImageTrainingEnv(training_env=FakeBaseEnv(ready=False, venv_dir=str(tmp_path)),
                           uv_binary="uv.exe", run=lambda argv: (0, ""))
    out = env.ensure_ready()
    assert out["ready"] is False and "base env not ready" in out["error"]


def test_ensure_ready_never_raises_on_install_failure(tmp_path):
    env = ImageTrainingEnv(training_env=FakeBaseEnv(ready=True, venv_dir=str(tmp_path)),
                           uv_binary="uv.exe", run=lambda argv: (1, "boom"))
    out = env.ensure_ready()
    assert out["ready"] is False and "boom" in out["error"]
