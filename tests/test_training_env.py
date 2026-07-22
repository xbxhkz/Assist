import os
from src.training.env import TrainingEnv


def test_status_not_installed(tmp_path):
    env = TrainingEnv(base_dir=str(tmp_path), uv_binary="uv.exe")
    assert env.status() == "not_installed"


def test_ensure_ready_runs_uv_steps_then_marks_ready(tmp_path):
    calls = []

    def fake_run(argv):
        calls.append(argv)
        # simulate uv creating the venv python on the venv step
        if argv[:2] == ["uv.exe", "venv"]:
            os.makedirs(os.path.dirname(TrainingEnv(base_dir=str(tmp_path)).venv_python()),
                        exist_ok=True)
            open(TrainingEnv(base_dir=str(tmp_path)).venv_python(), "w").close()
        return (0, "ok")

    env = TrainingEnv(base_dir=str(tmp_path), uv_binary="uv.exe", run=fake_run)
    out = env.ensure_ready()
    assert out["ready"] is True and out["error"] is None
    # python install, venv, torch install, stack install
    kinds = [c[1] for c in calls]
    assert kinds[:2] == ["python", "venv"] and "pip" in kinds
    assert env.status() == "ready"


def test_ensure_ready_idempotent_skips_when_ready(tmp_path):
    env = TrainingEnv(base_dir=str(tmp_path), uv_binary="uv.exe",
                      run=lambda argv: (0, ""))
    os.makedirs(os.path.dirname(env.venv_python()), exist_ok=True)
    open(env.venv_python(), "w").close()
    open(env._marker(), "w").close()
    calls = []
    env2 = TrainingEnv(base_dir=str(tmp_path), uv_binary="uv.exe",
                       run=lambda argv: (calls.append(argv), (0, ""))[1])
    assert env2.ensure_ready()["ready"] is True
    assert calls == []            # nothing re-run


def test_ensure_ready_never_raises_on_failure(tmp_path):
    env = TrainingEnv(base_dir=str(tmp_path), uv_binary="uv.exe",
                      run=lambda argv: (1, "boom"))
    out = env.ensure_ready()
    assert out["ready"] is False and "boom" in (out["error"] or "")
