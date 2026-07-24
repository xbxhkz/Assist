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


def test_python_install_failure_is_not_fatal(tmp_path):
    """`uv python install` only adds a convenience minor-version link. On some
    Windows setups that link creation fails (os error 448, Redirection Guard)
    even though a usable Python 3.11 is present, and `uv venv --python` finds it
    anyway — so a failure there must NOT abort the whole setup."""
    calls = []

    def fake_run(argv):
        calls.append(argv)
        if argv[1] == "python":
            return (1, "error: Failed to create Python minor version link directory "
                       "(os error 448)")
        if argv[1] == "venv":
            p = TrainingEnv(base_dir=str(tmp_path)).venv_python()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").close()
        return (0, "ok")

    env = TrainingEnv(base_dir=str(tmp_path), uv_binary="uv.exe", run=fake_run)
    out = env.ensure_ready()
    assert out["ready"] is True, out
    # it continued past the failed optional step through both pip steps
    assert [c[1] for c in calls] == ["python", "venv", "pip", "pip"]


def test_venv_failure_is_fatal_and_reports_earlier_soft_failure(tmp_path):
    def fake_run(argv):
        if argv[1] == "python":
            return (1, "minor version link os error 448")
        return (1, "venv boom")

    env = TrainingEnv(base_dir=str(tmp_path), uv_binary="uv.exe", run=fake_run)
    out = env.ensure_ready()
    assert out["ready"] is False
    assert "venv boom" in out["error"] and "448" in out["error"]


def test_ensure_ready_never_raises_on_failure(tmp_path):
    env = TrainingEnv(base_dir=str(tmp_path), uv_binary="uv.exe",
                      run=lambda argv: (1, "boom"))
    out = env.ensure_ready()
    assert out["ready"] is False and "boom" in (out["error"] or "")
