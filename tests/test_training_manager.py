import json
from src.training.config import TrainingConfig
from src.training.manager import TrainingManager


class FakeProc:
    def __init__(self, lines):
        self._lines = list(lines)
        self.stdout = self
        self._killed = False
    def readline(self):
        return self._lines.pop(0) if self._lines else ""
    def poll(self):
        return None if self._lines else 0
    def kill(self):
        self._killed = True
    def wait(self, timeout=None):
        return 0


class FakeEnv:
    def __init__(self, ready=True):
        self._ready = ready
    def ensure_ready(self, progress=None):
        return {"ready": self._ready, "error": None if self._ready else "no env"}
    def venv_python(self):
        return "venv/python"


def _cfg(tmp_path):
    ds = tmp_path / "d.jsonl"
    ds.write_text('{"text":"hi"}\n', encoding="utf-8")
    return TrainingConfig(base_model="x/Qwen2.5-0.5B", dataset_path=str(ds), steps=2)


def test_start_runs_sidecar_and_tracks_progress(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.manager.resolve_sidecar_script", lambda: "train.py")
    lines = [json.dumps({"event": "start", "total_steps": 2}) + "\n",
             json.dumps({"event": "step", "step": 1, "loss": 2.0, "vram_gb": 1.1}) + "\n",
             json.dumps({"event": "done", "output_dir": "o", "peak_vram_gb": 1.2}) + "\n"]
    captured = {}
    def spawn(argv):
        captured["argv"] = argv
        return FakeProc(lines)
    mgr = TrainingManager(env=FakeEnv(), spawn=spawn, free_vram=lambda: 6.4,
                          adapters_dir=str(tmp_path / "ad"))
    out = mgr.start(_cfg(tmp_path))
    assert out.get("started") is True
    # the pump thread runs to completion on the fake proc
    import time
    for _ in range(200):
        if mgr.status()["status"] in ("done", "error"):
            break
        time.sleep(0.01)
    assert "train.py" in captured["argv"] and "--config" in captured["argv"]
    st = mgr.status()
    assert st["status"] == "done" and st["last_step"] == 1 and st["peak_vram_gb"] == 1.2


def test_start_rejects_invalid_config(tmp_path):
    mgr = TrainingManager(env=FakeEnv(), spawn=lambda a: None, free_vram=lambda: 6.4)
    bad = TrainingConfig(base_model="", dataset_path="d.jsonl", steps=1)
    out = mgr.start(bad)
    assert "error" in out


def test_start_errors_when_env_not_ready(tmp_path):
    mgr = TrainingManager(env=FakeEnv(ready=False), spawn=lambda a: None, free_vram=lambda: 6.4)
    out = mgr.start(_cfg(tmp_path))
    assert "error" in out and "env" in out["error"].lower()


def test_start_never_raises_on_spawn_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.manager.resolve_sidecar_script", lambda: "train.py")
    def boom(argv):
        raise RuntimeError("cannot spawn")
    mgr = TrainingManager(env=FakeEnv(), spawn=boom, free_vram=lambda: 6.4,
                          adapters_dir=str(tmp_path / "ad"))
    out = mgr.start(_cfg(tmp_path))
    assert "error" in out


class BlockingProc:
    """A process whose readline blocks until kill(), then reports a nonzero exit."""
    def __init__(self):
        import threading as _t
        self._ev = _t.Event()
        self._killed = False
        self.stdout = self
    def readline(self):
        self._ev.wait()      # block until killed
        return ""            # then EOF
    def poll(self):
        return -9 if self._killed else None
    def kill(self):
        self._killed = True
        self._ev.set()
    def wait(self, timeout=None):
        return -9


def test_stop_status_not_overwritten_by_pump(tmp_path, monkeypatch):
    import time
    monkeypatch.setattr("src.training.manager.resolve_sidecar_script", lambda: "train.py")
    proc = BlockingProc()
    mgr = TrainingManager(env=FakeEnv(), spawn=lambda argv: proc, free_vram=lambda: 6.4,
                          adapters_dir=str(tmp_path / "ad"))
    assert mgr.start(_cfg(tmp_path)).get("started") is True
    time.sleep(0.05)                      # let the pump reach its blocking readline
    assert mgr.stop() == {"stopped": True}
    time.sleep(0.1)                       # let the pump run its finally after EOF
    assert mgr.status()["status"] == "stopped"   # NOT overwritten to "error"


def test_start_never_raises_when_free_vram_raises(tmp_path, monkeypatch):
    import time
    monkeypatch.setattr("src.training.manager.resolve_sidecar_script", lambda: "train.py")
    def boom_vram():
        raise RuntimeError("nvml down")
    lines = [json.dumps({"event": "done", "output_dir": "o", "peak_vram_gb": 1.0}) + "\n"]
    mgr = TrainingManager(env=FakeEnv(), spawn=lambda argv: FakeProc(lines),
                          free_vram=boom_vram, adapters_dir=str(tmp_path / "ad"))
    out = mgr.start(_cfg(tmp_path))
    assert out.get("started") is True     # a raising free_vram must not break start()
    for _ in range(200):
        if mgr.status()["status"] == "done":
            break
        time.sleep(0.01)
    assert mgr.status()["status"] == "done"


def test_list_adapters_never_raises_on_listdir_failure(tmp_path, monkeypatch):
    ad = tmp_path / "ad"
    ad.mkdir()
    mgr = TrainingManager(env=FakeEnv(), spawn=lambda a: None, free_vram=lambda: 6.4,
                          adapters_dir=str(ad))
    def boom(_):
        raise PermissionError("denied")
    monkeypatch.setattr("os.listdir", boom)
    assert mgr.list_adapters() == []   # must not raise
