# tests/test_image_training_manager.py
import json
import os
from src.image_training.config import ImageTrainingConfig
from src.image_training.manager import ImageTrainingManager


class FakeProc:
    def __init__(self, lines):
        self._lines = list(lines)
        self.stdout = self

    def readline(self):
        return self._lines.pop(0) if self._lines else ""

    def poll(self):
        return None if self._lines else 0

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


class FakeEnv:
    def __init__(self, ready=True):
        self._ready = ready

    def ensure_ready(self):
        return {"ready": self._ready, "error": None if self._ready else "no env"}

    def venv_python(self):
        return "venv/python"

    def status(self):
        return "ready" if self._ready else "not_installed"


class FakeStore:
    def __init__(self, images=None, path="/ds", trigger_word="mytrigger", error=None):
        self._images = images if images is not None else [{"filename": "0000.png", "caption": "a cat"}]
        self._path = path
        self._trigger_word = trigger_word
        self._error = error

    def load(self, name):
        if self._error:
            return {"error": self._error}
        return {"name": name, "path": self._path, "trigger_word": self._trigger_word,
                "images": self._images}


def _cfg():
    return ImageTrainingConfig(dataset_name="ds1", output_name="my-lora", steps=2)


def _mgr(tmp_path, **kw):
    kw.setdefault("env", FakeEnv())
    kw.setdefault("spawn", lambda a: FakeProc([]))
    kw.setdefault("dataset_store", FakeStore())
    kw.setdefault("runs_dir", str(tmp_path / "runs"))
    kw.setdefault("loras_dir", lambda: str(tmp_path / "loras"))
    return ImageTrainingManager(**kw)


def test_start_runs_sidecar_and_tracks_progress(tmp_path):
    lines = [json.dumps({"event": "start", "total_steps": 2}) + "\n",
             json.dumps({"event": "step", "step": 1, "loss": 0.5, "vram_gb": 5.0}) + "\n",
             json.dumps({"event": "done", "lora_path": "x.safetensors", "peak_vram_gb": 5.9}) + "\n"]
    captured = {}

    def spawn(argv):
        captured["argv"] = argv
        return FakeProc(lines)

    mgr = _mgr(tmp_path, spawn=spawn)
    out = mgr.start(_cfg())
    assert out.get("started") is True
    import time
    for _ in range(200):
        if mgr.status()["status"] in ("done", "error"):
            break
        time.sleep(0.01)
    assert "--config" in captured["argv"]
    st = mgr.status()
    assert st["status"] == "done" and st["last_step"] == 1 and st["peak_vram_gb"] == 5.9


def test_start_prepends_trigger_word_to_captions_and_targets_loras_dir(tmp_path):
    captured = {}

    def spawn(argv):
        captured["argv"] = argv
        return FakeProc([])

    mgr = _mgr(tmp_path, spawn=spawn)
    mgr.start(_cfg())
    cfg_path = [a for a in captured["argv"] if a.endswith("config.json")][0]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["images"][0]["caption"] == "mytrigger, a cat"
    assert cfg["lora_path"] == os.path.join(str(tmp_path / "loras"), "my-lora.safetensors")


def test_start_rejects_invalid_config(tmp_path):
    mgr = _mgr(tmp_path)
    bad = ImageTrainingConfig(dataset_name="", output_name="", steps=1)
    out = mgr.start(bad)
    assert "error" in out


def test_start_errors_when_dataset_missing(tmp_path):
    mgr = _mgr(tmp_path, dataset_store=FakeStore(error="dataset not found"))
    out = mgr.start(_cfg())
    assert "error" in out and "dataset" in out["error"].lower()


def test_start_errors_when_dataset_has_no_images(tmp_path):
    mgr = _mgr(tmp_path, dataset_store=FakeStore(images=[]))
    out = mgr.start(_cfg())
    assert "error" in out


def test_start_errors_when_env_not_ready(tmp_path):
    mgr = _mgr(tmp_path, env=FakeEnv(ready=False))
    out = mgr.start(_cfg())
    assert "error" in out and "env" in out["error"].lower()


def test_start_never_raises_on_spawn_failure(tmp_path):
    def boom(argv):
        raise RuntimeError("cannot spawn")
    mgr = _mgr(tmp_path, spawn=boom)
    out = mgr.start(_cfg())
    assert "error" in out


def test_start_never_raises_when_dataset_store_raises(tmp_path):
    class BoomStore:
        def load(self, name):
            raise RuntimeError("disk error")
    mgr = _mgr(tmp_path, dataset_store=BoomStore())
    out = mgr.start(_cfg())
    assert "error" in out


def test_stop_when_not_running(tmp_path):
    mgr = _mgr(tmp_path)
    assert mgr.stop() == {"stopped": False}


class BlockingProc:
    """A process whose readline blocks until kill(), then reports a
    nonzero exit -- deterministic stand-in for "still running", unlike
    racing on how fast a fake stdout drains."""
    def __init__(self):
        import threading as _t
        self._ev = _t.Event()
        self._killed = False
        self.stdout = self

    def readline(self):
        self._ev.wait()
        return ""

    def poll(self):
        return -9 if self._killed else None

    def kill(self):
        self._killed = True
        self._ev.set()

    def wait(self, timeout=None):
        return -9


def test_second_start_rejected_while_running(tmp_path):
    mgr = _mgr(tmp_path, spawn=lambda a: BlockingProc())
    assert mgr.start(_cfg()).get("started") is True
    out = mgr.start(_cfg())
    assert "error" in out and "already in progress" in out["error"]
