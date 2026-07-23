import json
import os
from src.training.convert_manager import AdapterConverter


class FakeEnv:
    def __init__(self, ready=True): self._ready = ready
    def ensure_ready(self, progress=None):
        return {"ready": self._ready, "error": None if self._ready else "no env"}
    def venv_python(self): return "venv/python"


def _adapter(tmp_path):
    d = tmp_path / "run-1"; d.mkdir()
    (d / "adapter_model.safetensors").write_text("x")
    (d / "run_config.json").write_text(json.dumps({"base_model": "x/Qwen2.5-0.5B"}))
    return str(d)


def test_convert_success(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.convert_manager.resolve_convert_script", lambda: "convert.py")
    d = _adapter(tmp_path)
    def spawn(argv):
        # simulate the sidecar writing adapter.gguf + emitting a done line
        open(os.path.join(d, "adapter.gguf"), "w").close()
        return (0, json.dumps({"event": "done", "adapter_gguf": os.path.join(d, "adapter.gguf")}))
    conv = AdapterConverter(env=FakeEnv(), spawn=spawn)
    out = conv.convert(d)
    assert out.get("ok") is True and out["adapter_gguf"].endswith("adapter.gguf")


def test_convert_env_not_ready(tmp_path):
    conv = AdapterConverter(env=FakeEnv(ready=False), spawn=lambda a: (0, ""))
    out = conv.convert(_adapter(tmp_path))
    assert "error" in out and "env" in out["error"].lower()


def test_convert_sidecar_error(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.convert_manager.resolve_convert_script", lambda: "convert.py")
    def spawn(argv): return (1, json.dumps({"event": "error", "message": "bad arch"}))
    conv = AdapterConverter(env=FakeEnv(), spawn=spawn)
    out = conv.convert(_adapter(tmp_path))
    assert "error" in out and "bad arch" in out["error"]


def test_convert_never_raises_on_spawn_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.convert_manager.resolve_convert_script", lambda: "convert.py")
    def boom(argv): raise RuntimeError("cannot spawn")
    conv = AdapterConverter(env=FakeEnv(), spawn=boom)
    out = conv.convert(_adapter(tmp_path))
    assert "error" in out
