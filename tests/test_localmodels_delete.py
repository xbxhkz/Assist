"""delete_model + disk-usage tests."""
import pytest

import src.localmodels.manager as mgr_mod
from src.localmodels.manager import LocalModelManager


class FakeProc:
    def __init__(self, pid=1):
        self.pid = pid
    def terminate(self): pass
    def wait(self, timeout=None): pass
    def kill(self): pass


def _serving_manager(tmp_path, monkeypatch):
    monkeypatch.setattr(mgr_mod, "MODELS_DIR", str(tmp_path))
    unreg = []
    mgr = LocalModelManager(
        spawn=lambda fn: None,           # we set state manually below
        port_chooser=lambda: 8123,
        probe=lambda url: True,
        register_endpoint=lambda name, base_url: "local-0",
        unregister_endpoint=lambda eid: unreg.append(eid),
        resolve_binary=lambda: "/bin/llama-server",
    )
    return mgr, unreg


def test_delete_rejects_unsafe(tmp_path, monkeypatch):
    mgr, _ = _serving_manager(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        mgr.delete_model("../evil.gguf")
    with pytest.raises(ValueError):
        mgr.delete_model("a/b.gguf")
    with pytest.raises(ValueError):
        mgr.delete_model("note.txt")


def test_delete_removes_file(tmp_path, monkeypatch):
    mgr, _ = _serving_manager(tmp_path, monkeypatch)
    (tmp_path / "m.gguf").write_bytes(b"xxxx")
    st = mgr.delete_model("m.gguf")
    assert not (tmp_path / "m.gguf").exists()
    assert st["running"] is False


def test_delete_stops_serving_model_first(tmp_path, monkeypatch):
    mgr, unreg = _serving_manager(tmp_path, monkeypatch)
    (tmp_path / "m.gguf").write_bytes(b"xxxx")
    # Simulate m.gguf currently serving.
    mgr._proc = FakeProc()
    mgr._state = {"model_path": str(tmp_path / "m.gguf"), "port": 8123,
                  "endpoint_id": "local-0", "pid": 1}
    mgr.delete_model("m.gguf")
    assert unreg == ["local-0"]                 # endpoint torn down
    assert not (tmp_path / "m.gguf").exists()   # file removed
    assert mgr.status()["running"] is False


def test_delete_missing_file_is_ok(tmp_path, monkeypatch):
    mgr, _ = _serving_manager(tmp_path, monkeypatch)
    st = mgr.delete_model("nope.gguf")          # no error
    assert st["running"] is False
