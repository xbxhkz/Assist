"""Persist-last-model + auto-serve-on-startup behavior."""
import json

import src.localmodels.manager as mgr_mod
from src.localmodels.manager import LocalModelManager, autoserve_last_model


class FakeProc:
    def poll(self):
        return None  # running

    def terminate(self):
        pass

    def wait(self, timeout=None):
        pass

    def kill(self):
        pass


def _manager(tmp_path, monkeypatch):
    monkeypatch.setattr(mgr_mod, "MODELS_DIR", str(tmp_path / "models"))
    return LocalModelManager(
        spawn=lambda argv: FakeProc(),
        port_chooser=lambda: 8123,
        probe=lambda url: True,
        register_endpoint=lambda name, base_url: "local-0",
        unregister_endpoint=lambda eid: None,
        resolve_binary=lambda device="cpu": "/bin/llama-server",
        sleep=lambda _s: None,
        now=lambda: 0.0,
    )


def test_start_persists_last_model(tmp_path, monkeypatch):
    mgr = _manager(tmp_path, monkeypatch)
    (tmp_path / "models").mkdir()
    model = str(tmp_path / "models" / "m.gguf")
    open(model, "wb").close()
    mgr.start(model)
    f = tmp_path / "last_model.json"
    assert f.exists()
    assert json.loads(f.read_text())["model_path"] == model


def test_autoserve_starts_persisted_model(tmp_path, monkeypatch):
    monkeypatch.setattr(mgr_mod, "MODELS_DIR", str(tmp_path / "models"))
    (tmp_path / "models").mkdir()
    model = str(tmp_path / "models" / "m.gguf")
    open(model, "wb").close()
    (tmp_path / "last_model.json").write_text(json.dumps({"model_path": model}))

    started = {}

    class FakeMgr:
        def status(self):
            return {"running": False}

        def start(self, p, device="cpu"):
            started["p"] = p
            started["device"] = device
            return {"running": True}

    autoserve_last_model(manager=FakeMgr())
    assert started["p"] == model
    assert started["device"] == "cpu"  # no persisted device -> cpu default


def test_autoserve_skips_missing_model_file(tmp_path, monkeypatch):
    monkeypatch.setattr(mgr_mod, "MODELS_DIR", str(tmp_path / "models"))
    (tmp_path / "last_model.json").write_text(
        json.dumps({"model_path": str(tmp_path / "gone.gguf")}))

    class FakeMgr:
        def status(self):
            return {"running": False}

        def start(self, p):
            raise AssertionError("must not start a missing model")

    assert autoserve_last_model(manager=FakeMgr()) is None


def test_autoserve_noop_when_no_state(tmp_path, monkeypatch):
    monkeypatch.setattr(mgr_mod, "MODELS_DIR", str(tmp_path / "models"))

    class FakeMgr:
        def status(self):
            return {"running": False}

        def start(self, p):
            raise AssertionError("must not start with no saved model")

    assert autoserve_last_model(manager=FakeMgr()) is None


def test_autoserve_noop_when_already_running(tmp_path, monkeypatch):
    monkeypatch.setattr(mgr_mod, "MODELS_DIR", str(tmp_path / "models"))
    (tmp_path / "models").mkdir()
    model = str(tmp_path / "models" / "m.gguf")
    open(model, "wb").close()
    (tmp_path / "last_model.json").write_text(json.dumps({"model_path": model}))

    class FakeMgr:
        def status(self):
            return {"running": True}

        def start(self, p):
            raise AssertionError("must not start when one is already running")

    assert autoserve_last_model(manager=FakeMgr()) is None
