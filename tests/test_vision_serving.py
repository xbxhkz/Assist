"""ensure_vision_served: auto-serve the configured vision model as a dedicated
local endpoint when it isn't already reachable, so a tool-calling chat model
and a vision model can run together (screen reading)."""
import pytest

import src.localmodels.manager as mgr


class FakeVisionMgr:
    def __init__(self, running=False):
        self._running = running
        self.started = None
    def status(self):
        return {"running": self._running}
    def start(self, path, device="cpu"):
        self.started = (path, device)
        self._running = True
        return {"running": True}


def test_noop_when_already_resolvable(tmp_path):
    """If the vision model already has an endpoint, don't serve a duplicate."""
    fm = FakeVisionMgr()
    got = mgr.ensure_vision_served(
        "vl.gguf", resolve=lambda n, o: ("http://x/v1", "vl.gguf", {}),
        manager=fm, models_dir=str(tmp_path))
    assert got == ("http://x/v1", "vl.gguf", {})
    assert fm.started is None  # never served


def test_serves_local_gguf_when_unresolved(tmp_path):
    (tmp_path / "vl.gguf").write_bytes(b"x")
    fm = FakeVisionMgr()
    calls = {"n": 0}
    def resolve(name, owner):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("not served")
        return ("http://127.0.0.1:8110/v1", "vl.gguf", {})
    got = mgr.ensure_vision_served("vl.gguf", resolve=resolve, manager=fm,
                                   models_dir=str(tmp_path))
    assert fm.started == (str(tmp_path / "vl.gguf"), "cpu")  # CPU keeps GPU for chat
    assert got[0].endswith("8110/v1")


def test_missing_gguf_raises(tmp_path):
    fm = FakeVisionMgr()
    with pytest.raises(ValueError, match="not served"):
        mgr.ensure_vision_served(
            "gone.gguf", resolve=lambda n, o: (_ for _ in ()).throw(ValueError("x")),
            manager=fm, models_dir=str(tmp_path))


def test_does_not_restart_if_vision_mgr_already_running(tmp_path):
    (tmp_path / "vl.gguf").write_bytes(b"x")
    fm = FakeVisionMgr(running=True)
    calls = {"n": 0}
    def resolve(name, owner):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("chat endpoint doesn't have it")
        return ("http://127.0.0.1:8110/v1", "vl.gguf", {})
    mgr.ensure_vision_served("vl.gguf", resolve=resolve, manager=fm,
                             models_dir=str(tmp_path))
    assert fm.started is None  # already running -> reused, not restarted
