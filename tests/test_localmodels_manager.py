"""State-machine tests for LocalModelManager using injected fakes (no real
process, port, or DB)."""
from src.localmodels.manager import LocalModelManager


class FakeProc:
    def __init__(self, pid=4321):
        self.pid = pid
        self.terminated = False
    def terminate(self):
        self.terminated = True


def make_manager(ready=True, spawned=None, registered=None, unregistered=None):
    spawned = spawned if spawned is not None else []
    registered = registered if registered is not None else []
    unregistered = unregistered if unregistered is not None else []

    def spawn(argv):
        p = FakeProc()
        spawned.append((argv, p))
        return p

    def register(name, base_url):
        eid = f"local-{len(registered)}"
        registered.append({"name": name, "base_url": base_url, "id": eid})
        return eid

    def unregister(endpoint_id):
        unregistered.append(endpoint_id)

    mgr = LocalModelManager(
        spawn=spawn,
        port_chooser=lambda: 8123,
        readiness=lambda url: ready,
        register_endpoint=register,
        unregister_endpoint=unregister,
        resolve_binary=lambda: "/bin/llama-server",
    )
    return mgr, spawned, registered, unregistered


def test_start_launches_and_registers():
    mgr, spawned, registered, _ = make_manager()
    st = mgr.start("/models/m.gguf")
    assert st == {"running": True, "model": "m.gguf", "port": 8123,
                  "endpoint_id": "local-0"}
    assert len(spawned) == 1
    assert registered[0]["base_url"] == "http://127.0.0.1:8123/v1"


def test_start_readiness_failure_kills_and_raises():
    import pytest
    mgr, spawned, registered, _ = make_manager(ready=False)
    with pytest.raises(RuntimeError):
        mgr.start("/models/m.gguf")
    assert spawned[0][1].terminated is True   # process killed
    assert registered == []                    # no dead endpoint registered
    assert mgr.status()["running"] is False


def test_start_twice_stops_previous_first():
    mgr, spawned, registered, unregistered = make_manager()
    mgr.start("/models/a.gguf")
    mgr.start("/models/b.gguf")
    assert spawned[0][1].terminated is True     # first process stopped
    assert unregistered == ["local-0"]          # first endpoint removed
    assert mgr.status()["model"] == "b.gguf"


def test_stop_terminates_and_unregisters():
    mgr, spawned, registered, unregistered = make_manager()
    mgr.start("/models/a.gguf")
    st = mgr.stop()
    assert st["running"] is False
    assert spawned[0][1].terminated is True
    assert unregistered == ["local-0"]


def test_status_when_idle():
    mgr, *_ = make_manager()
    assert mgr.status() == {"running": False, "model": None, "port": None,
                            "endpoint_id": None}
