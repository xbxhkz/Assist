"""State-machine tests for ImageModelManager using injected fakes."""
import itertools

from src.imagemodels.manager import ImageModelManager

FILES = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
         "clip_l": "/m/clip.safetensors", "vae": "/m/ae.safetensors"}


class FakeProc:
    def __init__(self, pid=4321, exit_code=None):
        self.pid = pid
        self._exit_code = exit_code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._exit_code

    def terminate(self):
        self.terminated = True
        if self._exit_code is None:
            self._exit_code = -15

    def wait(self, timeout=None):
        pass

    def kill(self):
        self.killed = True
        if self._exit_code is None:
            self._exit_code = -9


def make_manager(ready=True, spawned=None, registered=None, unregistered=None,
                 proc_exit_code=None):
    spawned = spawned if spawned is not None else []
    registered = registered if registered is not None else []
    unregistered = unregistered if unregistered is not None else []

    def spawn(argv):
        p = FakeProc(exit_code=proc_exit_code)
        spawned.append((argv, p))
        return p

    def register(name, base_url):
        eid = f"img-local-{len(registered)}"
        registered.append({"name": name, "base_url": base_url, "id": eid})
        return eid

    clock = itertools.count(0, 10)
    mgr = ImageModelManager(
        spawn=spawn, port_chooser=lambda: 8200, probe=lambda url: ready,
        register_endpoint=register, unregister_endpoint=lambda eid: unregistered.append(eid),
        resolve_binary=lambda device: "/bin/sd-server",
        log_path="/nonexistent/sd-server.log",
        sleep=lambda _s: None, now=lambda: next(clock), ready_timeout=45.0)
    return mgr, spawned, registered, unregistered


def test_start_launches_and_registers():
    mgr, spawned, registered, _ = make_manager()
    st = mgr.start(FILES, device="cpu")
    assert st["running"] is True and st["model"] == "flux.gguf"
    assert st["port"] == 8200 and st["endpoint_id"] == "img-local-0"
    assert registered[0]["base_url"] == "http://127.0.0.1:8200/v1"


def test_start_readiness_failure_kills_and_raises():
    import pytest
    mgr, spawned, registered, _ = make_manager(ready=False)
    with pytest.raises(RuntimeError, match="did not become ready"):
        mgr.start(FILES)
    assert spawned[0][1].terminated is True
    assert registered == []
    assert mgr.status()["running"] is False


def test_start_fails_fast_on_early_exit():
    import pytest
    mgr, spawned, registered, _ = make_manager(ready=True, proc_exit_code=1)
    with pytest.raises(RuntimeError, match="exited on startup"):
        mgr.start(FILES)
    assert registered == []


def test_stop_terminates_and_unregisters():
    mgr, spawned, registered, unregistered = make_manager()
    mgr.start(FILES)
    st = mgr.stop()
    assert st["running"] is False
    assert spawned[0][1].terminated is True
    assert unregistered == ["img-local-0"]


def test_status_when_idle():
    mgr, *_ = make_manager()
    assert mgr.status() == {"running": False, "model": None, "port": None,
                            "endpoint_id": None, "device": None}


def test_ready_timeout_scales_with_size():
    mgr, *_ = make_manager()
    assert mgr._timeout_for_bytes(2_000_000) == 45.0
    assert mgr._timeout_for_bytes(12_000_000_000) == 12.0 * 12


def test_terminate_escalates_to_force_kill():
    killed = []

    class StubbornProc:
        pid = 9999
        def poll(self): return None
        def terminate(self): pass
        def wait(self, timeout=None): pass
        def kill(self): pass

    mgr = ImageModelManager(
        spawn=lambda argv: StubbornProc(), port_chooser=lambda: 8200,
        probe=lambda url: True, register_endpoint=lambda name, base_url: "e",
        unregister_endpoint=lambda eid: None, resolve_binary=lambda device: "/b",
        log_path="/nonexistent/sd-server.log", sleep=lambda _s: None,
        now=lambda: 0.0, force_kill=lambda pid: killed.append(pid))
    mgr.start(FILES)
    mgr.stop()
    assert 9999 in killed
