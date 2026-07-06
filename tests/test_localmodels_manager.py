"""State-machine tests for LocalModelManager using injected fakes (no real
process, port, or DB)."""
import itertools

from src.localmodels.manager import LocalModelManager


class FakeProc:
    def __init__(self, pid=4321, exit_code=None):
        self.pid = pid
        self._exit_code = exit_code  # None = still running
        self.terminated = False
        self.waited = False
        self.killed = False
        self.wait_timeout = None
    def poll(self):
        return self._exit_code
    def terminate(self):
        self.terminated = True
    def wait(self, timeout=None):
        self.waited = True
        self.wait_timeout = timeout
    def kill(self):
        self.killed = True


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
        eid = f"local-{len(registered)}"
        registered.append({"name": name, "base_url": base_url, "id": eid})
        return eid

    def unregister(endpoint_id):
        unregistered.append(endpoint_id)

    clock = itertools.count(0, 10)  # fast fake clock so timeout loops don't wait
    mgr = LocalModelManager(
        spawn=spawn,
        port_chooser=lambda: 8123,
        probe=lambda url: ready,
        register_endpoint=register,
        unregister_endpoint=unregister,
        resolve_binary=lambda: "/bin/llama-server",
        log_path="/nonexistent/llama-server.log",  # no real log in tests
        sleep=lambda _s: None,
        now=lambda: next(clock),
        ready_timeout=45.0,
    )
    return mgr, spawned, registered, unregistered


def test_ready_timeout_scales_with_model_size():
    """A 45s cap is fine for the small bundled model but far too short for a
    large one (a 48GB model takes minutes to load), so the readiness timeout
    scales with file size while keeping the base value as a floor."""
    mgr, *_ = make_manager()
    assert mgr._timeout_for_bytes(2_000_000) == 45.0             # tiny -> floor
    assert mgr._timeout_for_bytes(48_000_000_000) == 48.0 * 12   # 48GB -> 576s


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
    with pytest.raises(RuntimeError, match="did not become ready"):
        mgr.start("/models/m.gguf")
    assert spawned[0][1].terminated is True   # process killed
    assert registered == []                    # no dead endpoint registered
    assert mgr.status()["running"] is False


def test_start_fails_fast_on_early_process_exit():
    """A server that exits during startup (e.g. unsupported model architecture)
    is detected immediately — not after the full readiness timeout — and the
    error names the likely cause."""
    import pytest
    # probe would say "up" if asked, but the process has already exited, so the
    # proc-liveness check must short-circuit before probing.
    mgr, spawned, registered, _ = make_manager(ready=True, proc_exit_code=1)
    with pytest.raises(RuntimeError, match="exited on startup"):
        mgr.start("/models/bad.gguf")
    assert registered == []
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


def test_stop_waits_for_process_exit():
    mgr, spawned, registered, _ = make_manager()
    mgr.start("/models/a.gguf")
    mgr.stop()
    proc = spawned[0][1]
    assert proc.terminated is True
    assert proc.waited is True
