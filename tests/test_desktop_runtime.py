"""Unit tests for the pure desktop-launcher helpers (no GUI, no network)."""
import socket

import src.desktop_runtime as dr


def test_choose_port_returns_preferred_when_free():
    # Grab a definitely-free port, release it, then ask for it.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    free = s.getsockname()[1]
    s.close()
    assert dr.choose_port(free) == free


def test_choose_port_falls_back_when_preferred_busy():
    # Occupy a port, then ask for it — must get a different, usable port.
    busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    busy_port = busy.getsockname()[1]
    try:
        chosen = dr.choose_port(busy_port)
        assert chosen != busy_port
        # The chosen port must itself be bindable.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", chosen))
        probe.close()
    finally:
        busy.close()


def test_local_origin_format():
    assert dr.local_origin(7000) == "http://127.0.0.1:7000"


def test_augment_allowed_origins_appends_when_missing():
    out = dr.augment_allowed_origins("http://127.0.0.1:7000",
                                     existing="http://localhost,http://127.0.0.1")
    parts = out.split(",")
    assert "http://127.0.0.1:7000" in parts
    assert "http://localhost" in parts  # existing preserved


def test_augment_allowed_origins_no_duplicate():
    out = dr.augment_allowed_origins("http://127.0.0.1:7000",
                                     existing="http://127.0.0.1:7000")
    assert out.split(",").count("http://127.0.0.1:7000") == 1


def test_make_uvicorn_server_config():
    async def dummy_app(scope, receive, send):  # minimal ASGI callable
        pass
    server = dr.make_uvicorn_server(dummy_app, "127.0.0.1", 7123)
    assert server.config.host == "127.0.0.1"
    assert server.config.port == 7123
    assert server.should_exit is False


def test_wait_for_server_ready_succeeds_after_retries():
    calls = {"n": 0}
    def opener(url):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("not up yet")
        return 200
    ok = dr.wait_for_server_ready("http://x/api/health", timeout=10.0,
                                  interval=0.0, opener=opener,
                                  sleep=lambda _s: None,
                                  now=lambda: 0.0)
    assert ok is True
    assert calls["n"] == 3


def test_wait_for_server_ready_times_out():
    times = iter([0.0, 0.1, 0.2, 999.0])
    def opener(url):
        raise ConnectionError("never up")
    ok = dr.wait_for_server_ready("http://x/api/health", timeout=1.0,
                                  interval=0.0, opener=opener,
                                  sleep=lambda _s: None,
                                  now=lambda: next(times))
    assert ok is False
