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


def test_null_writer_is_silent_and_not_a_tty():
    import launcher
    w = launcher.NullWriter()
    w.write("hello")  # must not raise
    w.flush()         # must not raise
    assert w.isatty() is False


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
    # Counter-based fake monotonic clock: advances by a fixed step on every
    # call and never exhausts, so the deadline is reliably crossed regardless
    # of exactly how many times the poll loop calls now().
    clock = {"t": 0.0}
    def now():
        clock["t"] += 0.1
        return clock["t"]
    def opener(url):
        raise ConnectionError("never up")
    ok = dr.wait_for_server_ready("http://x/api/health", timeout=1.0,
                                  interval=0.0, opener=opener,
                                  sleep=lambda _s: None,
                                  now=now)
    assert ok is False


def test_bundled_fastembed_cache_none_when_not_frozen(monkeypatch):
    monkeypatch.delattr(dr.sys, "frozen", raising=False)
    assert dr.bundled_fastembed_cache() is None


def test_bundled_fastembed_cache_returns_path_when_frozen(monkeypatch, tmp_path):
    cache = tmp_path / "fastembed_cache"
    cache.mkdir()
    monkeypatch.setattr(dr.sys, "frozen", True, raising=False)
    monkeypatch.setattr(dr.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert dr.bundled_fastembed_cache() == str(cache)


def test_bundled_fastembed_cache_none_when_dir_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(dr.sys, "frozen", True, raising=False)
    monkeypatch.setattr(dr.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert dr.bundled_fastembed_cache() is None  # no fastembed_cache subdir


def test_webview2_available_true_for_real_version():
    assert dr.webview2_runtime_available(read_pv=lambda: "121.0.2277.0") is True


def test_webview2_available_false_when_absent():
    assert dr.webview2_runtime_available(read_pv=lambda: None) is False


def test_webview2_available_false_for_zero_version():
    assert dr.webview2_runtime_available(read_pv=lambda: "0.0.0.0") is False


def test_launcher_imports_without_starting_gui():
    # Importing the launcher module must NOT start a server or window
    # (all side effects live under main()/__main__). It must expose main().
    import importlib
    launcher = importlib.import_module("launcher")
    assert hasattr(launcher, "main")
    assert callable(launcher.main)


def test_run_server_logging_errors_logs_exception(caplog):
    # A windowed (console-less) frozen build has no visible stdout/stderr, so
    # an exception on the background thread running the server (e.g. a
    # port-bind race between choose_port()'s check and the server's real
    # bind) would otherwise vanish with zero trace. This must land in the
    # log instead of being silently swallowed by the default thread
    # exception hook.
    class _FailingServer:
        def run(self):
            raise RuntimeError("boom")

    with caplog.at_level("ERROR"):
        dr.run_server_logging_errors(_FailingServer())

    assert "Background server thread failed" in caplog.text
    assert "boom" in caplog.text


def test_run_server_logging_errors_runs_normally_without_logging_error(caplog):
    calls = []

    class _OkServer:
        def run(self):
            calls.append(1)

    with caplog.at_level("ERROR"):
        dr.run_server_logging_errors(_OkServer())

    assert calls == [1]
    assert "Background server thread failed" not in caplog.text
