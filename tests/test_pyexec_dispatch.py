"""Frozen-safe "run python" plumbing.

Every `sys.executable <py-args>` call in app code forks the FULL APP in the
frozen build (sys.executable IS Assist.exe) — opening the Cookbook spawned a
second Assist + a second llama-server via its local dependency scan. All
call sites now go through python_argv(), and the launcher dispatches
`--run-py` like a real python CLI for the subset we use (-c / -m / script).
"""
import sys

import pytest

import src.pyexec as pyexec
import src.mcp_child_dispatch as dispatch


def test_python_argv_dev_mode():
    assert pyexec.python_argv("-I", "-c", "print(1)") == \
        [sys.executable or "python", "-I", "-c", "print(1)"]


def test_python_argv_frozen_mode(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    argv = pyexec.python_argv("script.py", "arg1")
    assert argv == [sys.executable, "--run-py", "script.py", "arg1"]


def test_dispatch_run_py_dash_c(tmp_path):
    marker = tmp_path / "c.txt"
    code = f"open(r'{marker}', 'w').write('ran')"
    with pytest.raises(SystemExit) as e:
        dispatch.maybe_dispatch(["Assist.exe", "--run-py", "-I", "-c", code])
    assert e.value.code == 0
    assert marker.read_text() == "ran"


def test_dispatch_run_py_script_with_args(tmp_path):
    marker = tmp_path / "s.txt"
    script = tmp_path / "s.py"
    script.write_text("import sys\nopen(sys.argv[1], 'w').write(sys.argv[2])\n")
    with pytest.raises(SystemExit) as e:
        dispatch.maybe_dispatch(
            ["Assist.exe", "--run-py", str(script), str(marker), "hello"])
    assert e.value.code == 0
    assert marker.read_text() == "hello"


def test_dispatch_run_py_module():
    # `-m platform` prints the platform string and exits 0 — a stdlib module
    # that is safe to execute and proves runpy module dispatch works.
    with pytest.raises(SystemExit) as e:
        dispatch.maybe_dispatch(["Assist.exe", "--run-py", "-m", "platform"])
    assert e.value.code == 0


def test_dispatch_run_py_propagates_exit_code():
    with pytest.raises(SystemExit) as e:
        dispatch.maybe_dispatch(
            ["Assist.exe", "--run-py", "-c", "import sys; sys.exit(3)"])
    assert e.value.code == 3


def test_dispatch_run_py_error_exits_nonzero(tmp_path):
    with pytest.raises(SystemExit) as e:
        dispatch.maybe_dispatch(
            ["Assist.exe", "--run-py", str(tmp_path / "missing.py")])
    assert e.value.code == 1


def test_normal_argv_untouched():
    assert dispatch.maybe_dispatch(["Assist.exe"]) is False


def test_frozen_reachable_call_sites_use_python_argv():
    """Grep-guard: the known frozen-reachable spawn sites must not call
    sys.executable directly."""
    import pathlib
    repo = pathlib.Path(pyexec.__file__).resolve().parents[1]
    for rel in ("routes/cookbook_routes.py", "src/agent_tools/subprocess_tools.py",
                "routes/shell_routes.py"):
        text = (repo / rel).read_text(encoding="utf-8")
        assert "python_argv" in text, f"{rel} does not use python_argv"
