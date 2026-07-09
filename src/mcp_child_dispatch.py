"""Dispatch python-mode invocations of the frozen exe to code, not the app.

The frozen build's sys.executable IS the app. Anything that spawns "python"
(built-in MCP servers, the agent's python tool, the Cookbook's dependency
scan, pip) would boot a full second app — which then registers MCP servers
and autoserves the last LLM, recursively (observed live: 21 Assist.exe
processes; a Cookbook open spawning a second llama-server). The launcher
calls maybe_dispatch(sys.argv) before booting anything.

Two modes:
  --run-mcp <script>            (legacy) run an MCP server script
  --run-py [flags] <target>     python-CLI subset: -c <code> | -m <module>
                                | <script> [args]; -I/-u/-B/-s/-E ignored
"""
import runpy
import sys
import traceback

_IGNORED_PY_FLAGS = {"-I", "-u", "-B", "-s", "-E"}


def _run_python_args(args) -> int:
    """Execute python-CLI style `args`; return an exit code."""
    i = 0
    while i < len(args) and args[i] in _IGNORED_PY_FLAGS:
        i += 1
    if i >= len(args):
        print("--run-py: nothing to run", file=sys.stderr)
        return 2
    head, rest = args[i], list(args[i + 1:])
    try:
        if head == "-c":
            if not rest:
                print("--run-py: -c needs code", file=sys.stderr)
                return 2
            sys.argv = ["-c", *rest[1:]]
            exec(compile(rest[0], "<string>", "exec"), {"__name__": "__main__"})
        elif head == "-m":
            if not rest:
                print("--run-py: -m needs a module", file=sys.stderr)
                return 2
            sys.argv = [rest[0], *rest[1:]]
            runpy.run_module(rest[0], run_name="__main__", alter_sys=True)
        else:
            sys.argv = [head, *rest]
            runpy.run_path(head, run_name="__main__")
    except SystemExit as e:
        code = e.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


def maybe_dispatch(argv) -> bool:
    """True when argv requested a child mode (caller must exit afterwards).

    Never falls through to the app on a dispatch request — even a missing
    script returns True/exits, so a bad path can't re-boot the full app.
    --run-py exits the process itself so callers see the real exit code.
    """
    if len(argv) >= 2 and argv[1] == "--run-py":
        sys.exit(_run_python_args(list(argv[2:])))
    if len(argv) < 3 or argv[1] != "--run-mcp":
        return False
    script = argv[2]
    try:
        runpy.run_path(script, run_name="__main__")
    except FileNotFoundError:
        print(f"MCP child: script not found: {script}", file=sys.stderr)
    except SystemExit:
        pass
    return True
