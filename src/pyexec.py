"""Frozen-safe way to run "python" from app code.

In dev, sys.executable is a real interpreter. In the frozen build it IS
Assist.exe — spawning it with python-style args boots a full second app
(which then autoserves the last LLM: the Cookbook's local scan forked the
whole stack this way). python_argv() routes frozen calls through the
launcher's --run-py dispatch (src.mcp_child_dispatch), which executes the
args like a python CLI instead of booting the app.
"""
import sys


def python_argv(*args) -> list:
    """argv that runs Python with `args`, safe in both dev and frozen."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "--run-py", *map(str, args)]
    return [sys.executable or "python", *map(str, args)]
