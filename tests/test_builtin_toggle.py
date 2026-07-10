"""Task 1 (Plugin/Connector Hub): `disabled_builtin_mcp` setting + boot-skip.

A built-in MCP server (image_gen/memory/rag/email) listed in the
`disabled_builtin_mcp` setting must never be connected by
`register_builtin_servers` at boot.

Built-ins connect via fire-and-forget background tasks: `_spawn_bg` does
`asyncio.create_task(coro)` and `register_builtin_servers` returns immediately
after scheduling them, with no `await` point in between. So a naive
`asyncio.run(bm.register_builtin_servers(FakeMgr()))` would return before any
`connect_server` call actually runs, and the positive assertions below
(`memory`/`image_gen` present) would fail even with a correct implementation.
To make the test genuinely observe the connects, `_spawn_bg` is replaced with
a fake that schedules the same coroutines via `asyncio.ensure_future` into a
list the test can `await` after `register_builtin_servers` returns. The NPX
branch's startup delay (`asyncio.sleep(3)`) is neutralized so gathering that
task doesn't stall the test; `_find_npx` stays patched to None (as in the
task brief) so the NPX branch connects a different server_id
(`builtin_browser`) and cannot corrupt the `connected` list either way.
"""
import asyncio

import src.settings as settings
import src.builtin_mcp as bm


def test_default_disabled_builtin_is_empty():
    assert settings.DEFAULT_SETTINGS.get("disabled_builtin_mcp") == []


def test_register_skips_disabled_builtins(monkeypatch):
    """A built-in listed in disabled_builtin_mcp must never connect at boot."""
    monkeypatch.setattr(bm, "MCP_DISABLED", False, raising=False)
    monkeypatch.setattr(bm, "get_setting",
                        lambda k, d=None: ["rag", "email"] if k == "disabled_builtin_mcp" else d,
                        raising=False)
    connected = []

    class FakeMgr:
        async def connect_server(self, server_id, name, transport, command, args, env):
            connected.append(server_id)
            return True

    # Neutralize the NPX-server registration path so only builtins are exercised.
    monkeypatch.setattr(bm, "_find_npx", lambda: None, raising=False)

    # register_builtin_servers schedules its connect coroutines via _spawn_bg
    # and returns before any of them run (see module docstring above). Replace
    # _spawn_bg with a fake that schedules the same coroutines but lets the
    # test await them afterwards, so the assertions observe real
    # connect_server calls instead of an always-empty `connected` list.
    scheduled = []

    def fake_spawn_bg(coro):
        task = asyncio.ensure_future(coro)
        scheduled.append(task)
        return task

    monkeypatch.setattr(bm, "_spawn_bg", fake_spawn_bg)

    # Make the NPX branch's startup delay instant so awaiting its captured
    # task doesn't stall the test for real wall-clock seconds.
    async def instant_sleep(*_a, **_kw):
        return None

    monkeypatch.setattr(bm.asyncio, "sleep", instant_sleep)

    async def run():
        await bm.register_builtin_servers(FakeMgr())
        await asyncio.gather(*scheduled, return_exceptions=True)

    asyncio.run(run())

    assert "rag" not in connected and "email" not in connected
    assert "memory" in connected and "image_gen" in connected
