import asyncio
from unittest.mock import patch

from src.mcp_manager import _format_mcp_connection_error, McpManager


def _raise_stop_here(*_args, **_kwargs):
    """Aborts _connect_stdio right after StdioServerParameters is built, so
    these tests never need to simulate a real MCP handshake -- connect_server's
    own try/except turns this into a clean `False` return, and by then the
    params spy below has already captured what we're checking."""
    raise RuntimeError("stop-here")


def test_playwright_mcp_connection_error_includes_install_hint():
    msg = _format_mcp_connection_error(
        "Browser (Playwright)",
        "npx",
        ["-y", "@playwright/mcp@latest", "--headless"],
        RuntimeError("package not found"),
    )

    assert "package not found" in msg
    assert "Browser MCP could not start" in msg
    assert "npx -y @playwright/mcp@latest --version" in msg
    assert "restart Odysseus" in msg


def test_generic_mcp_connection_error_preserves_original_error():
    msg = _format_mcp_connection_error(
        "Custom MCP",
        "python",
        ["server.py"],
        RuntimeError("boom"),
    )

    assert msg == "boom"


def test_http_transport_routes_to_start_http_connect():
    mgr = McpManager()

    async def fake_start(server_id, name, url):
        return "ROUTED"

    with patch.object(McpManager, "_start_http_connect", side_effect=fake_start) as m:
        result = asyncio.run(mgr.connect_server("id1", "n", "http", url="https://x/mcp"))
    assert result == "ROUTED"
    m.assert_called_once()


def test_stdio_connect_defaults_cwd_to_data_dir_when_not_given():
    """Root cause of a real user-reported bug: builtin_browser (Playwright's
    @playwright/mcp) is spawned via connect_server with no cwd, so it
    inherited THIS process's cwd -- {app}=Program Files for the installed
    frozen build, unwritable by a standard user -- and browser_take_screenshot
    crashed with a Windows EPERM trying to save its output there. Every
    stdio MCP child must get an explicit, writable cwd by default."""
    from src.constants import DATA_DIR
    import mcp

    captured = {}
    real_init = mcp.StdioServerParameters.__init__

    def spy_init(self, **kwargs):
        captured.update(kwargs)
        real_init(self, **kwargs)

    mgr = McpManager()
    with patch.object(mcp.StdioServerParameters, "__init__", spy_init), \
         patch("mcp.client.stdio.stdio_client", _raise_stop_here), \
         patch("src.mcp_manager._safe_errlog", return_value=None):
        result = asyncio.run(mgr.connect_server(
            "id1", "n", "stdio", command="npx", args=["-y", "@playwright/mcp@latest"],
        ))

    assert result is False
    assert captured.get("cwd") == DATA_DIR


def test_stdio_connect_honors_explicit_cwd_override():
    import mcp

    captured = {}
    real_init = mcp.StdioServerParameters.__init__

    def spy_init(self, **kwargs):
        captured.update(kwargs)
        real_init(self, **kwargs)

    mgr = McpManager()
    with patch.object(mcp.StdioServerParameters, "__init__", spy_init), \
         patch("mcp.client.stdio.stdio_client", _raise_stop_here), \
         patch("src.mcp_manager._safe_errlog", return_value=None):
        asyncio.run(mgr.connect_server(
            "id1", "n", "stdio", command="npx", args=["-y", "pkg"], cwd="/custom/dir",
        ))

    assert captured.get("cwd") == "/custom/dir"
