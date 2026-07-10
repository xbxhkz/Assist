import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
def _read(p): return (ROOT / p).read_text(encoding="utf-8")

def test_index_has_plugins_entry_and_modal():
    html = _read("static/index.html")
    for el in ('id="tool-plugins-btn"', 'id="plugins-modal"', 'id="plugins-list"'):
        assert el in html, f"{el} missing from index.html"
    assert 'src="/static/js/plugins.js"' in html

def test_plugins_js_fetches_all_three_sources():
    js = _read("static/js/plugins.js")
    for ep in ("/api/mcp/servers", "/api/mcp/builtins", "/api/integrations"):
        assert ep in js, f"{ep} not fetched in plugins.js"

def test_plugins_js_wires_actions():
    js = _read("static/js/plugins.js")
    for ref in ("/reconnect", "/api/mcp/builtins/", "/toggle",
                "/api/integrations/", "/test", "/api/mcp/servers/",
                "/api/integrations/presets"):
        assert ref in js, f"action {ref} not wired in plugins.js"
