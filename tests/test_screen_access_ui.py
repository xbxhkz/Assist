import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]

def _read(p): return (ROOT / p).read_text(encoding="utf-8")

def test_sidebar_has_screen_toggle():
    html = _read("static/index.html")
    assert 'id="screen-access-toggle"' in html
    assert 'id="screen-access-indicator"' in html

def test_js_posts_screen_access_setting():
    js = _read("static/js/screenAccess.js")
    assert "screen_access_enabled" in js
    assert "/api/auth/settings" in js

def test_manual_documents_screen_access():
    html = _read("static/index.html")
    assert "screen access" in html.lower()
