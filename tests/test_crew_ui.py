# tests/test_crew_ui.py
import pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_index_has_crew_modal_and_entries():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in ('id="crew-modal"', 'id="rail-crew"', 'id="tool-crew-btn"',
               '/static/js/crew.js',
               'id="crew-grid"', 'id="crew-new-btn"',
               'id="crew-form-name"', 'id="crew-form-avatar"', 'id="crew-form-personality"',
               'id="crew-form-model"', 'id="crew-form-endpoint"', 'id="crew-form-greeting"',
               'id="crew-form-tools"', 'id="crew-form-save"', 'id="crew-form-cancel"'):
        assert el in html, f"{el} missing from index.html"


def test_crew_rail_button_is_not_admin_gated():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    # Every other panel this session hides its rail button until isAdmin()
    # reveals it (style="display:none" baked into the HTML). Crew must NOT --
    # find the exact rail-crew button tag and confirm no display:none on it.
    import re
    m = re.search(r'<button class="icon-rail-btn" id="rail-crew"[^>]*>', html)
    assert m is not None, "rail-crew button not found"
    assert "display:none" not in m.group(0)


def test_crew_js_wires_routes():
    src = (ROOT / "static" / "js" / "crew.js").read_text(encoding="utf-8")
    for s in ('rail-crew', 'tool-crew-btn', 'Modals.register',
              "api('/api/crew')", '/api/crew/tool-names', 'createDirectChat'):
        assert s in src, f"{s} missing from crew.js"
    # Crew must NOT gate its own reveal behind isAdmin, unlike every admin panel.
    assert "isAdmin" not in src


def test_sessions_js_thread_crew_member_id_through_pending_chat():
    src = (ROOT / "static" / "js" / "sessions.js").read_text(encoding="utf-8")
    assert "crewMemberId" in src
    assert "crew_member_id" in src


def test_crew_js_syntax(tmp_path):
    src = (ROOT / "static" / "js" / "crew.js").read_text(encoding="utf-8")
    mjs = tmp_path / "crew.mjs"; mjs.write_text(src, encoding="utf-8")
    p = subprocess.run(["node", "--check", str(mjs)], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
