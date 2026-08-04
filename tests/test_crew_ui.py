# tests/test_crew_ui.py
import pathlib, re, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_index_has_crew_modal_and_entries():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in ('id="crew-modal"', 'id="rail-crew"', 'id="tool-crew-btn"',
               '/static/js/crew.js',
               'id="crew-grid"', 'id="crew-new-btn"',
               'id="crew-form-name"', 'id="crew-form-avatar"', 'id="crew-form-personality"',
               'id="crew-form-endpoint"', 'id="crew-form-greeting"',
               'id="crew-form-tools"', 'id="crew-form-save"', 'id="crew-form-cancel"'):
        assert el in html, f"{el} missing from index.html"


def test_index_crew_form_endpoint_is_a_select_not_a_free_text_url_input():
    """Critical 1/2 fix: the persona form binds to a registered endpoint via a
    <select>, not a free-text raw URL <input> -- a raw URL would 403 non-admin
    users and dropping straight to the endpoint wasn't validated at all."""
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="crew-form-model"' not in html
    import re
    m = re.search(r'<select id="crew-form-endpoint"[^>]*>', html)
    assert m is not None, "crew-form-endpoint must now be a <select>"


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


def test_crew_js_loads_endpoint_options_from_models_api():
    """Critical 1/2 fix: the edit form's endpoint picker is populated from
    /api/models (registered endpoints), not a free-text raw URL field."""
    src = (ROOT / "static" / "js" / "crew.js").read_text(encoding="utf-8")
    assert "loadEndpointOptions" in src
    assert "/api/models" in src
    assert "endpoint_id" in src


def test_new_chat_with_persona_falls_back_to_default_chat_endpoint():
    """Critical 1 fix: a persona with no model/endpoint override must not
    send an empty endpoint_url straight to createDirectChat (that 400s on
    the first message) -- it must fall back through /api/default-chat."""
    src = (ROOT / "static" / "js" / "crew.js").read_text(encoding="utf-8")
    assert "/api/default-chat" in src
    assert "newChatWithPersona" in src


def test_save_form_sends_endpoint_id_not_raw_endpoint_url():
    """Critical 2/3 frontend fix: saveForm must never send a raw endpoint_url
    in the create/update payload -- personas bind to a registered endpoint_id
    instead."""
    src = (ROOT / "static" / "js" / "crew.js").read_text(encoding="utf-8")
    import re
    save_form = re.search(r'async function saveForm\(\)\s*\{.*?\n\}', src, re.S)
    assert save_form is not None, "saveForm function not found"
    body = save_form.group(0)
    assert "endpoint_id: endpointId" in body
    assert "endpoint_url:" not in body


def test_sessions_js_thread_crew_member_id_through_pending_chat():
    src = (ROOT / "static" / "js" / "sessions.js").read_text(encoding="utf-8")
    assert "crewMemberId" in src
    assert "crew_member_id" in src


def test_set_pending_chat_merge_preserves_crew_member_id_and_still_clears_on_null():
    """Important 4 regression: setPendingChat used to REPLACE _pendingChat
    wholesale, so picking a model after starting a persona chat silently
    dropped crewMemberId with no error. It must now merge onto the existing
    object -- but an explicit setPendingChat(null) (used by modelPicker.js's
    stale-model cleanup and admin.js's deleted-endpoint cleanup) must still
    clear it outright, not be merged away by Object.assign ignoring a null
    source. This extracts the ACTUAL setPendingChat line from sessions.js
    and runs it in node, rather than re-implementing the logic in the test."""
    src = (ROOT / "static" / "js" / "sessions.js").read_text(encoding="utf-8")
    m = re.search(r"setPendingChat: \(v\) => \{ (.*?) \},", src)
    assert m is not None, "setPendingChat implementation not found in sessions.js"
    body = m.group(1)

    script = f"""
    let _pendingChat = {{ url: 'http://old', modelId: 'old-model', endpointId: 'old-ep', crewMemberId: 'persona-1' }};
    const setPendingChat = (v) => {{ {body} }};
    // Picking a new model must preserve the existing crewMemberId.
    setPendingChat({{ url: 'http://new', modelId: 'new-model', endpointId: 'new-ep' }});
    if (_pendingChat.crewMemberId !== 'persona-1') throw new Error('crewMemberId dropped: ' + JSON.stringify(_pendingChat));
    if (_pendingChat.modelId !== 'new-model') throw new Error('modelId not updated: ' + JSON.stringify(_pendingChat));
    // An explicit clear must still null it out, not merge away to a stale copy.
    setPendingChat(null);
    if (_pendingChat !== null) throw new Error('setPendingChat(null) did not clear: ' + JSON.stringify(_pendingChat));
    console.log('OK');
    """
    p = subprocess.run(["node", "-e", script], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    assert "OK" in p.stdout


def test_crew_js_syntax(tmp_path):
    src = (ROOT / "static" / "js" / "crew.js").read_text(encoding="utf-8")
    mjs = tmp_path / "crew.mjs"; mjs.write_text(src, encoding="utf-8")
    p = subprocess.run(["node", "--check", str(mjs)], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
