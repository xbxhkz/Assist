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
    instead. Finding 13 changed endpoint_id/model from unconditional object-
    literal properties to a conditional `payload.endpoint_id = endpointId`
    assignment (so an unmatched existing binding can be left out of the
    payload entirely rather than sent as a wipe) -- assert on the assignment
    form now used, same intent as before."""
    src = (ROOT / "static" / "js" / "crew.js").read_text(encoding="utf-8")
    import re
    save_form = re.search(r'async function saveForm\(\)\s*\{.*?\n\}', src, re.S)
    assert save_form is not None, "saveForm function not found"
    body = save_form.group(0)
    assert "payload.endpoint_id = endpointId" in body
    assert "endpoint_url:" not in body
    assert "endpoint_url =" not in body


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


def test_crew_js_load_endpoint_options_reads_models_extra():
    """Finding 13 part 1: loadEndpointOptions must include models_extra
    (matching modelPicker.js's own `(item.models || []).concat(item.models_extra
    || [])` idiom) -- otherwise a persona's model living only in models_extra
    contributes zero options, and the edit form silently resets the endpoint
    <select> to "" when opened, wiping the binding on the next save."""
    src = (ROOT / "static" / "js" / "crew.js").read_text(encoding="utf-8")
    m = re.search(r'async function loadEndpointOptions\(\)\s*\{.*?\n\}', src, re.S)
    assert m is not None, "loadEndpointOptions function not found"
    assert "models_extra" in m.group(0)


def test_crew_js_preserves_unmatched_endpoint_binding_on_save():
    """Finding 13 part 2 (+ fix-wave-3 finding 2): opening a persona whose
    endpoint/model binding has no matching <option> (offline endpoint, or a
    model that only lives in models_extra before part 1's fix applied) must
    not silently wipe that binding when the form is saved without
    deliberately touching the dropdown. This must also cover personas with
    only ONE of endpoint_id/model set (e.g. the default Assistant seeded by
    src/task_scheduler.py, which predates the registered-endpoint
    architecture and has model but no endpoint_id) -- those must still be
    treated as having a binding worth preserving, not require both fields.
    A DOM test of the actual preserve-on-save behavior would need a DOM test
    runner this codebase doesn't have for crew.js yet, so this checks the
    actual statements (not just identifier presence, which would stay green
    even if the reset/guard statements were deleted)."""
    src = (ROOT / "static" / "js" / "crew.js").read_text(encoding="utf-8")
    m_edit = re.search(r'async function openEditForm\([^)]*\)\s*\{.*?\n\}', src, re.S)
    assert m_edit is not None, "openEditForm function not found"
    edit_src = m_edit.group(0)
    assert "_editingHadUnmatchedBinding = false;" in edit_src
    assert "_editingHadUnmatchedBinding = true;" in edit_src
    # Finding 2: a persona with only ONE of endpoint_id/model set (e.g. the
    # default Assistant, which has no endpoint_id) must still be treated as
    # having a binding worth preserving -- not require both fields.
    assert "existing.endpoint_id || existing.model" in edit_src

    m_save = re.search(r'async function saveForm\(\)\s*\{.*?\n\}', src, re.S)
    assert m_save is not None, "saveForm function not found"
    save_src = m_save.group(0)
    assert "!_editingHadUnmatchedBinding" in save_src


def test_assistant_js_reads_enabled_tools_all():
    """Finding 10: assistant.js is a separate panel from crew.js (the
    singleton Assistant settings modal) that renders the same
    CrewMember.enabled_tools data through its own hardcoded TOOL_GROUPS
    checklist. It must read crew.enabled_tools_all (mirroring crew.js's own
    allOn check) in the same function that builds enabledTools -- otherwise
    opening Assistant settings for an "all"-tools Assistant renders every
    box unchecked, and a bare save silently converts "all" down to nothing."""
    src = (ROOT / "static" / "js" / "assistant.js").read_text(encoding="utf-8")
    m = re.search(r'function _renderSettingsBody\([^)]*\)\s*\{.*?\n\}', src, re.S)
    assert m is not None, "_renderSettingsBody function not found"
    body = m.group(0)
    assert "crew.enabled_tools_all" in body
    assert "allToolsOn" in body


def test_crew_js_syntax(tmp_path):
    src = (ROOT / "static" / "js" / "crew.js").read_text(encoding="utf-8")
    mjs = tmp_path / "crew.mjs"; mjs.write_text(src, encoding="utf-8")
    p = subprocess.run(["node", "--check", str(mjs)], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr


def test_crew_modal_has_voice_field():
    src = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="crew-form-voice-select"' in src
    assert 'id="crew-form-voice-input"' in src


def test_crew_js_loads_tts_provider_for_voice_picker():
    src = (ROOT / "static" / "js" / "crew.js").read_text(encoding="utf-8")
    assert "/api/tts/stats" in src


def test_crew_js_save_form_includes_tts_voice():
    src = (ROOT / "static" / "js" / "crew.js").read_text(encoding="utf-8")
    m = re.search(r'async function saveForm\(\)\s*\{.*?\n\}', src, re.S)
    assert m is not None, "saveForm function not found"
    assert "tts_voice" in m.group(0)


def test_crew_js_open_edit_form_populates_voice_field():
    src = (ROOT / "static" / "js" / "crew.js").read_text(encoding="utf-8")
    m = re.search(r'async function openEditForm\([^)]*\)\s*\{.*?\n\}', src, re.S)
    assert m is not None, "openEditForm function not found"
    assert "tts_voice" in m.group(0)


def test_assistant_js_has_voice_field():
    src = (ROOT / "static" / "js" / "assistant.js").read_text(encoding="utf-8")
    assert "assistant-voice-select" in src
    assert "assistant-voice-input" in src


def test_assistant_js_save_includes_tts_voice():
    src = (ROOT / "static" / "js" / "assistant.js").read_text(encoding="utf-8")
    assert "tts_voice" in src
