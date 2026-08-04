// static/js/crew.js
// Crew: multi-persona system. ES module — DOM controller over /api/crew.
// Unlike every admin-gated panel this session, Crew is NOT admin-gated --
// every authenticated user manages their own personas, matching the
// existing per-user Assistant feature it generalizes. Mirrors the
// established panel-controller shape (Modals, $, api) minus the admin-only gate.
import * as Modals from './modalManager.js';

function $(id) { return document.getElementById(id); }
let _crew = [];
let _editingId = null;

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
  });
}

async function api(path, opts) {
  const res = await fetch(path, Object.assign({ credentials: 'same-origin' }, opts || {}));
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data && data.detail;
    throw new Error(d || String(res.status));
  }
  return data;
}

function openCrew() {
  $('crew-modal').classList.remove('hidden');
  showListView();
  refreshList();
}
function closeCrew() { $('crew-modal').classList.add('hidden'); }

function showListView() {
  $('crew-list-view').style.display = '';
  $('crew-form-view').style.display = 'none';
}
function showFormView() {
  $('crew-list-view').style.display = 'none';
  $('crew-form-view').style.display = '';
}

async function refreshList() {
  try {
    const j = await api('/api/crew');
    _crew = j.crew || [];
    renderGrid();
  } catch (e) {}
}

function renderGrid() {
  const grid = $('crew-grid');
  if (!grid) return;
  grid.innerHTML = _crew.map(function (c) {
    const preview = esc((c.personality || '').slice(0, 80));
    return (
      '<div style="border:1px solid var(--border);border-radius:8px;padding:8px;width:200px">' +
      '<div style="font-weight:600">' + esc(c.name) + '</div>' +
      '<div style="font-size:12px;opacity:0.75;min-height:32px">' + preview + '</div>' +
      '<button class="btn" data-newchat="' + esc(c.id) + '">New Chat</button>' +
      '<button class="btn" data-edit="' + esc(c.id) + '">Edit</button>' +
      (c.is_default_assistant ? '' : '<button class="btn" data-delete="' + esc(c.id) + '">Delete</button>') +
      '</div>'
    );
  }).join('') || 'No personas yet.';

  grid.querySelectorAll('[data-newchat]').forEach(function (b) {
    b.addEventListener('click', function () { newChatWithPersona(b.getAttribute('data-newchat')); });
  });
  grid.querySelectorAll('[data-edit]').forEach(function (b) {
    b.addEventListener('click', function () { openEditForm(b.getAttribute('data-edit')); });
  });
  grid.querySelectorAll('[data-delete]').forEach(function (b) {
    b.addEventListener('click', function () { deletePersona(b.getAttribute('data-delete')); });
  });
}

async function newChatWithPersona(crewId) {
  const persona = _crew.find(function (c) { return c.id === crewId; });
  if (!persona) return;
  try {
    // Mirrors the app's existing "new chat" flow: createDirectChat only
    // stages the pending chat (no backend call yet); the actual session is
    // created by materializePendingSession() on the user's first message.
    // Both functions are extended (this task) to carry crew_member_id
    // through that same lazy path -- see sessions.js changes below.
    const { createDirectChat } = await import('./sessions.js');
    let url = '';
    let model = persona.model || '';
    let endpointId = persona.endpoint_id || null;
    if (!endpointId) {
      // Persona has no registered-endpoint override -- fall back to the
      // app's default chat endpoint (same pattern as sessions.js's own
      // auto-create-first-session flow) so the first message doesn't 400
      // for a missing endpoint_url. Never forward persona.endpoint_url
      // directly here -- it is a raw URL and would 403 non-admin users
      // under the server's registered-endpoint guard.
      try {
        const dc = await (await fetch('/api/default-chat', { credentials: 'same-origin' })).json();
        if (dc && dc.endpoint_url) {
          endpointId = dc.endpoint_id || null;
          url = dc.endpoint_url;
          if (!model) model = dc.model || '';
        }
      } catch (e) { /* best-effort fallback */ }
    }
    createDirectChat(url, model, endpointId, crewId);
    closeCrew();
  } catch (e) {
    console.error('newChatWithPersona failed:', e);
  }
}

let _endpointOptions = [];
async function loadEndpointOptions() {
  if (_endpointOptions.length) return _endpointOptions;
  try {
    const j = await api('/api/models');
    const opts = [];
    (j.items || []).forEach(function (item) {
      if (!item.endpoint_id) return;
      (item.models || []).forEach(function (m) {
        opts.push({ endpointId: item.endpoint_id, model: m, label: (item.endpoint_name || item.endpoint_id) + ' — ' + m });
      });
    });
    _endpointOptions = opts;
  } catch (e) { _endpointOptions = []; }
  return _endpointOptions;
}

let _toolNames = [];
async function loadToolNames() {
  if (_toolNames.length) return _toolNames;
  try {
    const j = await api('/api/crew/tool-names');
    _toolNames = j.tools || [];
  } catch (e) { _toolNames = []; }
  return _toolNames;
}

async function openEditForm(crewId) {
  _editingId = crewId || null;
  const existing = crewId ? _crew.find(function (c) { return c.id === crewId; }) : null;
  $('crew-form-name').value = existing ? existing.name : '';
  $('crew-form-avatar').value = existing ? (existing.avatar || '') : '';
  $('crew-form-personality').value = existing ? (existing.personality || '') : '';
  $('crew-form-greeting').value = existing ? (existing.greeting || '') : '';

  const opts = await loadEndpointOptions();
  const sel = $('crew-form-endpoint');
  if (sel) {
    sel.innerHTML = '<option value="">(use default — no override)</option>' +
      opts.map(function (o) {
        return '<option value="' + esc(o.endpointId) + '::' + esc(o.model) + '">' + esc(o.label) + '</option>';
      }).join('');
    if (existing && existing.endpoint_id && existing.model) {
      sel.value = existing.endpoint_id + '::' + existing.model;
    } else {
      sel.value = '';
    }
  }

  const names = await loadToolNames();
  const allOn = !!(existing && existing.enabled_tools_all);
  const enabled = new Set(existing ? (existing.enabled_tools || []) : []);
  const host = $('crew-form-tools');
  if (host) {
    host.innerHTML = names.map(function (t) {
      const checked = (allOn || enabled.has(t)) ? 'checked' : '';
      return '<label style="display:block"><input type="checkbox" value="' + esc(t) + '" ' + checked + '> ' + esc(t) + '</label>';
    }).join('');
  }
  showFormView();
}

function collectFormToolList() {
  const host = $('crew-form-tools');
  if (!host) return [];
  return Array.from(host.querySelectorAll('input[type=checkbox]:checked')).map(function (i) { return i.value; });
}

async function saveForm() {
  const endpointVal = $('crew-form-endpoint').value;
  const [endpointId, model] = endpointVal ? endpointVal.split('::') : [null, ''];
  const payload = {
    name: $('crew-form-name').value,
    avatar: $('crew-form-avatar').value,
    personality: $('crew-form-personality').value,
    model: model,
    endpoint_id: endpointId,
    greeting: $('crew-form-greeting').value,
  };
  if (_toolNames.length) {
    payload.enabled_tools = collectFormToolList();
  }
  try {
    if (_editingId) {
      await api('/api/crew/' + encodeURIComponent(_editingId), {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
    } else {
      await api('/api/crew', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
    }
    showListView();
    refreshList();
  } catch (e) {
    console.error('saveForm failed:', e);
  }
}

async function deletePersona(crewId) {
  try {
    await api('/api/crew/' + encodeURIComponent(crewId), { method: 'DELETE' });
    refreshList();
  } catch (e) {
    console.error('deletePersona failed:', e);
  }
}

function init() {
  const rail = $('rail-crew'); if (rail) rail.addEventListener('click', openCrew);
  const side = $('tool-crew-btn'); if (side) side.addEventListener('click', openCrew);
  const x = $('crew-close'); if (x) x.addEventListener('click', closeCrew);
  const newBtn = $('crew-new-btn'); if (newBtn) newBtn.addEventListener('click', function () { openEditForm(null); });
  const save = $('crew-form-save'); if (save) save.addEventListener('click', saveForm);
  const cancel = $('crew-form-cancel'); if (cancel) cancel.addEventListener('click', showListView);
  Modals.register('crew-modal', {
    railBtnId: 'rail-crew', sidebarBtnId: 'tool-crew-btn', closeFn: closeCrew,
  });
}

document.addEventListener('DOMContentLoaded', init);
