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
    createDirectChat(persona.endpoint_url || '', persona.model || '', null, crewId);
    closeCrew();
  } catch (e) {
    console.error('newChatWithPersona failed:', e);
  }
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
  $('crew-form-model').value = existing ? (existing.model || '') : '';
  $('crew-form-endpoint').value = existing ? (existing.endpoint_url || '') : '';
  $('crew-form-greeting').value = existing ? (existing.greeting || '') : '';

  const names = await loadToolNames();
  const enabled = new Set(existing ? (existing.enabled_tools || []) : []);
  const host = $('crew-form-tools');
  if (host) {
    host.innerHTML = names.map(function (t) {
      const checked = enabled.has(t) ? 'checked' : '';
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
  const payload = {
    name: $('crew-form-name').value,
    avatar: $('crew-form-avatar').value,
    personality: $('crew-form-personality').value,
    model: $('crew-form-model').value,
    endpoint_url: $('crew-form-endpoint').value,
    greeting: $('crew-form-greeting').value,
    enabled_tools: collectFormToolList(),
  };
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
