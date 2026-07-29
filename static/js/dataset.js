// Dataset builder/validator panel (AI Studio). ES module. Admin-only: the entries
// stay hidden unless /api/auth/status reports is_admin. Mirrors training.js.
import * as Modals from './modalManager.js';
import { ROW_FORMATS, formToRow } from './datasetCore.js';

function $(id) { return document.getElementById(id); }
let rows = [];

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
  });
}

async function api(path, opts) {
  const res = await fetch(path, Object.assign({ credentials: 'same-origin' }, opts || {}));
  const data = await res.json().catch(() => ({}));
  if (!res.ok) { const d = data && data.detail; throw new Error(d || String(res.status)); }
  return data;
}

async function isAdmin() {
  try { const d = await (await fetch('/api/auth/status', { credentials: 'same-origin' })).json(); return !!d.is_admin; }
  catch (e) { return false; }
}

function openDataset() { $('dataset-modal').classList.remove('hidden'); renderFields(); renderRows(); refreshSaved(); }
function closeDataset() { $('dataset-modal').classList.add('hidden'); }

function renderFields() {
  const fmt = $('dataset-format') ? $('dataset-format').value : 'text';
  const host = $('dataset-fields'); if (!host) return;
  host.innerHTML = (ROW_FORMATS[fmt] || []).map(function (k) {
    return '<label>' + k + '<br><textarea data-field="' + k + '" rows="2" style="width:100%"></textarea></label>';
  }).join('');
}

function renderRows() {
  const host = $('dataset-rows'); if (!host) return;
  host.innerHTML = rows.length
    ? rows.map(function (r, i) {
        return '<div>' + (i + 1) + '. ' + esc(JSON.stringify(r)).slice(0, 160) +
               ' <button class="btn" data-del="' + i + '">✕</button></div>';
      }).join('')
    : '<div style="opacity:0.6">No rows yet.</div>';
  host.querySelectorAll('[data-del]').forEach(function (b) {
    b.addEventListener('click', function () { rows.splice(parseInt(b.getAttribute('data-del'), 10), 1); renderRows(); });
  });
  const c = $('dataset-count'); if (c) c.textContent = rows.length + ' row(s)';
}

function addRow() {
  const fmt = $('dataset-format').value;
  const fields = {};
  $('dataset-fields').querySelectorAll('[data-field]').forEach(function (el) { fields[el.getAttribute('data-field')] = el.value; });
  const out = formToRow(fmt, fields);
  if (out.error) { alert(out.error); return; }
  rows.push(out.row);
  $('dataset-fields').querySelectorAll('[data-field]').forEach(function (el) { el.value = ''; });
  renderRows();
}

function importText() {
  const ta = $('dataset-import'); if (!ta) return;
  const added = []; let skipped = 0;
  ta.value.split('\n').forEach(function (line) {
    line = line.trim(); if (!line) return;
    try { added.push(JSON.parse(line)); } catch (e) { skipped += 1; }
  });
  if (!added.length) {
    alert('No valid JSON lines found' + (skipped ? ' (' + skipped + ' invalid line(s) skipped).' : '.'));
    return;
  }
  rows = rows.concat(added); ta.value = ''; renderRows();
  if (skipped) alert('Imported ' + added.length + ' row(s); skipped ' + skipped + ' invalid line(s).');
}

async function validate() {
  const out = $('dataset-report');
  try {
    const rep = await api('/api/datasets/validate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ rows: rows }),
    });
    const errs = (rep.errors || []).slice(0, 20).map(function (e) { return 'line ' + e.line + ': ' + esc(e.message); }).join('<br>');
    if (out) out.innerHTML = 'valid ' + rep.valid + ' / ' + rep.total + ' · shapes ' +
      esc(JSON.stringify(rep.stats.shapes)) + ' · ~' + rep.stats.approx_tokens + ' tokens' +
      (errs ? '<br>' + errs : '');
  } catch (e) { if (out) out.textContent = 'Validate failed: ' + e.message; }
}

async function save() {
  const name = $('dataset-name') ? $('dataset-name').value.trim() : '';
  if (!name) { alert('Enter a dataset name.'); return; }
  try {
    const r = await api('/api/datasets', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: name, rows: rows }),
    });
    alert('Saved: ' + r.path); refreshSaved();
  } catch (e) { alert('Save failed: ' + e.message); }
}

async function refreshSaved() {
  const host = $('dataset-saved'); if (!host) return;
  try {
    const j = await api('/api/datasets');
    host.innerHTML = (j.datasets || []).map(function (d) {
      return '<div>' + esc(d.name) + ' (' + d.rows + ' rows) ' +
             '<button class="btn" data-load="' + esc(d.name) + '">Load</button>' +
             '<button class="btn" data-delds="' + esc(d.name) + '">Delete</button></div>';
    }).join('') || 'None yet.';
    host.querySelectorAll('[data-load]').forEach(function (b) {
      b.addEventListener('click', function () { loadSaved(b.getAttribute('data-load')); });
    });
    host.querySelectorAll('[data-delds]').forEach(function (b) {
      b.addEventListener('click', function () {
        api('/api/datasets/' + encodeURIComponent(b.getAttribute('data-delds')), { method: 'DELETE' })
          .then(refreshSaved).catch(function () {});
      });
    });
  } catch (e) {}
}

async function loadSaved(name) {
  try { const j = await api('/api/datasets/' + encodeURIComponent(name)); rows = j.rows || []; renderRows(); }
  catch (e) { alert('Load failed: ' + e.message); }
}

function init() {
  isAdmin().then(function (ok) {
    if (!ok) return;
    ['rail-dataset', 'tool-dataset-btn'].forEach(function (id) { const b = $(id); if (b) b.style.display = ''; });
  });
  ['rail-dataset', 'tool-dataset-btn'].forEach(function (id) { const b = $(id); if (b) b.addEventListener('click', openDataset); });
  const x = $('dataset-close'); if (x) x.addEventListener('click', closeDataset);
  const fmt = $('dataset-format'); if (fmt) fmt.addEventListener('change', renderFields);
  const add = $('dataset-add'); if (add) add.addEventListener('click', addRow);
  const imp = $('dataset-import-btn'); if (imp) imp.addEventListener('click', importText);
  const val = $('dataset-validate'); if (val) val.addEventListener('click', validate);
  const sv = $('dataset-save'); if (sv) sv.addEventListener('click', save);
  Modals.register('dataset-modal', { railBtnId: 'rail-dataset', sidebarBtnId: 'tool-dataset-btn', closeFn: closeDataset });
}

document.addEventListener('DOMContentLoaded', init);
