// Image dataset prep (Image AI Studio, sub-project 1). ES module. Admin-only:
// entries stay hidden unless /api/auth/status reports is_admin. Mirrors dataset.js.
import * as Modals from './modalManager.js';

function $(id) { return document.getElementById(id); }
let workingSetId = null;
let images = [];  // [{id, caption}]

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

function openImageDataset() { $('imagedataset-modal').classList.remove('hidden'); renderGrid(); refreshSaved(); }
function closeImageDataset() { $('imagedataset-modal').classList.add('hidden'); }

function renderGrid() {
  const grid = $('imgds-grid'); if (!grid) return;
  grid.innerHTML = images.map(function (img) {
    return '<div style="width:140px">' +
      '<img src="/api/image-datasets/working/' + encodeURIComponent(workingSetId) + '/' + encodeURIComponent(img.id) +
      '" style="width:140px;height:140px;object-fit:cover;border-radius:6px">' +
      '<textarea data-cap="' + esc(img.id) + '" rows="2" style="width:140px;font-size:11px">' + esc(img.caption) + '</textarea>' +
      '<button class="btn" data-remove="' + esc(img.id) + '" style="width:100%">Remove</button>' +
      '</div>';
  }).join('');
  const c = $('imgds-count'); if (c) c.textContent = images.length + ' image(s)';
  grid.querySelectorAll('[data-cap]').forEach(function (ta) {
    ta.addEventListener('change', function () {
      const img = images.find(function (i) { return i.id === ta.getAttribute('data-cap'); });
      if (img) img.caption = ta.value;
    });
  });
  grid.querySelectorAll('[data-remove]').forEach(function (b) {
    b.addEventListener('click', function () {
      images = images.filter(function (i) { return i.id !== b.getAttribute('data-remove'); });
      renderGrid();
    });
  });
}

async function uploadFiles() {
  const input = $('imgds-file');
  if (!input || !input.files || !input.files.length) { alert('Choose image file(s) first.'); return; }
  const fd = new FormData();
  for (let i = 0; i < input.files.length; i++) fd.append('files', input.files[i]);
  try {
    const r = await api('/api/image-datasets/upload', { method: 'POST', body: fd });
    workingSetId = r.working_set_id;
    images = images.concat(r.images || []);
    renderGrid();
  } catch (e) { alert('Upload failed: ' + e.message); }
}

async function pickFromGallery() {
  const idsStr = prompt('Gallery image IDs to add (comma-separated):');
  if (!idsStr) return;
  const ids = idsStr.split(',').map(function (s) { return s.trim(); }).filter(Boolean);
  try {
    const r = await api('/api/image-datasets/from-gallery', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ working_set_id: workingSetId, ids: ids }),
    });
    workingSetId = r.working_set_id;
    images = images.concat(r.images || []);
    renderGrid();
  } catch (e) { alert('Gallery pick failed: ' + e.message); }
}

async function captionAll() {
  if (!workingSetId) { alert('Add images first.'); return; }
  const btn = $('imgds-caption-all');
  if (btn) { btn.disabled = true; btn.textContent = 'Captioning…'; }
  try {
    const r = await api('/api/image-datasets/caption', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ working_set_id: workingSetId }),
    });
    (r.images || []).forEach(function (u) {
      const img = images.find(function (i) { return i.id === u.id; });
      if (img) img.caption = u.caption;
    });
    renderGrid();
  } catch (e) { alert('Caption failed: ' + e.message); }
  finally { if (btn) { btn.disabled = false; btn.textContent = 'Caption all (AI)'; } }
}

function _captionMap() {
  const m = {};
  images.forEach(function (i) { m[i.id] = i.caption || ''; });
  return m;
}

async function validate() {
  const out = $('imgds-report');
  const trigger = $('imgds-trigger') ? $('imgds-trigger').value.trim() : '';
  try {
    const rep = await api('/api/image-datasets/validate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ working_set_id: workingSetId, captions: _captionMap(), trigger_word: trigger }),
    });
    const errs = (rep.errors || []).slice(0, 20).map(function (e) { return esc(e.id) + ': ' + esc(e.message); }).join('<br>');
    if (out) out.innerHTML = 'valid ' + rep.valid + ' / ' + rep.total +
      ' · duplicates ' + rep.stats.duplicates + ' · missing captions ' + rep.stats.missing_captions +
      (errs ? '<br>' + errs : '');
  } catch (e) { if (out) out.textContent = 'Validate failed: ' + e.message; }
}

async function save() {
  const name = $('imgds-name') ? $('imgds-name').value.trim() : '';
  if (!name) { alert('Enter a dataset name.'); return; }
  const trigger = $('imgds-trigger') ? $('imgds-trigger').value.trim() : '';
  try {
    const r = await api('/api/image-datasets', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ working_set_id: workingSetId, name: name, trigger_word: trigger, captions: _captionMap() }),
    });
    alert('Saved: ' + r.path);
    images = []; workingSetId = null; renderGrid(); refreshSaved();
  } catch (e) { alert('Save failed: ' + e.message); }
}

async function refreshSaved() {
  const host = $('imgds-saved'); if (!host) return;
  try {
    const j = await api('/api/image-datasets');
    host.innerHTML = (j.datasets || []).map(function (d) {
      return '<div>' + esc(d.name) + ' (' + d.images + ' images) ' +
             '<button class="btn" data-delds="' + esc(d.name) + '">Delete</button></div>';
    }).join('') || 'None yet.';
    host.querySelectorAll('[data-delds]').forEach(function (b) {
      b.addEventListener('click', function () {
        api('/api/image-datasets/' + encodeURIComponent(b.getAttribute('data-delds')), { method: 'DELETE' })
          .then(refreshSaved).catch(function () {});
      });
    });
  } catch (e) {}
}

function init() {
  isAdmin().then(function (ok) {
    if (!ok) return;
    ['rail-imagedataset', 'tool-imagedataset-btn'].forEach(function (id) { const b = $(id); if (b) b.style.display = ''; });
  });
  ['rail-imagedataset', 'tool-imagedataset-btn'].forEach(function (id) { const b = $(id); if (b) b.addEventListener('click', openImageDataset); });
  const x = $('imagedataset-close'); if (x) x.addEventListener('click', closeImageDataset);
  const src = $('imgds-source');
  if (src) src.addEventListener('change', function () {
    const isGallery = src.value === 'gallery';
    const f = $('imgds-file'); if (f) f.style.display = isGallery ? 'none' : '';
    const g = $('imgds-gallery-pick'); if (g) g.style.display = isGallery ? '' : 'none';
  });
  const f = $('imgds-file'); if (f) f.addEventListener('change', uploadFiles);
  const g = $('imgds-gallery-pick'); if (g) g.addEventListener('click', pickFromGallery);
  const capAll = $('imgds-caption-all'); if (capAll) capAll.addEventListener('click', captionAll);
  const val = $('imgds-validate'); if (val) val.addEventListener('click', validate);
  const sv = $('imgds-save'); if (sv) sv.addEventListener('click', save);
  Modals.register('imagedataset-modal', { railBtnId: 'rail-imagedataset', sidebarBtnId: 'tool-imagedataset-btn', closeFn: closeImageDataset });
}

document.addEventListener('DOMContentLoaded', init);
