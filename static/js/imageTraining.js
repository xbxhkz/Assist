// static/js/imageTraining.js
// Image LoRA Training panel. ES module — DOM controller over the admin-gated
// /api/image-training engine. Admin-only: the rail button stays hidden unless
// /api/auth/status reports is_admin. Mirrors training.js (Modals, $, api, isAdmin).
import * as Modals from './modalManager.js';
import { formToConfig, renderStatusLine } from './imageTrainingCore.js';

function $(id) { return document.getElementById(id); }
let pollTimer = null;

async function api(path, opts) {
  const res = await fetch(path, Object.assign({ credentials: 'same-origin' }, opts || {}));
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data && data.detail;
    throw new Error((d && d.errors ? d.errors.join('; ') : d) || String(res.status));
  }
  return data;
}

async function isAdmin() {
  try {
    const d = await (await fetch('/api/auth/status', { credentials: 'same-origin' })).json();
    return !!d.is_admin;
  } catch (e) { return false; }
}

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
  });
}

function openImageTraining() {
  $('image-training-modal').classList.remove('hidden');
  refreshEnv(); refreshDatasetList(); resumeIfRunning();
}
function closeImageTraining() { $('image-training-modal').classList.add('hidden'); stopPolling(); }

async function refreshEnv() {
  try {
    const j = await api('/api/image-training/env');
    $('imgtrain-env-status').textContent = j.status || 'unknown';
    const ready = j.status === 'ready';
    $('imgtrain-run-card').style.opacity = ready ? '1' : '0.5';
    $('imgtrain-env-setup').style.display = ready ? 'none' : '';
    const start = $('imgtrain-start'); if (start) start.disabled = !ready;
  } catch (e) {
    $('imgtrain-env-status').textContent = 'error';
    const start = $('imgtrain-start'); if (start) start.disabled = true;
  }
}

async function setupEnv() {
  const p = $('imgtrain-env-progress');
  if (p) p.textContent = 'Setting up (adds diffusers to the training venv)…';
  $('imgtrain-env-setup').disabled = true;
  try {
    const j = await api('/api/image-training/env/setup', { method: 'POST' });
    if (p) p.textContent = j.ready ? 'Ready.' : ('Failed: ' + (j.error || 'unknown'));
  } catch (e) { if (p) p.textContent = 'Failed: ' + e.message; }
  $('imgtrain-env-setup').disabled = false; refreshEnv();
}

function collectConfig() {
  return formToConfig({
    dataset_name: $('imgtrain-dataset').value, output_name: $('imgtrain-output-name').value,
    rank: $('imgtrain-rank').value, lora_alpha: $('imgtrain-alpha').value,
    learning_rate: $('imgtrain-lr').value, steps: $('imgtrain-steps').value,
    resolution: $('imgtrain-resolution').value,
  });
}

async function startRun() {
  const prog = $('imgtrain-progress'); if (prog) prog.textContent = 'Starting…';
  try {
    await api('/api/image-training/runs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(collectConfig()),
    });
    if (prog) prog.textContent = 'Started.';
    startPolling();
  } catch (e) { if (prog) prog.textContent = 'Error: ' + e.message; }
}

async function stopRun() {
  const prog = $('imgtrain-progress'); if (prog) prog.textContent = 'Stopping…';
  try {
    await api('/api/image-training/runs/stop', { method: 'POST' });
    if (prog) prog.textContent = 'Stopped.';
  } catch (e) { if (prog) prog.textContent = 'Stop failed: ' + e.message; }
}

function startPolling() { stopPolling(); pollTimer = setInterval(pollStatus, 1500); pollStatus(); }
function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

function renderStatus(s) {
  const prog = $('imgtrain-progress'); if (!prog) return;
  let line = renderStatusLine(s);
  if (s.status === 'done') line += ' — find it in the LoRA manager (Image models card)';
  prog.textContent = line;
}

async function pollStatus() {
  try {
    const s = await api('/api/image-training/runs/current');
    renderStatus(s);
    if (s.status === 'done' || s.status === 'error' || s.status === 'stopped') stopPolling();
  } catch (e) {}
}

// On (re)open, show the current run's status and resume polling if it's live.
// startPolling() calls stopPolling() first, so this is idempotent with any timer.
async function resumeIfRunning() {
  try {
    const s = await api('/api/image-training/runs/current');
    if (s.status && s.status !== 'idle') renderStatus(s);
    if (s.status === 'running') startPolling();
  } catch (e) {}
}

async function refreshDatasetList() {
  try {
    const j = await api('/api/image-datasets');
    const dl = $('imgtrain-dataset-suggestions');
    if (dl) dl.innerHTML = (j.datasets || []).map(function (d) {
      return '<option value="' + esc(d.name) + '"></option>';
    }).join('');
  } catch (e) {}
}

function init() {
  // Admin-only: reveal BOTH the icon-rail button and the sidebar Tools entry.
  isAdmin().then(function (ok) {
    if (!ok) return;
    ['rail-imagetraining', 'tool-imagetraining-btn'].forEach(function (id) {
      const b = $(id); if (b) b.style.display = '';
    });
  });
  const rail = $('rail-imagetraining'); if (rail) rail.addEventListener('click', openImageTraining);
  const side = $('tool-imagetraining-btn'); if (side) side.addEventListener('click', openImageTraining);
  const x = $('imgtrain-close'); if (x) x.addEventListener('click', closeImageTraining);
  const setup = $('imgtrain-env-setup'); if (setup) setup.addEventListener('click', setupEnv);
  const start = $('imgtrain-start'); if (start) start.addEventListener('click', startRun);
  const stop = $('imgtrain-stop'); if (stop) stop.addEventListener('click', stopRun);
  Modals.register('image-training-modal', {
    railBtnId: 'rail-imagetraining', sidebarBtnId: 'tool-imagetraining-btn', closeFn: closeImageTraining,
  });
}

document.addEventListener('DOMContentLoaded', init);
