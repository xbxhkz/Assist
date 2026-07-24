// Training panel (LoRA/QLoRA). ES module — DOM controller over the admin-gated
// /api/training engine. Admin-only: the rail button stays hidden unless
// /api/auth/status reports is_admin. Mirrors workflows.js (Modals, $, api, isAdmin).
import * as Modals from './modalManager.js';
import { formToConfig, vramHint, parseParamsB } from './trainingCore.js';
import { adapterActions } from './serveCore.js';

function $(id) { return document.getElementById(id); }
let pollTimer = null;
let freeGib = null;

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

function openTraining() { $('training-modal').classList.remove('hidden'); refreshEnv(); refreshAdapters(); loadFreeVram(); resumeIfRunning(); }
function closeTraining() { $('training-modal').classList.add('hidden'); stopPolling(); }

async function loadFreeVram() {
  try {
    const j = await api('/api/hwfit/usage');
    const g = j.gpus && j.gpus[0];
    if (g && g.vram_total_gb != null && g.vram_used_gb != null) {
      freeGib = Math.round((g.vram_total_gb - g.vram_used_gb) * 100) / 100;
    } else { freeGib = null; }
  } catch (e) { freeGib = null; }
  updateHint();
}

function updateHint() {
  const model = $('training-model');
  const h = vramHint(parseParamsB(model ? model.value : ''), freeGib);
  const el = $('training-vram-hint'); if (el) el.textContent = h.label;
}

async function refreshEnv() {
  try {
    const j = await api('/api/training/env');
    $('training-env-status').textContent = j.status || 'unknown';
    const ready = j.status === 'ready';
    $('training-run-card').style.opacity = ready ? '1' : '0.5';
    $('training-env-setup').style.display = ready ? 'none' : '';
    const start = $('training-start'); if (start) start.disabled = !ready;
  } catch (e) {
    $('training-env-status').textContent = 'error';
    const start = $('training-start'); if (start) start.disabled = true;
  }
}

async function setupEnv() {
  const p = $('training-env-progress');
  if (p) p.textContent = 'Setting up (downloads several GB, one time)…';
  $('training-env-setup').disabled = true;
  try {
    const j = await api('/api/training/env/setup', { method: 'POST' });
    if (p) p.textContent = j.ready ? 'Ready.' : ('Failed: ' + (j.error || 'unknown'));
  } catch (e) { if (p) p.textContent = 'Failed: ' + e.message; }
  $('training-env-setup').disabled = false; refreshEnv();
}

function collectConfig() {
  return formToConfig({
    base_model: $('training-model').value, dataset_path: $('training-dataset').value,
    mode: $('training-mode').value, steps: $('training-steps').value, epochs: $('training-epochs').value,
    lora_r: $('training-r').value, lora_alpha: $('training-alpha').value, lora_dropout: $('training-dropout').value,
    batch_size: $('training-batch').value, learning_rate: $('training-lr').value,
  });
}

async function startRun() {
  const prog = $('training-progress'); if (prog) prog.textContent = 'Starting…';
  try {
    const j = await api('/api/training/runs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(collectConfig()),
    });
    if (prog) prog.textContent = j.warning ? ('Started — ' + j.warning) : 'Started.';
    startPolling();
  } catch (e) { if (prog) prog.textContent = 'Error: ' + e.message; }
}

async function stopRun() {
  const prog = $('training-progress'); if (prog) prog.textContent = 'Stopping…';
  try {
    await api('/api/training/runs/stop', { method: 'POST' });
    if (prog) prog.textContent = 'Stopped.';
  } catch (e) { if (prog) prog.textContent = 'Stop failed: ' + e.message; }
}

function startPolling() { stopPolling(); pollTimer = setInterval(pollStatus, 1500); pollStatus(); }
function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

function renderStatus(s) {
  const bits = ['status: ' + s.status];
  if (s.last_step != null) bits.push('step ' + s.last_step);
  if (s.loss != null) bits.push('loss ' + s.loss);
  if (s.vram_gb != null) bits.push('vram ' + s.vram_gb + ' GB');
  if (s.error) bits.push('(' + s.error + ')');
  const prog = $('training-progress'); if (prog) prog.textContent = bits.join(' · ');
}

async function pollStatus() {
  try {
    const s = await api('/api/training/runs/current');
    renderStatus(s);
    if (s.status === 'done' || s.status === 'error' || s.status === 'stopped') { stopPolling(); refreshAdapters(); }
  } catch (e) {}
}

// On (re)open, show the current run's status and resume polling if it's live.
// startPolling() calls stopPolling() first, so this is idempotent with any timer.
async function resumeIfRunning() {
  try {
    const s = await api('/api/training/runs/current');
    if (s.status && s.status !== 'idle') renderStatus(s);
    if (s.status === 'running') startPolling();
  } catch (e) {}
}

async function refreshAdapters() {
  try {
    const j = await api('/api/training/adapters');
    const host = $('training-adapters');
    if (!host) return;
    const rows = (j.adapters || []).map(function (a) {
      const act = adapterActions(a);
      const btns =
        (act.canConvert ? '<button class="btn" data-conv="' + esc(a.run_id) + '">Convert to GGUF</button>' : '') +
        (act.canServe ? '<button class="btn" data-serve="' + esc(a.run_id) + '">Serve</button>' : '');
      return '<div>' + (a.complete ? '✅' : '⏳') + ' ' + esc(a.run_id) + ' — ' +
             esc(a.base_model || '?') + ' ' + btns + '</div>';
    });
    host.innerHTML = rows.join('') || 'None yet.';
    host.querySelectorAll('[data-conv]').forEach(function (b) {
      b.addEventListener('click', function () { convertAdapter(b.getAttribute('data-conv')); });
    });
    host.querySelectorAll('[data-serve]').forEach(function (b) {
      b.addEventListener('click', function () { serveAdapter(b.getAttribute('data-serve')); });
    });
  } catch (e) {}
}

async function convertAdapter(runId) {
  const host = $('training-adapters');
  try {
    if (host) host.textContent = 'Converting ' + runId + ' to GGUF…';
    await api('/api/training/adapters/' + encodeURIComponent(runId) + '/convert', { method: 'POST' });
  } catch (e) { alert('Convert failed: ' + e.message); }
  refreshAdapters();
}

async function serveAdapter(runId) {
  try {
    const st = await api('/api/training/adapters/' + encodeURIComponent(runId));
    let base = st.base_match && st.base_match.matched;
    base = window.prompt('Base GGUF to serve with this adapter (must match ' +
                         (st.base_model || 'the base') + '):', base || '');
    if (!base) return;
    await api('/api/training/adapters/' + encodeURIComponent(runId) + '/serve', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_gguf: base }),
    });
    alert('Serving — pick the tuned model in the model selector.');
  } catch (e) { alert('Serve failed: ' + e.message); }
}

function init() {
  // Admin-only: reveal BOTH the icon-rail button and the sidebar Tools entry.
  isAdmin().then(function (ok) {
    if (!ok) return;
    ['rail-training', 'tool-training-btn'].forEach(function (id) {
      const b = $(id); if (b) b.style.display = '';
    });
  });
  const rail = $('rail-training'); if (rail) rail.addEventListener('click', openTraining);
  const side = $('tool-training-btn'); if (side) side.addEventListener('click', openTraining);
  const x = $('training-close'); if (x) x.addEventListener('click', closeTraining);
  const setup = $('training-env-setup'); if (setup) setup.addEventListener('click', setupEnv);
  const start = $('training-start'); if (start) start.addEventListener('click', startRun);
  const stop = $('training-stop'); if (stop) stop.addEventListener('click', stopRun);
  const model = $('training-model'); if (model) model.addEventListener('input', updateHint);
  Modals.register('training-modal', {
    railBtnId: 'rail-training', sidebarBtnId: 'tool-training-btn', closeFn: closeTraining,
  });
}

document.addEventListener('DOMContentLoaded', init);
