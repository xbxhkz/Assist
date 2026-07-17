// Visual workflow editor. ES module: imports the pure graph core and the modal
// manager. This file is the DOM/SVG view + controller. Admin-only (the API is
// admin-gated); the rail button stays hidden unless /api/auth/status says so.
import * as Modals from './modalManager.js';
import * as G from './workflowGraph.js';

let graph = G.createGraph();
let currentId = null;             // server id of the loaded workflow (null = unsaved)
let selected = null;              // {kind:'node'|'edge', ...} — set by the view (Task 4)

function $(id) { return document.getElementById(id); }

function msg(text, isErr) {
  const m = $('wf-msg');
  if (m) { m.textContent = text || ''; m.style.color = isErr ? 'var(--red,#ff5555)' : ''; }
}

async function api(path, opts) {
  const res = await fetch(path, Object.assign({ credentials: 'same-origin' }, opts || {}));
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data && data.detail;
    const errs = d && d.errors ? d.errors.join('; ') : (d || String(res.status));
    throw new Error(errs);
  }
  return data;
}

// Replaced in Task 4 (canvas render) and Task 5 (list). Stubs keep openWorkflows
// callable after Task 3.
function render() {}
async function refreshList() {}

function openWorkflows() {
  $('workflows-modal').classList.remove('hidden');
  refreshList();
  render();
}

function closeWorkflows() {
  $('workflows-modal').classList.add('hidden');
}

async function isAdmin() {
  try {
    const d = await (await fetch('/api/auth/status', { credentials: 'same-origin' })).json();
    return !!d.is_admin;
  } catch (e) { return false; }
}

function init() {
  isAdmin().then((ok) => { const b = $('rail-workflows'); if (b && ok) b.style.display = ''; });
  const rail = $('rail-workflows'); if (rail) rail.addEventListener('click', openWorkflows);
  const x = $('wf-close'); if (x) x.addEventListener('click', closeWorkflows);
  Modals.register('workflows-modal', {
    railBtnId: 'rail-workflows', sidebarBtnId: 'tool-workflows-btn', closeFn: closeWorkflows,
  });
}

document.addEventListener('DOMContentLoaded', init);
