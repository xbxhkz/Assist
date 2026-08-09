// static/js/missionControl.js
// Mission Control: read-only dashboard aggregating models, hardware, task
// queue, memory, and integrations. Each widget fetches and renders
// independently -- one widget's failure never blocks another's. NOT
// admin-gated by this file itself; each widget's own endpoint enforces
// whatever access rule it already has (e.g. integrations 403s for
// non-admins -- expected, not a bug here).
import * as Modals from './modalManager.js';

function $(id) { return document.getElementById(id); }

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
  });
}

async function api(path) {
  const res = await fetch(path, { credentials: 'same-origin' });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data && data.detail;
    throw new Error(typeof d === 'string' ? d : (res.statusText || String(res.status)));
  }
  return data;
}

function setCardBody(widgetId, html) {
  const body = $('mc-body-' + widgetId);
  if (body) body.innerHTML = html;
}

function setCardError(widgetId, message) {
  const body = $('mc-body-' + widgetId);
  if (body) {
    body.classList.add('mc-error');
    body.textContent = 'Failed to load: ' + message;
  }
}

async function loadModelsWidget() {
  const body = $('mc-body-models');
  if (body) body.classList.remove('mc-error');
  try {
    const data = await api('/api/models');
    const items = data.items || [];
    const online = items.filter(function (i) { return !i.offline; }).length;
    const modelCount = items.reduce(function (sum, i) { return sum + (i.models || []).length; }, 0);
    setCardBody('models', esc(online) + ' / ' + esc(items.length) + ' endpoints online, ' + esc(modelCount) + ' models total');
  } catch (e) {
    setCardError('models', e.message);
  }
}

function refreshWidget(widgetId) {
  if (widgetId === 'models') loadModelsWidget();
}

function loadAllWidgets() {
  loadModelsWidget();
}

function openMissionControl() {
  $('mission-control-modal').classList.remove('hidden');
  loadAllWidgets();
}
function closeMissionControl() { $('mission-control-modal').classList.add('hidden'); }

function init() {
  const rail = $('rail-mission-control'); if (rail) rail.addEventListener('click', openMissionControl);
  const side = $('tool-mission-control-btn'); if (side) side.addEventListener('click', openMissionControl);
  const x = $('mission-control-close'); if (x) x.addEventListener('click', closeMissionControl);
  document.querySelectorAll('.mission-control-refresh').forEach(function (btn) {
    btn.addEventListener('click', function () { refreshWidget(btn.getAttribute('data-widget')); });
  });
  const openModels = $('mc-open-models');
  if (openModels) openModels.addEventListener('click', function () {
    closeMissionControl();
    const modelBtn = $('model-picker-btn');
    if (modelBtn) modelBtn.click();
  });
  Modals.register('mission-control-modal', {
    railBtnId: 'rail-mission-control', sidebarBtnId: 'tool-mission-control-btn', closeFn: closeMissionControl,
  });
}

document.addEventListener('DOMContentLoaded', init);
