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

async function loadHardwareWidget() {
  const body = $('mc-body-hardware');
  if (body) body.classList.remove('mc-error');
  try {
    const data = await api('/api/hwfit/usage');
    const gpuLine = (data.gpus || []).map(function (g) {
      return esc(g.name || 'GPU') + ': ' + esc(Math.round(g.util_percent || 0)) + '%';
    }).join(', ') || 'No GPU detected';
    setCardBody('hardware',
      'CPU ' + esc(Math.round(data.cpu_percent || 0)) + '% · ' +
      'RAM ' + esc(data.ram_used_gb || 0) + '/' + esc(data.ram_total_gb || 0) + ' GB<br>' +
      gpuLine);
  } catch (e) {
    setCardError('hardware', e.message);
  }
}

async function loadTasksWidget() {
  const body = $('mc-body-tasks');
  if (body) body.classList.remove('mc-error');
  try {
    const data = await api('/api/tasks');
    const tasks = data.tasks || [];
    const active = tasks.filter(function (t) { return t.status === 'active'; }).length;
    const paused = tasks.filter(function (t) { return t.status === 'paused'; }).length;
    const recent = tasks
      .filter(function (t) { return t.last_run; })
      .sort(function (a, b) { return (b.last_run || '').localeCompare(a.last_run || ''); })
      .slice(0, 3);
    const recentHtml = recent.map(function (t) {
      return '<div>' + esc(t.name) + ' — ' + esc(t.status) + '</div>';
    }).join('') || '<div>No recent runs</div>';
    setCardBody('tasks', esc(active) + ' active, ' + esc(paused) + ' paused, ' + esc(tasks.length) + ' total<br>' + recentHtml);
  } catch (e) {
    setCardError('tasks', e.message);
  }
}

async function loadMemoryWidget() {
  const body = $('mc-body-memory');
  if (body) body.classList.remove('mc-error');
  try {
    const data = await api('/api/memory/timeline');
    const items = (data.timeline || []).slice(0, 3);
    const itemsHtml = items.map(function (m) {
      const text = (m.text || '').slice(0, 60);
      return '<div>' + esc(text) + (m.text && m.text.length > 60 ? '…' : '') + '</div>';
    }).join('') || '<div>No memories yet</div>';
    setCardBody('memory', esc(data.total || 0) + ' memories total<br>' + itemsHtml);
  } catch (e) {
    setCardError('memory', e.message);
  }
}

async function loadIntegrationsWidget() {
  const body = $('mc-body-integrations');
  if (body) body.classList.remove('mc-error');
  try {
    const data = await api('/api/auth/integrations');
    const items = data.integrations || [];
    const enabled = items.filter(function (i) { return i.enabled; }).length;
    const listHtml = items.slice(0, 5).map(function (i) {
      return '<div>' + esc(i.name) + ' — ' + (i.enabled ? 'enabled' : 'disabled') + '</div>';
    }).join('') || '<div>No integrations configured</div>';
    setCardBody('integrations', esc(enabled) + ' / ' + esc(items.length) + ' enabled<br>' + listHtml);
  } catch (e) {
    setCardError('integrations', e.message);
  }
}

async function loadToolCallsWidget() {
  const body = $('mc-body-tool-calls');
  if (body) body.classList.remove('mc-error');
  try {
    const data = await api('/api/tool-calls?limit=3');
    const items = data.tool_calls || [];
    const listHtml = items.map(function (c) {
      const cmd = (c.command || '').slice(0, 40);
      return '<div>' + esc(c.tool || '?') + ': ' + esc(cmd) + '</div>';
    }).join('') || '<div>No tool calls yet</div>';
    const suffix = data.has_more ? '+' : '';
    setCardBody('tool-calls', esc(items.length) + suffix + ' recent<br>' + listHtml);
  } catch (e) {
    setCardError('tool-calls', e.message);
  }
}

async function loadActiveAgentsWidget() {
  const body = $('mc-body-active-agents');
  if (body) body.classList.remove('mc-error');
  try {
    const data = await api('/api/agent-runs/active');
    const items = data.active || [];
    const listHtml = items.map(function (a) {
      return '<div><a href="#" class="mc-active-agent-session" data-session-id="' + esc(a.session_id) + '">' + esc(a.session_name || 'Unknown') + '</a></div>';
    }).join('') || '<div>No active agents right now</div>';
    setCardBody('active-agents', esc(items.length) + ' active<br>' + listHtml);
  } catch (e) {
    setCardError('active-agents', e.message);
  }
}

async function loadWorkflowRunsWidget() {
  const body = $('mc-body-workflow-runs');
  if (body) body.classList.remove('mc-error');
  try {
    const data = await api('/api/workflow-runs/active');
    const items = data.active || [];
    const listHtml = items.map(function (w) {
      return '<div>' + esc(w.workflow_name || w.workflow_id || '?') + ' (' + esc(w.trigger || '?') + ')</div>';
    }).join('') || '<div>No workflows running right now</div>';
    setCardBody('workflow-runs', esc(items.length) + ' running<br>' + listHtml);
  } catch (e) {
    setCardError('workflow-runs', e.message);
  }
}

function refreshWidget(widgetId) {
  if (widgetId === 'models') loadModelsWidget();
  if (widgetId === 'hardware') loadHardwareWidget();
  if (widgetId === 'tasks') loadTasksWidget();
  if (widgetId === 'memory') loadMemoryWidget();
  if (widgetId === 'integrations') loadIntegrationsWidget();
  if (widgetId === 'tool-calls') loadToolCallsWidget();
  if (widgetId === 'active-agents') loadActiveAgentsWidget();
  if (widgetId === 'workflow-runs') loadWorkflowRunsWidget();
}

function loadAllWidgets() {
  loadModelsWidget();
  loadHardwareWidget();
  loadTasksWidget();
  loadMemoryWidget();
  loadIntegrationsWidget();
  loadToolCallsWidget();
  loadActiveAgentsWidget();
  loadWorkflowRunsWidget();
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
  const openHardware = $('mc-open-hardware');
  if (openHardware) openHardware.addEventListener('click', function () {
    closeMissionControl();
    const hwmon = $('hwmon');
    if (hwmon) {
      hwmon.open = true;
      hwmon.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  });
  const openTasks = $('mc-open-tasks');
  if (openTasks) openTasks.addEventListener('click', function () {
    closeMissionControl();
    const tasksBtn = $('rail-tasks');
    if (tasksBtn) tasksBtn.click();
  });
  const openMemory = $('mc-open-memory');
  if (openMemory) openMemory.addEventListener('click', function () {
    closeMissionControl();
    const memoryBtn = $('tool-memory-btn');
    if (memoryBtn) memoryBtn.click();
  });
  const openIntegrations = $('mc-open-integrations');
  if (openIntegrations) openIntegrations.addEventListener('click', function () {
    closeMissionControl();
    const pluginsBtn = $('tool-plugins-btn');
    if (pluginsBtn) pluginsBtn.click();
  });
  const openToolCalls = $('mc-open-tool-calls');
  if (openToolCalls) openToolCalls.addEventListener('click', function () {
    closeMissionControl();
    const toolCallsBtn = $('tool-tool-calls-btn');
    if (toolCallsBtn) toolCallsBtn.click();
  });
  const activeAgentsBody = $('mc-body-active-agents');
  if (activeAgentsBody) activeAgentsBody.addEventListener('click', function (ev) {
    const link = ev.target.closest('.mc-active-agent-session');
    if (!link) return;
    ev.preventDefault();
    const sid = link.getAttribute('data-session-id');
    if (sid) {
      closeMissionControl();
      if (window.sessionModule && window.sessionModule.selectSession) {
        window.sessionModule.selectSession(sid);
      }
    }
  });
  const openWorkflowRuns = $('mc-open-workflow-runs');
  if (openWorkflowRuns) openWorkflowRuns.addEventListener('click', function () {
    closeMissionControl();
    const workflowsBtn = $('tool-workflows-btn');
    if (workflowsBtn) workflowsBtn.click();
  });
  Modals.register('mission-control-modal', {
    railBtnId: 'rail-mission-control', sidebarBtnId: 'tool-mission-control-btn', closeFn: closeMissionControl,
  });
}

document.addEventListener('DOMContentLoaded', init);
