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
// ── canvas geometry ──
const HEADER = 26, PORT_GAP = 20, NODE_W = 150, DOT = 10;

function portOffsetY(idx) { return HEADER + PORT_GAP / 2 + idx * PORT_GAP; }

function portPos(node, dir, port) {
  const ports = dir === 'in' ? G.inputPortsOf(node) : G.outputPortsOf(node);
  const i = Math.max(0, ports.indexOf(port));
  return { x: dir === 'in' ? node.x : node.x + NODE_W, y: node.y + portOffsetY(i) };
}

function nodeHeight(node) {
  const rows = Math.max(G.inputPortsOf(node).length, G.outputPortsOf(node).length, 1);
  return HEADER + rows * PORT_GAP + 8;
}

let pending = null;   // active wire drag: {fromNode, fromPort}

function render(opts) {
  const canvas = $('wf-canvas');
  if (!canvas) return;
  // wipe node divs (keep the <svg> overlay)
  Array.from(canvas.querySelectorAll('.wf-node')).forEach((el) => el.remove());
  graph.nodes.forEach((n) => canvas.appendChild(renderNode(n)));
  renderWires();
  if (!opts || !opts.keepInspector) renderInspector();
  const unwired = G.unwiredPorts(graph).length;
  const runBtn = $('wf-run');
  if (runBtn) runBtn.textContent = unwired ? `Run ▶ (${unwired} unwired)` : 'Run ▶';
}

function renderNode(node) {
  const div = document.createElement('div');
  div.className = 'wf-node';
  div.dataset.id = node.id;
  const sel = selected && selected.kind === 'node' && selected.id === node.id;
  div.style.cssText = `position:absolute;left:${node.x}px;top:${node.y}px;width:${NODE_W}px;`
    + `height:${nodeHeight(node)}px;background:var(--bg);border:1px solid ${sel ? 'var(--accent)' : 'var(--border)'};`
    + 'border-radius:6px;font-size:11px;user-select:none;box-shadow:0 1px 3px rgba(0,0,0,0.3);';

  const head = document.createElement('div');
  head.className = 'wf-node-head';
  head.style.cssText = `height:${HEADER}px;display:flex;align-items:center;gap:4px;padding:0 4px;`
    + 'border-bottom:1px solid var(--border);cursor:move;background:var(--panel);border-radius:6px 6px 0 0;';
  const type = document.createElement('span');
  type.textContent = node.type;
  type.style.cssText = 'opacity:0.6;text-transform:uppercase;font-size:9px;';
  const idIn = document.createElement('input');
  idIn.value = node.id;
  idIn.style.cssText = 'flex:1;min-width:0;background:transparent;border:none;color:var(--text);font-size:11px;';
  idIn.addEventListener('pointerdown', (e) => e.stopPropagation());  // don't start a drag
  idIn.addEventListener('change', () => {
    try { G.setNodeId(graph, node.id, idIn.value); render(); }
    catch (err) { idIn.value = node.id; msg(err.message, true); }
  });
  const del = document.createElement('button');
  del.textContent = '×';
  del.style.cssText = 'border:none;background:transparent;color:var(--text);cursor:pointer;font-size:14px;line-height:1;';
  del.addEventListener('pointerdown', (e) => e.stopPropagation());
  del.addEventListener('click', () => {
    G.removeNode(graph, node.id);
    if (selected && selected.kind === 'node' && selected.id === node.id) selected = null;
    render();
  });
  head.appendChild(type); head.appendChild(idIn); head.appendChild(del);
  head.addEventListener('pointerdown', (e) => startNodeDrag(e, node));
  head.addEventListener('click', () => selectNode(node.id));
  div.appendChild(head);

  G.inputPortsOf(node).forEach((port, i) => div.appendChild(portDot(node, 'in', port, i)));
  G.outputPortsOf(node).forEach((port, i) => div.appendChild(portDot(node, 'out', port, i)));
  return div;
}

function portDot(node, dir, port, i) {
  const wiredIn = dir === 'in'
    && !graph.edges.some((e) => e.to_node === node.id && e.to_port === port);
  const dot = document.createElement('div');
  dot.title = port;
  dot.dataset.node = node.id; dot.dataset.port = port; dot.dataset.dir = dir;
  const top = portOffsetY(i) - DOT / 2;
  const left = dir === 'in' ? -DOT / 2 : NODE_W - DOT / 2;
  dot.style.cssText = `position:absolute;top:${top}px;left:${left}px;width:${DOT}px;height:${DOT}px;`
    + `border-radius:50%;background:${wiredIn ? 'var(--red,#ff5555)' : 'var(--accent)'};`
    + 'border:1px solid var(--bg);cursor:crosshair;';
  const label = document.createElement('span');
  label.textContent = port;
  label.style.cssText = `position:absolute;top:${portOffsetY(i) - 7}px;font-size:9px;opacity:0.7;`
    + (dir === 'in' ? 'left:8px;' : `left:${NODE_W - 8}px;transform:translateX(-100%);`);
  dot.addEventListener('pointerdown', (e) => {
    e.stopPropagation();
    if (dir === 'out') startWire(e, node.id, port);
  });
  dot.addEventListener('pointerup', (e) => {
    if (dir === 'in' && pending) finishWire(node.id, port);
  });
  const wrap = document.createDocumentFragment();
  wrap.appendChild(dot); wrap.appendChild(label);
  const holder = document.createElement('div');
  holder.appendChild(wrap);
  return holder;
}

// ── node dragging ──
function startNodeDrag(e, node) {
  const canvas = $('wf-canvas');
  const rect = canvas.getBoundingClientRect();
  const offX = e.clientX - rect.left + canvas.scrollLeft - node.x;
  const offY = e.clientY - rect.top + canvas.scrollTop - node.y;
  function move(ev) {
    G.setNodePos(graph, node.id,
      Math.max(0, ev.clientX - rect.left + canvas.scrollLeft - offX),
      Math.max(0, ev.clientY - rect.top + canvas.scrollTop - offY));
    render();
  }
  function up() { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); }
  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', up);
}

// ── wiring ──
function canvasPoint(e) {
  const canvas = $('wf-canvas');
  const rect = canvas.getBoundingClientRect();
  return { x: e.clientX - rect.left + canvas.scrollLeft, y: e.clientY - rect.top + canvas.scrollTop };
}

function startWire(e, fromNode, fromPort) {
  pending = { fromNode, fromPort };
  function move(ev) { renderWires(canvasPoint(ev)); }
  function up() {
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', up);
    pending = null; renderWires();
  }
  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', up);
}

function finishWire(toNode, toPort) {
  if (!pending) return;
  const ok = G.addEdge(graph, pending.fromNode, pending.fromPort, toNode, toPort);
  if (!ok) msg('Cannot connect (type/direction/cycle)', true); else msg('');
  pending = null; render();
}

function bezier(x1, y1, x2, y2) {
  const dx = Math.max(30, Math.abs(x2 - x1) / 2);
  return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
}

function renderWires(cursor) {
  const svg = $('wf-wires');
  if (!svg) return;
  svg.innerHTML = '';
  graph.edges.forEach((e) => {
    const from = G.nodeById(graph, e.from_node);
    const to = G.nodeById(graph, e.to_node);
    if (!from || !to) return;
    const a = portPos(from, 'out', e.from_port);
    const b = portPos(to, 'in', e.to_port);
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', bezier(a.x, a.y, b.x, b.y));
    path.setAttribute('fill', 'none');
    const sel = selected && selected.kind === 'edge' && selected.edge
      && selected.edge.to_node === e.to_node && selected.edge.to_port === e.to_port;
    path.setAttribute('stroke', sel ? 'var(--accent)' : 'var(--text)');
    path.setAttribute('stroke-width', sel ? '2.5' : '1.5');
    path.setAttribute('opacity', '0.8');
    path.style.pointerEvents = 'stroke';
    path.style.cursor = 'pointer';
    path.addEventListener('click', () => selectEdge(e));
    svg.appendChild(path);
  });
  if (pending && cursor) {
    const from = G.nodeById(graph, pending.fromNode);
    const a = portPos(from, 'out', pending.fromPort);
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', bezier(a.x, a.y, cursor.x, cursor.y));
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', 'var(--accent)');
    path.setAttribute('stroke-width', '2');
    path.setAttribute('stroke-dasharray', '4 3');
    svg.appendChild(path);
  }
}

// ── selection + inspector ──
function selectNode(id) { selected = { kind: 'node', id }; render(); }
function selectEdge(edge) { selected = { kind: 'edge', edge }; render(); }

const _FIELDS = {
  input: [['name', 'text'], ['default', 'text']],
  template: [['template', 'area']],
  llm: [['prompt', 'area'], ['model', 'text'], ['system', 'area']],
  tool: [['tool', 'text'], ['args', 'area']],
  output: [['name', 'text']],
};

function renderInspector() {
  const host = $('wf-inspector');
  if (!host) return;
  if (!selected || selected.kind !== 'node') { host.style.display = 'none'; host.innerHTML = ''; return; }
  const node = G.nodeById(graph, selected.id);
  if (!node) { host.style.display = 'none'; return; }
  host.style.display = '';
  host.innerHTML = '';
  const title = document.createElement('div');
  title.textContent = `${node.type} · ${node.id}`;
  title.style.cssText = 'font-weight:600;margin-bottom:8px;font-size:12px;';
  host.appendChild(title);
  const cfg = Object.assign({}, node.config);
  if (node.type === 'branch') { renderBranchInspector(host, node, cfg); return; }
  (_FIELDS[node.type] || []).forEach(([key, kind]) => {
    const label = document.createElement('label');
    label.textContent = key;
    label.style.cssText = 'display:block;font-size:10px;opacity:0.7;margin-top:6px;';
    const field = document.createElement(kind === 'area' ? 'textarea' : 'input');
    field.value = cfg[key] || '';
    field.style.cssText = 'width:100%;background:var(--panel);color:var(--text);border:1px solid var(--border);'
      + 'border-radius:4px;padding:4px 6px;font-size:12px;box-sizing:border-box;'
      + (kind === 'area' ? 'min-height:56px;resize:vertical;' : '');
    field.addEventListener('input', () => {
      cfg[key] = field.value;
      G.setConfig(graph, node.id, cfg);
      render({ keepInspector: true });   // re-derive ports live (e.g. new {slot}) w/o losing focus
    });
    host.appendChild(label); host.appendChild(field);
  });
}
function renderBranchInspector(host, node, cfg) {
  const FCSS = 'width:100%;background:var(--panel);color:var(--text);border:1px solid var(--border);'
    + 'border-radius:4px;padding:4px 6px;font-size:12px;box-sizing:border-box;';
  function lbl(t) {
    const l = document.createElement('label');
    l.textContent = t; l.style.cssText = 'display:block;font-size:10px;opacity:0.7;margin-top:6px;';
    return l;
  }
  // mode dropdown (changing it shows/hides the prompt -> full render)
  host.appendChild(lbl('mode'));
  const modeSel = document.createElement('select');
  modeSel.style.cssText = FCSS;
  ['match', 'llm'].forEach((mo) => {
    const o = document.createElement('option'); o.value = mo; o.textContent = mo;
    if ((cfg.mode || 'match') === mo) o.selected = true;
    modeSel.appendChild(o);
  });
  modeSel.addEventListener('change', () => {
    cfg.mode = modeSel.value; G.setConfig(graph, node.id, cfg); render();
  });
  host.appendChild(modeSel);
  // cases list editor
  host.appendChild(lbl('cases'));
  const cases = Array.isArray(cfg.cases) ? cfg.cases.slice() : [];
  function commitCases(list) {   // single funnel for structural edits -> full render regenerates `cases` fresh
    cfg.cases = list.filter((x) => x && x.trim());
    G.setConfig(graph, node.id, cfg);
    render();
  }
  function nextCaseLabel(list) {
    let n = list.length + 1;
    while (list.includes(`case${n}`)) n += 1;
    return `case${n}`;
  }
  cases.forEach((c, i) => {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:4px;margin-top:2px;';
    const inp = document.createElement('input');
    inp.value = c; inp.style.cssText = FCSS + 'flex:1;';
    inp.addEventListener('input', () => { cases[i] = inp.value; });
    inp.addEventListener('change', () => { commitCases(cases); });   // relabel port on blur -> prune orphaned wires + redraw
    const del = document.createElement('button');
    del.textContent = '×'; del.style.cssText = 'font-size:12px;';
    del.addEventListener('click', () => {
      cases.splice(i, 1); commitCases(cases);
    });
    row.appendChild(inp); row.appendChild(del); host.appendChild(row);
  });
  const addBtn = document.createElement('button');
  addBtn.textContent = '+ case'; addBtn.style.cssText = 'margin-top:4px;font-size:11px;';
  addBtn.addEventListener('click', () => {
    cases.push(nextCaseLabel(cases));
    commitCases(cases);
  });
  host.appendChild(addBtn);
  // prompt (llm mode only)
  if ((cfg.mode || 'match') === 'llm') {
    host.appendChild(lbl('prompt'));
    const pf = document.createElement('textarea');
    pf.value = cfg.prompt || ''; pf.style.cssText = FCSS + 'min-height:56px;resize:vertical;';
    pf.addEventListener('input', () => {
      cfg.prompt = pf.value; G.setConfig(graph, node.id, cfg); render({ keepInspector: true });
    });
    host.appendChild(pf);
  }
}
async function refreshList() {
  const host = $('wf-list');
  if (!host) return;
  host.innerHTML = '';
  const newBtn = document.createElement('button');
  newBtn.textContent = '+ New';
  newBtn.style.cssText = 'width:100%;margin-bottom:6px;';
  newBtn.addEventListener('click', newWorkflow);
  host.appendChild(newBtn);
  let list = [];
  try { list = (await api('/api/workflows')).workflows || []; }
  catch (e) { msg(e.message, true); return; }
  list.forEach((w) => {
    const row = document.createElement('div');
    row.textContent = w.name || w.id;
    row.title = `${w.id} · ${w.nodes} nodes`;
    row.style.cssText = 'padding:4px 6px;cursor:pointer;border-radius:4px;font-size:12px;overflow:hidden;'
      + 'text-overflow:ellipsis;white-space:nowrap;' + (w.id === currentId ? 'background:var(--panel);' : '');
    row.addEventListener('click', () => loadWorkflow(w.id));
    host.appendChild(row);
  });
  if (!list.length) {
    const e = document.createElement('div');
    e.style.cssText = 'font-size:12px;opacity:0.6;padding:4px;';
    e.textContent = 'No workflows yet.';
    host.appendChild(e);
  }
}

function newWorkflow() {
  graph = G.createGraph();
  currentId = null;
  selected = null;
  const nameEl = $('wf-name'); if (nameEl) nameEl.value = '';
  const results = $('wf-results'); if (results) { results.style.display = 'none'; results.innerHTML = ''; }
  msg('');
  render();
  const trHost = $('wf-triggers'); if (trHost && trHost.style.display !== 'none') renderTriggers();
}

async function loadWorkflow(id) {
  let wf;
  try { wf = await api(`/api/workflows/${encodeURIComponent(id)}`); }
  catch (e) { msg(e.message, true); return; }
  const needsLayout = (wf.nodes || []).some((n) => typeof n.x !== 'number' || typeof n.y !== 'number');
  graph = G.createGraph(wf);
  currentId = wf.id;
  if (needsLayout) { try { G.autoLayout(graph); } catch (e) { /* cyclic legacy graph */ } }
  const nameEl = $('wf-name'); if (nameEl) nameEl.value = graph.name || '';
  selected = null;
  const results = $('wf-results'); if (results) { results.style.display = 'none'; results.innerHTML = ''; }
  msg('');
  render();
  refreshList();
  const trHost = $('wf-triggers'); if (trHost && trHost.style.display !== 'none') renderTriggers();
}

async function save() {
  const nameEl = $('wf-name');
  if (nameEl) G.setName(graph, nameEl.value);
  const body = G.toJSON(graph);
  if (currentId) body.id = currentId;
  const saved = await api('/api/workflows', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  currentId = saved.id;
  graph.id = saved.id;
  await refreshList();
  return saved;
}

async function run() {
  try { await save(); }
  catch (e) { msg(e.message, true); return; }
  const inputs = {};
  for (const { name, default: def } of G.runInputNames(graph)) {
    if (!name) continue;
    const v = window.prompt(`Input "${name}":`, def || '');
    if (v === null) { msg('Run cancelled'); return; }
    inputs[name] = v;
  }
  let result;
  try {
    result = await api(`/api/workflows/${encodeURIComponent(currentId)}/run`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ inputs }),
    });
  } catch (e) { msg(e.message, true); return; }
  msg('Ran');
  renderResults(result);
}

const _PILL = { ok: '#3ba55d', error: '#ff5555', skipped: '#888' };

function renderResults(result) {
  const host = $('wf-results');
  if (!host) return;
  host.style.display = '';
  host.innerHTML = '';
  const outs = document.createElement('div');
  outs.style.cssText = 'font-size:12px;margin-bottom:8px;';
  const outEntries = Object.entries(result.outputs || {});
  outs.innerHTML = '<b>Outputs:</b> ' + (outEntries.length
    ? outEntries.map(([k, v]) => `${escapeHtml(k)} = ${escapeHtml(String(v))}`).join(' · ')
    : '<span style="opacity:0.6">(none)</span>');
  host.appendChild(outs);
  (result.log || []).forEach((entry) => {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:8px;align-items:center;font-size:11px;padding:2px 0;cursor:pointer;';
    const pill = document.createElement('span');
    pill.textContent = entry.status;
    pill.style.cssText = `background:${_PILL[entry.status] || '#888'};color:#fff;border-radius:8px;`
      + 'padding:0 6px;font-size:9px;text-transform:uppercase;';
    const label = document.createElement('span');
    label.style.cssText = 'flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
    label.textContent = `${entry.node} (${entry.type}) ${entry.error || entry.output || ''}`;
    const ms = document.createElement('span');
    ms.style.cssText = 'opacity:0.6;'; ms.textContent = `${entry.ms}ms`;
    row.appendChild(pill); row.appendChild(label); row.appendChild(ms);
    row.addEventListener('click', () => selectNode(entry.node));
    host.appendChild(row);
  });
}

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

// ── triggers panel: schedule/event/webhook a saved workflow via /api/tasks ──
const _EVENTS = ['message_sent', 'session_created'];

async function renderTriggers() {
  const host = $('wf-triggers');
  if (!host) return;
  host.style.display = '';
  host.innerHTML = '';
  if (!currentId) {
    host.textContent = 'Save the workflow first to add triggers.';
    host.style.opacity = '0.6';
    return;
  }
  host.style.opacity = '';
  let all = [];
  try { all = (await api('/api/tasks')).tasks || []; }   // list_tasks -> {"tasks": [...]}
  catch (e) { host.textContent = 'Could not load triggers: ' + e.message; return; }
  const mine = all.filter((t) => t.task_type === 'workflow' && t.action === currentId);
  const list = document.createElement('div');
  mine.forEach((t) => {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:6px;align-items:center;font-size:12px;padding:2px 0;';
    const label = document.createElement('span');
    label.style.cssText = 'flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
    const when = t.trigger_type === 'schedule' ? (t.schedule || 'schedule')
      : t.trigger_type === 'event' ? ('on ' + (t.trigger_event || 'event'))
      : 'webhook';
    label.textContent = `${t.name || t.id} · ${when}`;
    row.appendChild(label);
    if (t.trigger_type === 'webhook' && t.webhook_token) {
      row.appendChild(_btn('Copy URL', () => {
        const url = `${location.origin}/api/tasks/${t.id}/webhook/${t.webhook_token}`;
        if (navigator.clipboard) navigator.clipboard.writeText(url);
        msg('Webhook URL copied');
      }));
    }
    row.appendChild(_btn('Delete', async () => {
      try { await api('/api/tasks/' + encodeURIComponent(t.id), { method: 'DELETE' }); renderTriggers(); }
      catch (e) { msg('Delete failed: ' + e.message, true); }
    }));
    list.appendChild(row);
  });
  if (!mine.length) {
    const e = document.createElement('div');
    e.style.cssText = 'font-size:12px;opacity:0.6;'; e.textContent = 'No triggers yet.';
    list.appendChild(e);
  }
  host.appendChild(list);
  host.appendChild(_triggerForm());
}

function _btn(label, fn) {
  const b = document.createElement('button'); b.type = 'button'; b.textContent = label;
  b.style.cssText = 'font-size:11px;'; b.addEventListener('click', fn); return b;
}

function _triggerForm() {
  const form = document.createElement('div');
  form.style.cssText = 'margin-top:8px;border-top:1px dashed var(--border);padding-top:6px;font-size:12px;';
  const typeSel = document.createElement('select');
  ['schedule', 'event', 'webhook'].forEach((v) => { const o = document.createElement('option'); o.value = v; o.textContent = v; typeSel.appendChild(o); });
  const cfg = document.createElement('span');
  function renderCfg() {
    cfg.innerHTML = '';
    if (typeSel.value === 'schedule') {
      const s = document.createElement('select'); s.id = 'wf-tr-sched';
      ['once', 'daily', 'weekly'].forEach((v) => { const o = document.createElement('option'); o.value = v; o.textContent = v; s.appendChild(o); });
      const time = document.createElement('input'); time.id = 'wf-tr-time'; time.value = '09:00'; time.style.width = '60px';
      cfg.appendChild(s); cfg.appendChild(time);
    } else if (typeSel.value === 'event') {
      const s = document.createElement('select'); s.id = 'wf-tr-event';
      _EVENTS.forEach((v) => { const o = document.createElement('option'); o.value = v; o.textContent = v; s.appendChild(o); });
      const n = document.createElement('input'); n.id = 'wf-tr-count'; n.type = 'number'; n.value = '1'; n.style.width = '48px';
      cfg.appendChild(s); cfg.appendChild(n);
    }
  }
  typeSel.addEventListener('change', renderCfg); renderCfg();
  // fixed-inputs mini-editor: one row per input node
  const inputsWrap = document.createElement('div');
  inputsWrap.style.cssText = 'margin:4px 0;';
  const inputNodes = graph.nodes.filter((n) => n.type === 'input');
  inputNodes.forEach((n) => {
    const nm = (n.config || {}).name || n.id;
    const row = document.createElement('div'); row.style.cssText = 'display:flex;gap:4px;align-items:center;';
    const lab = document.createElement('span'); lab.textContent = nm; lab.style.cssText = 'width:80px;opacity:0.7;';
    const val = document.createElement('input'); val.dataset.input = nm; val.value = (n.config || {}).default || ''; val.style.flex = '1';
    row.appendChild(lab); row.appendChild(val); inputsWrap.appendChild(row);
  });
  form.appendChild(typeSel); form.appendChild(cfg);
  form.appendChild(inputsWrap);
  form.appendChild(_btn('Add trigger', () => _addTrigger(typeSel.value, inputsWrap)));
  return form;
}

async function _addTrigger(triggerType, inputsWrap) {
  const fixed = {};
  inputsWrap.querySelectorAll('input[data-input]').forEach((el) => {
    if (el.value !== '') fixed[el.dataset.input] = el.value;
  });
  const body = {
    task_type: 'workflow', action: currentId, trigger_type: triggerType,
    prompt: JSON.stringify(fixed),
  };
  if (triggerType === 'schedule') {
    body.schedule = ($('wf-tr-sched') || {}).value || 'daily';
    body.scheduled_time = ($('wf-tr-time') || {}).value || '09:00';
  } else if (triggerType === 'event') {
    body.trigger_event = ($('wf-tr-event') || {}).value || 'message_sent';
    body.trigger_count = parseInt(($('wf-tr-count') || {}).value || '1', 10);
  }
  try {
    await api('/api/tasks', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    msg('Trigger added'); renderTriggers();
  } catch (e) { msg('Add trigger failed: ' + e.message, true); }
}

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
  // Admin-only: reveal BOTH the icon-rail button and the sidebar Tools entry.
  isAdmin().then((ok) => {
    if (!ok) return;
    ['rail-workflows', 'tool-workflows-btn'].forEach((id) => {
      const b = $(id); if (b) b.style.display = '';
    });
  });
  const rail = $('rail-workflows'); if (rail) rail.addEventListener('click', openWorkflows);
  const sideBtn = $('tool-workflows-btn'); if (sideBtn) sideBtn.addEventListener('click', openWorkflows);
  const x = $('wf-close'); if (x) x.addEventListener('click', closeWorkflows);
  const add = $('wf-add');
  if (add) add.addEventListener('click', () => {
    const type = window.prompt('Node type: input, template, llm, tool, or output', 'template');
    if (!type || !G.NODE_TYPES.includes(type)) return;
    const canvas = $('wf-canvas');
    const node = G.addNode(graph, type,
      40 + (canvas ? canvas.scrollLeft : 0), 40 + (canvas ? canvas.scrollTop : 0));
    selectNode(node.id);
  });
  const saveBtn = $('wf-save');
  if (saveBtn) saveBtn.addEventListener('click', () => {
    save().then((s) => msg(`Saved (${s.id})`)).catch((e) => msg(e.message, true));
  });
  const runBtn = $('wf-run');
  if (runBtn) runBtn.addEventListener('click', run);
  const trBtn = $('wf-triggers-btn');
  if (trBtn) trBtn.addEventListener('click', () => {
    const host = $('wf-triggers');
    if (host && host.style.display === 'none') renderTriggers();
    else if (host) host.style.display = 'none';
  });
  document.addEventListener('keydown', (e) => {
    const modal = $('workflows-modal');
    if (!modal || modal.classList.contains('hidden')) return;   // only when the editor is open
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || e.target.isContentEditable) return;  // don't hijack typing
    if ((e.key === 'Delete' || e.key === 'Backspace') && selected && selected.kind === 'edge') {
      const ed = selected.edge;
      G.removeEdge(graph, ed.from_node, ed.from_port, ed.to_node, ed.to_port);
      selected = null;
      render();
      e.preventDefault();
    }
  });
  Modals.register('workflows-modal', {
    railBtnId: 'rail-workflows', sidebarBtnId: 'tool-workflows-btn', closeFn: closeWorkflows,
  });
}

document.addEventListener('DOMContentLoaded', init);
