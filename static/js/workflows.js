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

function render() {
  const canvas = $('wf-canvas');
  if (!canvas) return;
  // wipe node divs (keep the <svg> overlay)
  Array.from(canvas.querySelectorAll('.wf-node')).forEach((el) => el.remove());
  graph.nodes.forEach((n) => canvas.appendChild(renderNode(n)));
  renderWires();
  renderInspector();
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
      render();                 // re-derive ports live (e.g. new {slot})
    });
    host.appendChild(label); host.appendChild(field);
  });
}
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
  const add = $('wf-add');
  if (add) add.addEventListener('click', () => {
    const type = window.prompt('Node type: input, template, llm, tool, or output', 'template');
    if (!type || !G.NODE_TYPES.includes(type)) return;
    const canvas = $('wf-canvas');
    const node = G.addNode(graph, type,
      40 + (canvas ? canvas.scrollLeft : 0), 40 + (canvas ? canvas.scrollTop : 0));
    selectNode(node.id);
  });
  Modals.register('workflows-modal', {
    railBtnId: 'rail-workflows', sidebarBtnId: 'tool-workflows-btn', closeFn: closeWorkflows,
  });
}

document.addEventListener('DOMContentLoaded', init);
