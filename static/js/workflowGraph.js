// Pure, DOM-free workflow graph core: node/port model, cycle-guarded wiring,
// auto-layout, and engine-JSON round-trip. Unit-tested in Node. The {slot}
// derivation MUST match Python's model.slots_of (re \{(\w+)\}, ordered-unique).
export const NODE_TYPES = ['input', 'template', 'llm', 'tool', 'output', 'branch'];

const _SLOT_RE = /\{([\p{L}\p{N}_]+)\}/gu;
const _SLOT_SOURCE = { template: 'template', llm: 'prompt', tool: 'args' };
const _OUTPUT_PORT = { input: 'value', template: 'text', llm: 'text', tool: 'result' };

export function slotsOf(text) {
  const out = [];
  const s = text == null ? '' : String(text);
  let m;
  _SLOT_RE.lastIndex = 0;
  while ((m = _SLOT_RE.exec(s)) !== null) {
    if (!out.includes(m[1])) out.push(m[1]);
  }
  return out;
}

export function inputPortsOf(node) {
  const t = node && node.type;
  if (t === 'output' || t === 'branch') return ['value'];
  const key = _SLOT_SOURCE[t];
  if (!key) return [];                 // input (and unknown) take no wires
  return slotsOf((node.config || {})[key]);
}

export function outputPortsOf(node) {
  if (node && node.type === 'branch') {
    const cases = (node.config || {}).cases;
    const list = Array.isArray(cases) ? cases.filter((c) => typeof c === 'string' && c.trim()) : [];
    return list.concat(['else']);
  }
  const p = _OUTPUT_PORT[node && node.type];
  return p ? [p] : [];
}

export function createGraph(wf = {}) {
  const nodes = (wf.nodes || []).map((n) => ({
    id: n.id, type: n.type, config: Object.assign({}, n.config || {}),
    x: typeof n.x === 'number' ? n.x : 0, y: typeof n.y === 'number' ? n.y : 0,
  }));
  const edges = (wf.edges || []).map((e) => ({
    from_node: e.from_node, from_port: e.from_port, to_node: e.to_node, to_port: e.to_port,
  }));
  return { id: wf.id || '', name: wf.name || '', nodes, edges };
}

export function nodeById(graph, id) {
  return graph.nodes.find((n) => n.id === id);
}

export function addNode(graph, type, x = 40, y = 40) {
  let i = 1;
  while (nodeById(graph, `${type}${i}`)) i += 1;
  const node = { id: `${type}${i}`, type, config: {}, x, y };
  graph.nodes.push(node);
  return node;
}

// Drop any edge whose to_port is no longer a valid input port of `id`
// (e.g. after a template's {slots} changed). Shared by removeNode/setConfig.
function _pruneEdges(graph, id) {
  const node = nodeById(graph, id);
  if (!node) return;
  const ins = inputPortsOf(node);
  const outs = outputPortsOf(node);
  graph.edges = graph.edges.filter(
    (e) => (e.to_node !== id || ins.includes(e.to_port))
        && (e.from_node !== id || outs.includes(e.from_port)),
  );
}

export function removeNode(graph, id) {
  graph.nodes = graph.nodes.filter((n) => n.id !== id);
  graph.edges = graph.edges.filter((e) => e.from_node !== id && e.to_node !== id);
}

export function setConfig(graph, id, config) {
  const node = nodeById(graph, id);
  if (!node) return;
  node.config = Object.assign({}, config);
  _pruneEdges(graph, id);
}

export function setNodeId(graph, oldId, newId) {
  const id = (newId || '').trim();
  if (!id) throw new Error('node id cannot be empty');
  if (id !== oldId && nodeById(graph, id)) throw new Error('duplicate node id');
  const node = nodeById(graph, oldId);
  if (!node) throw new Error('unknown node');
  node.id = id;
  graph.edges.forEach((e) => {
    if (e.from_node === oldId) e.from_node = id;
    if (e.to_node === oldId) e.to_node = id;
  });
}

export function setNodePos(graph, id, x, y) {
  const node = nodeById(graph, id);
  if (node) { node.x = x; node.y = y; }
}

export function setName(graph, name) { graph.name = name || ''; }

export function topoOrder(graph) {
  const ids = graph.nodes.map((n) => n.id);
  const indeg = {};
  const adj = {};
  ids.forEach((i) => { indeg[i] = 0; adj[i] = []; });
  graph.edges.forEach((e) => {
    if (e.from_node in indeg && e.to_node in indeg) {
      adj[e.from_node].push(e.to_node);
      indeg[e.to_node] += 1;
    }
  });
  const q = ids.filter((i) => indeg[i] === 0);
  const order = [];
  while (q.length) {
    const cur = q.shift();
    order.push(cur);
    adj[cur].forEach((nxt) => { indeg[nxt] -= 1; if (indeg[nxt] === 0) q.push(nxt); });
  }
  if (order.length !== ids.length) throw new Error('cycle detected');
  return order;
}

// Assign x/y by longest-path depth (column) and stacking index (row).
export function autoLayout(graph) {
  const order = topoOrder(graph);          // throws on cycle
  const preds = {};
  graph.nodes.forEach((n) => { preds[n.id] = []; });
  graph.edges.forEach((e) => { if (preds[e.to_node]) preds[e.to_node].push(e.from_node); });
  const layer = {};
  order.forEach((id) => {
    layer[id] = preds[id].length ? Math.max(...preds[id].map((p) => layer[p])) + 1 : 0;
  });
  const rowInLayer = {};
  graph.nodes.forEach((n) => {
    const L = layer[n.id] || 0;
    const row = rowInLayer[L] || 0;
    rowInLayer[L] = row + 1;
    n.x = 40 + L * 210;
    n.y = 40 + row * 120;
  });
}

export function runInputNames(graph) {
  return graph.nodes
    .filter((n) => n.type === 'input')
    .map((n) => ({ name: (n.config || {}).name || '', default: (n.config || {}).default || '' }));
}

export function toJSON(graph) {
  return {
    id: graph.id, name: graph.name,
    nodes: graph.nodes.map((n) => ({
      id: n.id, type: n.type, config: Object.assign({}, n.config), x: n.x, y: n.y,
    })),
    edges: graph.edges.map((e) => ({
      from_node: e.from_node, from_port: e.from_port, to_node: e.to_node, to_port: e.to_port,
    })),
  };
}

// ── edges: cycle-guarded wiring ──

// Can `to` already reach `from` by following edges? Used to reject an edge
// from->to that would close a cycle.
function _reaches(graph, start, target) {
  const seen = new Set();
  const stack = [start];
  while (stack.length) {
    const cur = stack.pop();
    if (cur === target) return true;
    if (seen.has(cur)) continue;
    seen.add(cur);
    graph.edges.forEach((e) => { if (e.from_node === cur) stack.push(e.to_node); });
  }
  return false;
}

export function canConnect(graph, fromNode, fromPort, toNode, toPort) {
  if (fromNode === toNode) return false;
  const from = nodeById(graph, fromNode);
  const to = nodeById(graph, toNode);
  if (!from || !to) return false;
  if (!outputPortsOf(from).includes(fromPort)) return false;
  if (!inputPortsOf(to).includes(toPort)) return false;
  if (_reaches(graph, toNode, fromNode)) return false;   // would create a cycle
  return true;
}

export function addEdge(graph, fromNode, fromPort, toNode, toPort) {
  if (!canConnect(graph, fromNode, fromPort, toNode, toPort)) return false;
  // one edge per input port: drop any existing wire into (toNode,toPort)
  graph.edges = graph.edges.filter((e) => !(e.to_node === toNode && e.to_port === toPort));
  graph.edges.push({ from_node: fromNode, from_port: fromPort, to_node: toNode, to_port: toPort });
  return true;
}

export function removeEdge(graph, fromNode, fromPort, toNode, toPort) {
  graph.edges = graph.edges.filter((e) => !(
    e.from_node === fromNode && e.from_port === fromPort
    && e.to_node === toNode && e.to_port === toPort));
}

export function unwiredPorts(graph) {
  const wired = new Set(graph.edges.map((e) => `${e.to_node} ${e.to_port}`));
  const out = [];
  graph.nodes.forEach((n) => {
    inputPortsOf(n).forEach((port) => {
      if (!wired.has(`${n.id} ${port}`)) out.push({ node: n.id, port });
    });
  });
  return out;
}
