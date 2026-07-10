// Plugins & Connectors modal: aggregates built-in MCP servers, user-added MCP
// servers, and connectors (integrations) into one unified list. Mirrors the
// plain-script style of help.js. Row action buttons are filled in by a later
// task — this only renders a placeholder container per row.
(function () {
  function $(id) { return document.getElementById(id); }

  const TYPE_LABEL = { builtin: 'Built-in', mcp: 'MCP', connector: 'Connector' };

  function statusColor(status) {
    if (status === 'connected') return 'var(--green, #50fa7b)';
    if (status === 'error') return 'var(--red, #ff5555)';
    if (status === 'disabled') return 'color-mix(in srgb, var(--fg) 40%, transparent)';
    return 'color-mix(in srgb, var(--fg) 50%, transparent)';
  }

  function open() {
    const m = $('plugins-modal');
    if (m) m.classList.remove('hidden');
    refresh();
  }

  function close() {
    const m = $('plugins-modal');
    if (m) m.classList.add('hidden');
  }

  function setMsg(text, isError) {
    const el = $('plugins-msg');
    if (!el) return;
    el.textContent = text || '';
    el.style.color = isError ? 'var(--red, #ff5555)' : '';
  }

  function normalizeBuiltins(data) {
    const items = (data && data.builtins) || [];
    return items.map((b) => ({
      id: b.id,
      name: b.name,
      type: 'builtin',
      status: b.status,
      tools: b.tool_count,
      error: b.error,
    }));
  }

  function normalizeServers(data) {
    const items = Array.isArray(data) ? data : [];
    return items.map((s) => ({
      id: s.id,
      name: s.name,
      type: 'mcp',
      status: s.status,
      tools: s.tool_count,
      error: s.error,
      needs_oauth: s.needs_oauth,
      has_oauth: s.has_oauth,
    }));
  }

  function normalizeIntegrations(data) {
    const items = (data && data.integrations) || [];
    return items.map((i) => ({
      id: i.id,
      name: i.name,
      type: 'connector',
      status: i.enabled ? 'connected' : 'disabled',
      tools: undefined,
      error: undefined,
    }));
  }

  function renderRow(item) {
    const row = document.createElement('div');
    row.className = 'list-item';
    row.dataset.pluginId = item.id;
    row.dataset.pluginType = item.type;

    const badge = document.createElement('span');
    badge.textContent = TYPE_LABEL[item.type] || item.type;
    badge.style.cssText = 'flex-shrink:0;font-size:10px;font-weight:600;opacity:0.7;border:1px solid currentColor;border-radius:4px;padding:1px 5px;margin-right:8px;';
    row.appendChild(badge);

    const label = document.createElement('span');
    label.className = 'grow';
    const dot = document.createElement('span');
    dot.style.cssText = `display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:${statusColor(item.status)};`;
    label.appendChild(dot);
    const nameSpan = document.createElement('span');
    nameSpan.textContent = item.name;
    label.appendChild(nameSpan);
    const statusSpan = document.createElement('span');
    statusSpan.style.cssText = 'opacity:0.6;font-size:12px;margin-left:6px;';
    let statusText = item.status || 'unknown';
    if (typeof item.tools === 'number') statusText += ` · ${item.tools} tools`;
    if (item.error) statusText += ` · ${item.error}`;
    statusSpan.textContent = statusText;
    label.appendChild(statusSpan);
    row.appendChild(label);

    // Placeholder actions container -- filled in by a later task.
    const actions = document.createElement('span');
    actions.className = 'plugin-row-actions';
    actions.dataset.pluginActions = item.id;
    row.appendChild(actions);

    return row;
  }

  function render(items) {
    const list = $('plugins-list');
    if (!list) return;
    list.innerHTML = '';
    if (!items.length) {
      const empty = document.createElement('div');
      empty.className = 'admin-empty';
      empty.textContent = 'No plugins or connectors found';
      list.appendChild(empty);
      return;
    }
    items.forEach((item) => list.appendChild(renderRow(item)));
  }

  async function fetchJson(url) {
    const res = await fetch(url, { credentials: 'same-origin' });
    if (!res.ok) throw new Error(`${url}: ${res.status}`);
    return res.json();
  }

  async function refresh() {
    setMsg('Loading…', false);
    const results = await Promise.all([
      fetchJson('/api/mcp/builtins').then((d) => ({ ok: true, items: normalizeBuiltins(d) }), (e) => ({ ok: false, error: e })),
      fetchJson('/api/mcp/servers').then((d) => ({ ok: true, items: normalizeServers(d) }), (e) => ({ ok: false, error: e })),
      fetchJson('/api/integrations').then((d) => ({ ok: true, items: normalizeIntegrations(d) }), (e) => ({ ok: false, error: e })),
    ]);

    const items = [];
    const errors = [];
    results.forEach((r) => {
      if (r.ok) items.push(...r.items);
      else errors.push(r.error && r.error.message ? r.error.message : String(r.error));
    });

    render(items);
    setMsg(errors.length ? `Some sources failed to load: ${errors.join('; ')}` : '', errors.length > 0);
  }

  document.addEventListener('DOMContentLoaded', () => {
    $('tool-plugins-btn')?.addEventListener('click', open);
    $('close-plugins-modal')?.addEventListener('click', close);
    $('plugins-modal')?.addEventListener('click', (e) => {
      if (e.target === $('plugins-modal')) close();
    });
    $('plugins-refresh-btn')?.addEventListener('click', refresh);
  });

  window.PluginsModal = { open, close, refresh };
})();
