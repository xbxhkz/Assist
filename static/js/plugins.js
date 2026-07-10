// Plugins & Connectors modal: aggregates built-in MCP servers, user-added MCP
// servers, and connectors (integrations) into one unified list. Mirrors the
// plain-script style of help.js. Per-row action buttons and the
// Add-from-catalog picker/forms are built dynamically (createElement +
// addEventListener only — no inline handlers, per CSP).
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
    closeAddPanel();
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

    const actions = document.createElement('span');
    actions.className = 'plugin-row-actions';
    actions.dataset.pluginActions = item.id;
    actions.style.cssText = 'flex-shrink:0;display:flex;gap:4px;margin-left:8px;';
    renderActions(item, actions);
    row.appendChild(actions);

    return row;
  }

  function actionBtn(label, danger) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = danger ? 'admin-btn-delete' : 'admin-btn-sm';
    b.textContent = label;
    return b;
  }

  // Fills the per-row actions container, dispatched by row.type.
  function renderActions(item, actions) {
    if (item.type === 'builtin') {
      const isEnabled = item.status !== 'disabled';
      const toggleBtn = actionBtn(isEnabled ? 'Disable' : 'Enable');
      toggleBtn.addEventListener('click', async () => {
        toggleBtn.disabled = true;
        try {
          const res = await fetch(`/api/mcp/builtins/${item.id}/toggle`, {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: !isEnabled }),
          });
          if (!res.ok) throw new Error(String(res.status));
          await refresh();
        } catch (e) {
          setMsg(`Failed to toggle ${item.name}: ${e.message}`, true);
          toggleBtn.disabled = false;
        }
      });
      actions.appendChild(toggleBtn);
      return;
    }

    if (item.type === 'mcp') {
      if (item.needs_oauth) {
        const authBtn = actionBtn('Authorize');
        authBtn.addEventListener('click', () => {
          window.open(`/api/mcp/oauth/authorize/${item.id}`, '_blank');
        });
        actions.appendChild(authBtn);
      }

      const reconnectBtn = actionBtn('Reconnect');
      reconnectBtn.addEventListener('click', async () => {
        reconnectBtn.disabled = true;
        try {
          const res = await fetch(`/api/mcp/servers/${item.id}/reconnect`, {
            method: 'POST',
            credentials: 'same-origin',
          });
          if (!res.ok) throw new Error(String(res.status));
          await refresh();
        } catch (e) {
          setMsg(`Reconnect failed for ${item.name}: ${e.message}`, true);
          reconnectBtn.disabled = false;
        }
      });
      actions.appendChild(reconnectBtn);

      const toolsBtn = actionBtn('Tools');
      toolsBtn.addEventListener('click', async () => {
        try {
          const res = await fetch(`/api/mcp/servers/${item.id}/tools`, { credentials: 'same-origin' });
          if (!res.ok) throw new Error(String(res.status));
          const tools = await res.json();
          const list = Array.isArray(tools) ? tools : [];
          const names = list.map((t) => t.name).join(', ');
          setMsg(`${item.name}: ${list.length} tools${names ? ' — ' + names : ''}`, false);
        } catch (e) {
          setMsg(`Failed to load tools for ${item.name}: ${e.message}`, true);
        }
      });
      actions.appendChild(toolsBtn);

      const deleteBtn = actionBtn('Delete', true);
      deleteBtn.addEventListener('click', async () => {
        if (!confirm(`Delete MCP server "${item.name}"?`)) return;
        try {
          const res = await fetch(`/api/mcp/servers/${item.id}`, {
            method: 'DELETE',
            credentials: 'same-origin',
          });
          if (!res.ok) throw new Error(String(res.status));
          await refresh();
        } catch (e) {
          setMsg(`Failed to delete ${item.name}: ${e.message}`, true);
        }
      });
      actions.appendChild(deleteBtn);
      return;
    }

    if (item.type === 'connector') {
      const testBtn = actionBtn('Test');
      testBtn.addEventListener('click', async () => {
        testBtn.disabled = true;
        try {
          const res = await fetch(`/api/auth/integrations/${item.id}/test`, {
            method: 'POST',
            credentials: 'same-origin',
          });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) throw new Error((data && data.detail) || String(res.status));
          setMsg(`${item.name}: ${data.message || (data.ok ? 'OK' : 'Failed')}`, data.ok === false);
        } catch (e) {
          setMsg(`Test failed for ${item.name}: ${e.message}`, true);
        } finally {
          testBtn.disabled = false;
        }
      });
      actions.appendChild(testBtn);

      const deleteBtn = actionBtn('Delete', true);
      deleteBtn.addEventListener('click', async () => {
        if (!confirm(`Delete connector "${item.name}"?`)) return;
        try {
          const res = await fetch(`/api/auth/integrations/${item.id}`, {
            method: 'DELETE',
            credentials: 'same-origin',
          });
          if (!res.ok) throw new Error(String(res.status));
          await refresh();
        } catch (e) {
          setMsg(`Failed to delete ${item.name}: ${e.message}`, true);
        }
      });
      actions.appendChild(deleteBtn);
    }
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

  // ── Add-from-catalog: a small picker + forms built dynamically ──

  let addPanelEl = null;

  function ensureAddPanel() {
    if (addPanelEl) return addPanelEl;
    addPanelEl = document.createElement('div');
    addPanelEl.id = 'plugins-add-panel';
    addPanelEl.style.cssText = 'border:1px solid var(--border);border-radius:6px;padding:10px;margin-bottom:8px;display:none;';
    const list = $('plugins-list');
    if (list && list.parentNode) list.parentNode.insertBefore(addPanelEl, list);
    return addPanelEl;
  }

  function closeAddPanel() {
    if (!addPanelEl) return;
    addPanelEl.style.display = 'none';
    addPanelEl.innerHTML = '';
  }

  function fieldRow(labelText, inputEl) {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'margin-bottom:6px;';
    const label = document.createElement('label');
    label.textContent = labelText;
    label.style.cssText = 'display:block;font-size:11px;opacity:0.65;margin-bottom:2px;';
    wrap.appendChild(label);
    wrap.appendChild(inputEl);
    return wrap;
  }

  function textInput(placeholder) {
    const inp = document.createElement('input');
    inp.type = 'text';
    inp.placeholder = placeholder || '';
    inp.style.cssText = 'width:100%;box-sizing:border-box;padding:4px 6px;border:1px solid var(--border);border-radius:4px;background:var(--bg, transparent);color:var(--fg);font-size:12px;';
    return inp;
  }

  function formMsg() {
    const el = document.createElement('div');
    el.style.cssText = 'font-size:11px;margin:4px 0;min-height:14px;';
    return el;
  }

  function backButton(onClick) {
    const b = actionBtn('← Back');
    b.style.marginBottom = '6px';
    b.addEventListener('click', onClick);
    return b;
  }

  function heading(text) {
    const h = document.createElement('div');
    h.textContent = text;
    h.style.cssText = 'font-weight:600;font-size:12px;margin:2px 0 8px;';
    return h;
  }

  function showAddPicker() {
    const panel = ensureAddPanel();
    panel.innerHTML = '';
    panel.style.display = '';

    panel.appendChild(heading('Add…'));

    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:6px;flex-wrap:wrap;';
    const choices = [
      ['Custom MCP server', showCustomMcpForm],
      ['Connector preset', showConnectorPresetForm],
      ['Custom connector', showCustomConnectorForm],
    ];
    choices.forEach(([label, fn]) => {
      const b = actionBtn(label);
      b.addEventListener('click', () => fn(panel));
      row.appendChild(b);
    });
    const cancelBtn = actionBtn('Cancel');
    cancelBtn.addEventListener('click', closeAddPanel);
    row.appendChild(cancelBtn);
    panel.appendChild(row);
  }

  function showCustomMcpForm(panel) {
    panel.innerHTML = '';
    panel.appendChild(backButton(showAddPicker));
    panel.appendChild(heading('Add custom MCP server'));

    const nameInp = textInput('e.g. my-server');
    const cmdInp = textInput('e.g. npx');
    const argsInp = textInput('["-y","package-name"]');
    argsInp.value = '[]';
    const envInp = textInput('{"KEY":"value"}');
    envInp.value = '{}';

    panel.appendChild(fieldRow('Name', nameInp));
    panel.appendChild(fieldRow('Command', cmdInp));
    panel.appendChild(fieldRow('Args (JSON array)', argsInp));
    panel.appendChild(fieldRow('Env (JSON object)', envInp));

    const msg = formMsg();
    panel.appendChild(msg);

    const submitBtn = actionBtn('Add server');
    submitBtn.addEventListener('click', async () => {
      const name = nameInp.value.trim();
      const command = cmdInp.value.trim();
      if (!name) { msg.textContent = 'Name is required'; msg.style.color = 'var(--red, #ff5555)'; return; }
      if (!command) { msg.textContent = 'Command is required'; msg.style.color = 'var(--red, #ff5555)'; return; }
      const args = argsInp.value.trim() || '[]';
      const env = envInp.value.trim() || '{}';
      try { JSON.parse(args); } catch { msg.textContent = 'Args must be valid JSON'; msg.style.color = 'var(--red, #ff5555)'; return; }
      try { JSON.parse(env); } catch { msg.textContent = 'Env must be valid JSON'; msg.style.color = 'var(--red, #ff5555)'; return; }

      const fd = new FormData();
      fd.append('name', name);
      fd.append('transport', 'stdio');
      fd.append('command', command);
      fd.append('args', args);
      fd.append('env', env);

      submitBtn.disabled = true;
      msg.textContent = 'Adding…'; msg.style.color = '';
      try {
        const res = await fetch('/api/mcp/servers', { method: 'POST', credentials: 'same-origin', body: fd });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error((data && data.detail) || String(res.status));
        closeAddPanel();
        setMsg(`Added MCP server "${name}"`, false);
        await refresh();
      } catch (e) {
        msg.textContent = `Failed: ${e.message}`; msg.style.color = 'var(--red, #ff5555)';
        submitBtn.disabled = false;
      }
    });
    panel.appendChild(submitBtn);
  }

  async function showConnectorPresetForm(panel) {
    panel.innerHTML = '';
    panel.appendChild(backButton(showAddPicker));
    panel.appendChild(heading('Add connector from preset'));

    const loading = document.createElement('div');
    loading.textContent = 'Loading presets…';
    loading.style.cssText = 'font-size:11px;opacity:0.7;';
    panel.appendChild(loading);

    let presets = {};
    try {
      const res = await fetch('/api/auth/integrations/presets', { credentials: 'same-origin' });
      if (!res.ok) throw new Error(String(res.status));
      const data = await res.json();
      presets = (data && data.presets) || {};
    } catch (e) {
      loading.textContent = `Failed to load presets: ${e.message}`;
      loading.style.color = 'var(--red, #ff5555)';
      return;
    }
    loading.remove();

    const keys = Object.keys(presets);
    if (!keys.length) {
      const empty = document.createElement('div');
      empty.className = 'admin-empty';
      empty.textContent = 'No presets available';
      panel.appendChild(empty);
      return;
    }

    const select = document.createElement('select');
    select.style.cssText = 'width:100%;padding:4px 6px;margin-bottom:6px;';
    const blankOpt = document.createElement('option');
    blankOpt.value = '';
    blankOpt.textContent = '— choose a preset —';
    select.appendChild(blankOpt);
    keys.forEach((key) => {
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = (presets[key] && presets[key].name) || key;
      select.appendChild(opt);
    });
    panel.appendChild(fieldRow('Preset', select));

    const nameInp = textInput('Name');
    const baseUrlInp = textInput('Base URL');
    const apiKeyInp = textInput('API key (optional)');
    apiKeyInp.type = 'password';
    panel.appendChild(fieldRow('Name', nameInp));
    panel.appendChild(fieldRow('Base URL', baseUrlInp));
    panel.appendChild(fieldRow('API key', apiKeyInp));

    select.addEventListener('change', () => {
      const p = presets[select.value];
      if (!p) return;
      nameInp.value = p.name || select.value;
      baseUrlInp.value = p.base_url || '';
    });

    const msg = formMsg();
    panel.appendChild(msg);

    const submitBtn = actionBtn('Add connector');
    submitBtn.addEventListener('click', async () => {
      const name = nameInp.value.trim();
      const baseUrl = baseUrlInp.value.trim();
      if (!name) { msg.textContent = 'Name is required'; msg.style.color = 'var(--red, #ff5555)'; return; }
      submitBtn.disabled = true;
      msg.textContent = 'Adding…'; msg.style.color = '';
      try {
        const res = await fetch('/api/auth/integrations', {
          method: 'POST',
          credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name,
            base_url: baseUrl,
            api_key: apiKeyInp.value,
            preset: select.value || undefined,
            enabled: true,
          }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error((data && data.detail) || String(res.status));
        closeAddPanel();
        setMsg(`Added connector "${name}"`, false);
        await refresh();
      } catch (e) {
        msg.textContent = `Failed: ${e.message}`; msg.style.color = 'var(--red, #ff5555)';
        submitBtn.disabled = false;
      }
    });
    panel.appendChild(submitBtn);
  }

  function showCustomConnectorForm(panel) {
    panel.innerHTML = '';
    panel.appendChild(backButton(showAddPicker));
    panel.appendChild(heading('Add custom connector'));

    const nameInp = textInput('Name');
    const baseUrlInp = textInput('Base URL');
    const apiKeyInp = textInput('API key (optional)');
    apiKeyInp.type = 'password';
    panel.appendChild(fieldRow('Name', nameInp));
    panel.appendChild(fieldRow('Base URL', baseUrlInp));
    panel.appendChild(fieldRow('API key', apiKeyInp));

    const msg = formMsg();
    panel.appendChild(msg);

    const submitBtn = actionBtn('Add connector');
    submitBtn.addEventListener('click', async () => {
      const name = nameInp.value.trim();
      const baseUrl = baseUrlInp.value.trim();
      if (!name) { msg.textContent = 'Name is required'; msg.style.color = 'var(--red, #ff5555)'; return; }
      if (!baseUrl) { msg.textContent = 'Base URL is required'; msg.style.color = 'var(--red, #ff5555)'; return; }
      submitBtn.disabled = true;
      msg.textContent = 'Adding…'; msg.style.color = '';
      try {
        const res = await fetch('/api/auth/integrations', {
          method: 'POST',
          credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, base_url: baseUrl, api_key: apiKeyInp.value, enabled: true }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error((data && data.detail) || String(res.status));
        closeAddPanel();
        setMsg(`Added connector "${name}"`, false);
        await refresh();
      } catch (e) {
        msg.textContent = `Failed: ${e.message}`; msg.style.color = 'var(--red, #ff5555)';
        submitBtn.disabled = false;
      }
    });
    panel.appendChild(submitBtn);
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
      fetchJson('/api/auth/integrations').then((d) => ({ ok: true, items: normalizeIntegrations(d) }), (e) => ({ ok: false, error: e })),
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
    $('plugins-add-btn')?.addEventListener('click', () => {
      const panel = ensureAddPanel();
      if (panel.style.display === 'none') showAddPicker();
      else closeAddPanel();
    });
  });

  window.PluginsModal = { open, close, refresh };
})();
