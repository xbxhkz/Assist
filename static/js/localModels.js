// Minimal Local Models UI (Phase 3a): list local GGUF files, serve/stop one at
// a time, and show live status. Mirrors the Cookbook modal conventions.
(function () {
  function $(id) { return document.getElementById(id); }

  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
  }

  function fmtSize(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(1) + ' GB';
    if (n >= 1e6) return (n / 1e6).toFixed(0) + ' MB';
    return (n / 1e3).toFixed(0) + ' KB';
  }

  async function refresh() {
    const statusEl = $('localmodels-status');
    const listEl = $('localmodels-list');
    if (!listEl) return;
    let status = { running: false };
    try { status = await api('/api/localmodels/status'); } catch (e) {}
    statusEl.textContent = status.running
      ? `Running: ${status.model} (port ${status.port})`
      : 'No model running';
    let data = { models: [] };
    try { data = await api('/api/localmodels/models'); } catch (e) {}
    listEl.innerHTML = '';
    data.models.forEach((m) => {
      const row = document.createElement('div');
      row.className = 'list-item';
      const label = document.createElement('span');
      label.className = 'grow';
      label.textContent = `${m.name} — ${fmtSize(m.size)}`;
      const btn = document.createElement('button');
      const isRunning = status.running && status.model === m.name;
      btn.textContent = isRunning ? 'Stop' : 'Serve';
      btn.onclick = async () => {
        btn.disabled = true;
        try {
          if (isRunning) await api('/api/localmodels/stop', { method: 'POST' });
          else await api('/api/localmodels/serve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_path: m.path }),
          });
        } catch (e) { alert('Local model error: ' + e.message); }
        await refresh();
      };
      row.appendChild(label);
      row.appendChild(btn);
      listEl.appendChild(row);
    });
    if (!data.models.length) {
      listEl.innerHTML = '<div class="list-item"><span class="grow">No .gguf models found. Add one to the models folder.</span></div>';
    }
  }

  function open() {
    const modal = $('localmodels-modal');
    if (modal) { modal.classList.remove('hidden'); refresh(); }
  }
  function close() {
    const modal = $('localmodels-modal');
    if (modal) modal.classList.add('hidden');
  }

  document.addEventListener('DOMContentLoaded', () => {
    const openBtn = $('tool-localmodels-btn');
    if (openBtn) openBtn.addEventListener('click', open);
    const closeBtn = $('close-localmodels-modal');
    if (closeBtn) closeBtn.addEventListener('click', close);
  });

  window.LocalModels = { open, close, refresh };
})();
