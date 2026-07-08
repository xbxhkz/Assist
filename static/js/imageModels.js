// Minimal Image Models UI: pick a FLUX .gguf, choose CPU/GPU, serve it via the
// bundled sd-server. A served model auto-registers as an image endpoint, so the
// existing gallery/chat image generation then uses it. Mirrors localModels.js.
(function () {
  function $(id) { return document.getElementById(id); }

  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
  }

  let pickedPath = '';

  function refreshPicker() {
    try { window.modelsModule?.refreshModels?.(true); } catch (e) {}
  }

  function device() {
    const el = document.querySelector('input[name="imagemodels-device"]:checked');
    return el ? el.value : 'cpu';
  }

  function fmtBytes(n) {
    if (!n && n !== 0) return '';
    const gb = n / (1024 ** 3);
    return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(n / (1024 ** 2)).toFixed(0)} MB`;
  }

  async function serveModel(path, msgPrefix) {
    const msg = $('imagemodels-msg');
    if (msg) { msg.style.color = ''; msg.textContent = `${msgPrefix} image models load slowly — this can take a few minutes.`; }
    const body = { diffusion_model: path, device: device() };
    const steps = parseInt($('imagemodels-steps')?.value, 10);
    if (steps) body.steps = steps;
    try {
      await api('/api/imagemodels/serve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (msg) msg.textContent = '';
    } catch (e) {
      if (msg) { msg.style.color = 'var(--red)'; msg.textContent = e.message; }
      throw e;
    }
  }

  function renderList(models, st) {
    const listEl = $('imagemodels-list');
    if (!listEl) return;
    listEl.innerHTML = '';
    models.forEach((m) => {
      const row = document.createElement('div');
      row.className = 'list-item';
      const label = document.createElement('span');
      label.className = 'grow';
      label.textContent = `${m.name} — ${fmtBytes(m.size)}`;
      const btn = document.createElement('button');
      const isRunning = st.running && st.model === m.name;
      btn.textContent = isRunning ? 'Stop' : 'Serve';
      btn.onclick = async () => {
        btn.disabled = true;
        try {
          if (isRunning) {
            await api('/api/imagemodels/stop', { method: 'POST' });
          } else {
            btn.textContent = 'Starting…';
            await serveModel(m.path, `Starting ${m.name}…`);
          }
        } catch (e) {}
        await refresh();
        refreshPicker();
      };
      row.appendChild(label);
      row.appendChild(btn);
      listEl.appendChild(row);
    });
  }

  async function refresh() {
    const statusEl = $('imagemodels-status');
    const serveBtn = $('imagemodels-serve-btn');
    const stopBtn = $('imagemodels-stop-btn');
    if (!statusEl) return;
    let st = { running: false };
    try { st = await api('/api/imagemodels/status'); } catch (e) {}
    if (st.running) {
      statusEl.textContent = `Running: ${st.model} on ${st.device === 'gpu' ? 'GPU' : 'CPU'} (port ${st.port})`;
      if (stopBtn) stopBtn.style.display = '';
      if (serveBtn) serveBtn.style.display = 'none';
    } else {
      statusEl.textContent = 'No image model running';
      if (stopBtn) stopBtn.style.display = 'none';
      if (serveBtn) serveBtn.style.display = '';
    }
    let data = { models: [] };
    try { data = await api('/api/imagemodels/models'); } catch (e) {}
    renderList(data.models || [], st);
  }

  function setupBrowse() {
    const btn = $('imagemodels-browse-btn');
    if (!btn) return;
    if (!(window.pywebview && window.pywebview.api && window.pywebview.api.pick_image_model)) return;
    btn.style.display = '';
    btn.onclick = async () => {
      let p = '';
      try { p = await window.pywebview.api.pick_image_model(); } catch (e) {}
      if (!p) return;
      pickedPath = p;
      const pk = $('imagemodels-picked');
      if (pk) pk.textContent = p;
      const serveBtn = $('imagemodels-serve-btn');
      if (serveBtn) serveBtn.disabled = false;
    };
  }

  async function serve() {
    if (!pickedPath) return;
    const serveBtn = $('imagemodels-serve-btn');
    if (serveBtn) serveBtn.disabled = true;
    try {
      await serveModel(pickedPath, 'Starting…');
      await refresh();
      refreshPicker();
    } catch (e) {
      if (serveBtn) serveBtn.disabled = false;
    }
  }

  async function stop() {
    try { await api('/api/imagemodels/stop', { method: 'POST' }); } catch (e) {}
    await refresh();
    refreshPicker();
  }

  function open() { refresh(); setupBrowse(); }

  document.addEventListener('DOMContentLoaded', () => {
    $('imagemodels-serve-btn')?.addEventListener('click', serve);
    $('imagemodels-stop-btn')?.addEventListener('click', stop);
    // Refresh image status whenever the Local Models modal opens.
    $('tool-localmodels-btn')?.addEventListener('click', open);
  });

  window.ImageModels = { open, refresh };
})();
