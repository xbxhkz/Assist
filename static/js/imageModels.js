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
    const msg = $('imagemodels-msg');
    const serveBtn = $('imagemodels-serve-btn');
    if (serveBtn) serveBtn.disabled = true;
    if (msg) { msg.style.color = ''; msg.textContent = 'Starting… image models load slowly — this can take a few minutes.'; }
    try {
      await api('/api/imagemodels/serve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ diffusion_model: pickedPath, device: device() }),
      });
      if (msg) msg.textContent = '';
      await refresh();
      refreshPicker();
    } catch (e) {
      if (msg) { msg.style.color = 'var(--red)'; msg.textContent = e.message; }
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
