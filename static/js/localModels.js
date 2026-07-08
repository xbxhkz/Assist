// Minimal Local Models UI (Phase 3a): list local GGUF files, serve/stop one at
// a time, and show live status. Mirrors the Cookbook modal conventions.
(function () {
  function $(id) { return document.getElementById(id); }

  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
  }

  function fmtBytes(n) {
    if (!n) return '';
    if (n >= 1e9) return (n / 1e9).toFixed(1) + ' GB';
    if (n >= 1e6) return (n / 1e6).toFixed(0) + ' MB';
    return (n / 1e3).toFixed(0) + ' KB';
  }

  let downloadedNames = new Set();

  // After serving/stopping/deleting a local model we change the ModelEndpoint
  // set, so force the main chat model picker to reload (bypassing its 30s
  // caches). Without this the served model doesn't appear in the model
  // selector until the picker's cache ages out — it looks like nothing ran.
  function refreshPicker() {
    try { window.modelsModule?.refreshModels?.(true); } catch (e) {}
  }

  async function pollDownload() {
    const prog = $('localmodels-progress');
    if (!prog) return;
    let st = { downloading: false };
    try { st = await api('/api/localmodels/download/status'); } catch (e) {}
    if (st.downloading) {
      const pct = st.pct != null ? st.pct + '%' : fmtBytes(st.bytes);
      prog.style.display = 'block';
      prog.innerHTML = `Downloading ${st.filename}: ${pct} ` +
        `<button id="localmodels-cancel-btn">Cancel</button>`;
      const cancel = $('localmodels-cancel-btn');
      if (cancel) cancel.onclick = () => api('/api/localmodels/download/cancel', { method: 'POST' });
      setTimeout(pollDownload, 800);
    } else {
      prog.style.display = 'none';
      if (st.error) alert('Download error: ' + st.error);
      refresh();  // a finished .gguf now shows in the serve list
    }
  }

  async function doSearch() {
    const q = ($('localmodels-search') || {}).value || '';
    const resultsEl = $('localmodels-results');
    if (!resultsEl) return;
    resultsEl.innerHTML = 'Searching…';
    let data = { results: [] };
    try { data = await api('/api/localmodels/catalog/search?q=' + encodeURIComponent(q)); }
    catch (e) { resultsEl.innerHTML = 'Search failed: ' + e.message; return; }
    resultsEl.innerHTML = '';
    data.results.forEach((r) => {
      const row = document.createElement('div');
      row.className = 'list-item';
      const label = document.createElement('span');
      label.className = 'grow';
      label.textContent = `${r.repo}  (${r.downloads.toLocaleString()} downloads)`;
      const btn = document.createElement('button');
      btn.textContent = 'Files';
      btn.onclick = () => listFiles(r.repo, row);
      row.appendChild(label);
      row.appendChild(btn);
      resultsEl.appendChild(row);
    });
    if (!data.results.length) resultsEl.textContent = 'No GGUF models found.';
  }

  async function listFiles(repo, afterRow) {
    let data = { files: [] };
    try { data = await api('/api/localmodels/catalog/files?repo=' + encodeURIComponent(repo)); }
    catch (e) { alert('Could not list files: ' + e.message); return; }
    data.files.sort((a, b) =>
      (_verdictRank[(a.fit || {}).verdict] ?? 3) - (_verdictRank[(b.fit || {}).verdict] ?? 3)
      || (a.size - b.size));
    let anchor = afterRow;
    data.files.forEach((f) => {
      const row = document.createElement('div');
      row.className = 'list-item';
      row.style.paddingLeft = '18px';
      const label = document.createElement('span');
      label.className = 'grow';
      label.textContent = `${f.filename} — ${fmtBytes(f.size)}`;
      const _badge = fitBadge(f.fit);
      if (_badge) label.insertAdjacentHTML('beforeend', _badge);
      const btn = document.createElement('button');
      const have = downloadedNames.has(f.filename);
      btn.textContent = have ? 'Downloaded' : 'Download';
      btn.disabled = have;
      btn.onclick = async () => {
        btn.disabled = true;
        try {
          await api('/api/localmodels/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: f.url, filename: f.filename }),
          });
          pollDownload();
        } catch (e) { alert('Download error: ' + e.message); btn.disabled = false; }
      };
      row.appendChild(label);
      row.appendChild(btn);
      anchor.insertAdjacentElement('afterend', row);
      anchor = row;
    });
  }

  function serveDevice() {
    const el = document.querySelector('input[name="localmodels-device"]:checked');
    return el ? el.value : 'cpu';
  }

  function runningLabel(status) {
    return `Running: ${status.model} on ${status.device === 'gpu' ? 'GPU' : 'CPU'} (port ${status.port})`;
  }

  async function refresh() {
    const statusEl = $('localmodels-status');
    const listEl = $('localmodels-list');
    if (!listEl) return;
    let status = { running: false };
    try { status = await api('/api/localmodels/status'); } catch (e) {}
    if (statusEl) {
      statusEl.textContent = status.running ? runningLabel(status) : 'No model running';
    }
    let data = { models: [] };
    try { data = await api('/api/localmodels/models'); } catch (e) {}
    downloadedNames = new Set((data.models || []).map((m) => m.name));
    if (statusEl && data.disk_bytes != null) {
      const base = status.running ? runningLabel(status) : 'No model running';
      statusEl.textContent = `${base}  ·  ${data.models.length} models · ${fmtBytes(data.disk_bytes)}`;
    }
    listEl.innerHTML = '';
    data.models.forEach((m) => {
      const row = document.createElement('div');
      row.className = 'list-item';
      const label = document.createElement('span');
      label.className = 'grow';
      label.textContent = `${m.name} — ${fmtBytes(m.size)}`;
      if (m.external) {
        label.insertAdjacentHTML('beforeend',
          ' <span style="font-size:10px;color:var(--accent,#45c4b0);border:1px solid var(--border);border-radius:6px;padding:1px 5px;margin-left:6px;" title="A file linked from elsewhere on your computer">Linked</span>');
      }
      const btn = document.createElement('button');
      const isRunning = status.running && status.model === m.name;
      btn.textContent = isRunning ? 'Stop' : 'Serve';
      btn.onclick = async () => {
        btn.disabled = true;
        const statusEl = $('localmodels-status');
        try {
          if (isRunning) {
            await api('/api/localmodels/stop', { method: 'POST' });
          } else {
            // Loading can take minutes for a large model — show progress so it
            // doesn't look frozen while llama-server mmaps the weights.
            btn.textContent = 'Starting…';
            if (statusEl) statusEl.textContent = `Starting ${m.name}… (large models can take a few minutes)`;
            await api('/api/localmodels/serve', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ model_path: m.path, device: serveDevice() }),
            });
          }
        } catch (e) { alert('Local model error: ' + e.message); }
        await refresh();
        refreshPicker();  // reflect the started/stopped model in the chat picker
      };
      row.appendChild(label);
      row.appendChild(btn);
      const del = document.createElement('button');
      del.style.marginLeft = '4px';
      if (m.external) {
        // Linked model: unlink from the list only — never delete the file.
        del.textContent = 'Remove';
        del.onclick = async () => {
          if (!confirm('Remove linked model ' + m.name + '? The file stays on your disk.')) return;
          try { await api('/api/localmodels/remove-external', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: m.path }) }); }
          catch (e) { alert('Remove error: ' + e.message); }
          await refresh();
          refreshPicker();
        };
      } else {
        del.textContent = 'Delete';
        del.onclick = async () => {
          if (!confirm('Delete ' + m.name + '?')) return;
          try { await api('/api/localmodels/delete', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: m.name }) }); }
          catch (e) { alert('Delete error: ' + e.message); }
          await refresh();
          refreshPicker();  // a deleted model may have torn down its endpoint
        };
      }
      row.appendChild(del);
      listEl.appendChild(row);
    });
    if (!data.models.length) {
      listEl.innerHTML = '<div class="list-item"><span class="grow">No .gguf models found. Add one to the models folder.</span></div>';
    }
  }

  // "Browse for a .gguf…" — links a model that lives anywhere on disk. The
  // native file dialog only exists in the desktop (pywebview) app, so the
  // button stays hidden in a plain browser.
  function setupBrowse() {
    const btn = $('localmodels-browse-btn');
    if (!btn) return;
    if (!(window.pywebview && window.pywebview.api && window.pywebview.api.pick_gguf)) return;
    btn.style.display = '';
    btn.onclick = async () => {
      let path = '';
      try { path = await window.pywebview.api.pick_gguf(); } catch (e) {}
      if (!path) return;  // cancelled
      try {
        await api('/api/localmodels/add-external', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path }),
        });
        await refresh();
        refreshPicker();
      } catch (e) { alert('Could not add model: ' + e.message); }
    };
  }

  function open() {
    const modal = $('localmodels-modal');
    if (modal) { modal.classList.remove('hidden'); refresh(); loadHardware(); loadRecommendations(); setupBrowse(); }
  }
  function close() {
    const modal = $('localmodels-modal');
    if (modal) modal.classList.add('hidden');
  }

  let hardware = null;

  async function loadHardware() {
    const el = $('localmodels-hardware');
    if (!el) return;
    try { hardware = await api('/api/localmodels/hardware'); } catch (e) { hardware = null; }
    if (!hardware) { el.textContent = 'Hardware: unknown'; return; }
    const gpu = hardware.has_gpu
      ? `${hardware.gpu_name || 'GPU'} (${hardware.vram_gb} GB VRAM)` : 'no GPU';
    el.textContent = `Your machine: ${hardware.ram_gb} GB RAM · ${gpu}`;
  }

  function fitBadge(fit) {
    if (!fit) return '';
    const map = { gpu: ['Fits on GPU', '#3fb950'], ram: ['Fits in RAM', '#d29922'],
                  too_big: ['Too big', '#f85149'] };
    const [label, color] = map[fit.verdict] || ['', '#888'];
    return `<span style="color:${color};font-size:11px;margin-left:6px;">${label}</span>`;
  }

  const _verdictRank = { gpu: 0, ram: 1, too_big: 2 };

  async function loadRecommendations() {
    const el = $('localmodels-recommendations');
    if (!el) return;
    let data = { recommendations: [] };
    try { data = await api('/api/localmodels/recommendations'); } catch (e) {}
    if (!data.recommendations || !data.recommendations.length) { el.innerHTML = ''; return; }
    el.innerHTML = '<div style="font-size:11px;opacity:0.6;margin-bottom:4px;">Recommended for your machine</div>';
    data.recommendations.forEach((r) => {
      const chip = document.createElement('button');
      chip.textContent = r.name;
      chip.style.cssText = 'margin:2px 4px 2px 0;font-size:11px;';
      chip.onclick = () => {
        const inp = $('localmodels-search');
        if (inp) { inp.value = r.name; }
        doSearch();
      };
      el.appendChild(chip);
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    const openBtn = $('tool-localmodels-btn');
    if (openBtn) openBtn.addEventListener('click', open);
    const closeBtn = $('close-localmodels-modal');
    if (closeBtn) closeBtn.addEventListener('click', close);
    const searchBtn = $('localmodels-search-btn');
    if (searchBtn) searchBtn.addEventListener('click', doSearch);
    const searchInput = $('localmodels-search');
    if (searchInput) searchInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') doSearch(); });
  });

  window.LocalModels = { open, close, refresh, doSearch, pollDownload, loadHardware, loadRecommendations };
})();
