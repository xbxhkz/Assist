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

  function fmtBytes(n) {
    if (!n) return '';
    if (n >= 1e9) return (n / 1e9).toFixed(1) + ' GB';
    if (n >= 1e6) return (n / 1e6).toFixed(0) + ' MB';
    return (n / 1e3).toFixed(0) + ' KB';
  }

  let downloadedNames = new Set();

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
    data.files.forEach((f) => {
      const row = document.createElement('div');
      row.className = 'list-item';
      row.style.paddingLeft = '18px';
      const label = document.createElement('span');
      label.className = 'grow';
      label.textContent = `${f.filename} — ${fmtBytes(f.size)}`;
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
      afterRow.insertAdjacentElement('afterend', row);
    });
  }

  async function refresh() {
    const statusEl = $('localmodels-status');
    const listEl = $('localmodels-list');
    if (!listEl) return;
    let status = { running: false };
    try { status = await api('/api/localmodels/status'); } catch (e) {}
    if (statusEl) {
      statusEl.textContent = status.running
        ? `Running: ${status.model} (port ${status.port})`
        : 'No model running';
    }
    let data = { models: [] };
    try { data = await api('/api/localmodels/models'); } catch (e) {}
    downloadedNames = new Set((data.models || []).map((m) => m.name));
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
    const searchBtn = $('localmodels-search-btn');
    if (searchBtn) searchBtn.addEventListener('click', doSearch);
  });

  window.LocalModels = { open, close, refresh, doSearch, pollDownload };
})();
