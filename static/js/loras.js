// LoRA manager UI (Image models card). Search Civitai, download from HF/URL/file,
// list/delete installed LoRAs, copy the <lora:name:weight> tag to paste into a prompt.
(function () {
  function $(id) { return document.getElementById(id); }
  function msg(t, err) { const m = $('lora-msg'); if (m) { m.textContent = t || ''; m.style.color = err ? 'var(--red,#ff5555)' : ''; } }

  async function api(path, opts) {
    const res = await fetch(path, Object.assign({ credentials: 'same-origin' }, opts || {}));
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error((data && data.detail) || String(res.status));
    return data;
  }

  function btn(label, fn) { const b = document.createElement('button'); b.type = 'button'; b.textContent = label; b.addEventListener('click', fn); return b; }

  async function refreshInstalled() {
    const host = $('lora-installed'); if (!host) return;
    let loras = [];
    try { loras = (await api('/api/loras')).loras || []; } catch (e) { return; }
    host.innerHTML = '';
    if (!loras.length) { const e = document.createElement('div'); e.style.cssText = 'font-size:12px;opacity:0.6;'; e.textContent = 'No LoRAs installed.'; host.appendChild(e); return; }
    loras.forEach((l) => {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;gap:6px;align-items:center;font-size:12px;padding:2px 0;';
      const nm = document.createElement('span'); nm.style.cssText = 'flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'; nm.textContent = l.name;
      const tag = '<lora:' + l.name + ':0.8>';
      row.appendChild(nm);
      row.appendChild(btn('Copy tag', () => { navigator.clipboard && navigator.clipboard.writeText(tag); msg('Copied ' + tag); }));
      row.appendChild(btn('Delete', async () => { try { await api('/api/loras/' + encodeURIComponent(l.name), { method: 'DELETE' }); msg('Deleted ' + l.name); refreshInstalled(); } catch (e) { msg('Delete failed: ' + e.message, true); } }));
      host.appendChild(row);
    });
  }

  async function download(body) {
    msg('Downloading…');
    try { const d = await api('/api/loras/download', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }); msg('Installed ' + d.lora.name); refreshInstalled(); }
    catch (e) { msg('Download failed: ' + e.message, true); }
  }

  async function civitaiSearch() {
    const q = ($('lora-civitai-q') || {}).value || '';
    const host = $('lora-search-results'); if (!host) return;
    msg('Searching Civitai…'); host.innerHTML = '';
    let results = [];
    try { results = (await api('/api/loras/civitai/search?q=' + encodeURIComponent(q))).results || []; msg(''); }
    catch (e) { msg('Search failed: ' + e.message, true); return; }
    results.forEach((r) => {
      const row = document.createElement('div'); row.style.cssText = 'display:flex;gap:6px;align-items:center;font-size:12px;padding:3px 0;border-bottom:1px solid var(--border,#333);';
      const info = document.createElement('div'); info.style.cssText = 'flex:1;min-width:0;';
      const nm = document.createElement('div'); nm.style.cssText = 'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'; nm.textContent = r.name + (r.base_model ? '  ·  ' + r.base_model : '');
      info.appendChild(nm);
      if (r.trigger_words && r.trigger_words.length) { const tw = document.createElement('div'); tw.style.cssText = 'opacity:0.6;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'; tw.textContent = 'triggers: ' + r.trigger_words.join(', '); info.appendChild(tw); }
      row.appendChild(info);
      const dl = btn('Download', () => download({ source: 'civitai', download_url: r.download_url, file_name: r.file_name }));
      if (!r.download_url) dl.disabled = true;
      row.appendChild(dl);
      host.appendChild(row);
    });
    if (!results.length) { const e = document.createElement('div'); e.style.cssText = 'font-size:12px;opacity:0.6;'; e.textContent = 'No results.'; host.appendChild(e); }
  }

  async function uploadFile() {
    const inp = $('lora-file'); if (!inp || !inp.files || !inp.files[0]) { msg('Choose a .safetensors file first', true); return; }
    const fd = new FormData(); fd.append('file', inp.files[0]);
    msg('Importing…');
    try { const d = await api('/api/loras/upload', { method: 'POST', body: fd }); msg('Imported ' + d.lora.name); refreshInstalled(); }
    catch (e) { msg('Import failed: ' + e.message, true); }
  }

  document.addEventListener('DOMContentLoaded', () => {
    $('lora-civitai-search-btn') && $('lora-civitai-search-btn').addEventListener('click', civitaiSearch);
    $('lora-hf-btn') && $('lora-hf-btn').addEventListener('click', () => download({ source: 'hf', repo: ($('lora-hf-repo') || {}).value || '', filename: ($('lora-hf-file') || {}).value || '' }));
    $('lora-url-btn') && $('lora-url-btn').addEventListener('click', () => download({ source: 'url', url: ($('lora-url') || {}).value || '', name: ($('lora-url-name') || {}).value || 'lora' }));
    $('lora-file-btn') && $('lora-file-btn').addEventListener('click', uploadFile);
    refreshInstalled();
  });
})();
