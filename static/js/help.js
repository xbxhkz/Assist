// Help modal: manual (static content in index.html), tutorials launcher, and
// About. Mirrors the plain-script style of localModels.js/imageModels.js.
(function () {
  function $(id) { return document.getElementById(id); }

  // Tours registered in slashCommands.js (category "Tours") plus /demo.
  const TUTORIALS = [
    { cmd: '/demo',          label: 'Full product tour',   desc: 'The complete guided walkthrough' },
    { cmd: '/tour-settings', label: 'Settings',            desc: 'Models, integrations, appearance' },
    { cmd: '/tour-cookbook', label: 'Cookbook',            desc: 'Hardware, downloads, serving' },
    { cmd: '/tour-theme',    label: 'Theme editor',        desc: 'Colors, presets, customization' },
    { cmd: '/tour-gallery',  label: 'Gallery',             desc: 'Photos, albums, image editor' },
    { cmd: '/tour-library',  label: 'Library & documents', desc: 'Document storage and the editor' },
    { cmd: '/tour-brain',    label: 'Brain',               desc: 'Memories, tidy, skills' },
    { cmd: '/tour-notes',    label: 'Notes',               desc: 'Quick notes and note chat' },
    { cmd: '/tour-task-1',   label: 'Tasks',               desc: 'Built-ins, runs, scheduling' },
    { cmd: '/tour-research', label: 'Deep Research',       desc: 'Long-form research runs' },
    { cmd: '/tour-compare',  label: 'Model comparison',    desc: 'Side-by-side model answers' },
  ];

  function open() {
    const m = $('help-modal');
    if (m) m.classList.remove('hidden');
    renderTutorials();
    fillAbout();
  }

  function close() {
    const m = $('help-modal');
    if (m) m.classList.add('hidden');
  }

  function switchTab(tabId) {
    document.querySelectorAll('#help-tabs .admin-tab').forEach((b) => {
      b.classList.toggle('active', b.dataset.helpTab === tabId);
    });
    document.querySelectorAll('.help-tab-panel').forEach((p) => {
      p.style.display = p.id === tabId ? '' : 'none';
    });
  }

  function startTutorial(cmd) {
    close();
    const input = $('message');
    if (!input) return;
    // Route through the normal composer path so the tour runs exactly as if
    // the user typed the slash command.
    input.value = cmd;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new KeyboardEvent('keydown', {
      key: 'Enter', code: 'Enter', bubbles: true, cancelable: true,
    }));
  }

  function renderTutorials() {
    const list = $('help-tutorials-list');
    if (!list || list.childElementCount) return;
    TUTORIALS.forEach((t) => {
      const row = document.createElement('div');
      row.className = 'list-item';
      const label = document.createElement('span');
      label.className = 'grow';
      label.innerHTML = `<b>${t.label}</b> <span style="opacity:0.6;font-size:12px;">— ${t.desc}</span>`;
      const btn = document.createElement('button');
      btn.textContent = 'Start';
      btn.onclick = () => startTutorial(t.cmd);
      row.appendChild(label);
      row.appendChild(btn);
      list.appendChild(row);
    });
  }

  async function fillAbout() {
    const el = $('help-about-version');
    if (!el || el.dataset.filled) return;
    try {
      const r = await fetch('/api/ready', { credentials: 'same-origin' });
      const data = await r.json();
      el.textContent = data.version || '—';
      el.dataset.filled = '1';
    } catch (e) { el.textContent = '—'; }
  }

  document.addEventListener('DOMContentLoaded', () => {
    $('tool-help-btn')?.addEventListener('click', open);
    $('close-help-modal')?.addEventListener('click', close);
    $('help-modal')?.addEventListener('click', (e) => {
      if (e.target === $('help-modal')) close();
    });
    document.querySelectorAll('#help-tabs .admin-tab').forEach((b) => {
      b.addEventListener('click', () => switchTab(b.dataset.helpTab));
    });
  });

  window.HelpModal = { open, close };
})();
