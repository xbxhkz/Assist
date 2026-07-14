// Shell-execution sidebar toggle: reflects/updates the global `shell_exec_enabled`
// setting that gates the powershell/cmd tools. Mirrors inputControl.js. Defaults
// off and is reset off server-side on every restart (src/settings.py).
(function () {
  function $(id) { return document.getElementById(id); }

  function reflect(enabled) {
    const toggle = $('shell-exec-toggle');
    const indicator = $('shell-exec-indicator');
    if (toggle) toggle.checked = !!enabled;
    if (indicator) indicator.style.display = enabled ? '' : 'none';
  }

  async function load() {
    try {
      const res = await fetch('/api/auth/settings', { credentials: 'same-origin' });
      if (!res.ok) return;
      const settings = await res.json();
      reflect(!!settings.shell_exec_enabled);
    } catch (e) { console.warn('Failed to load shell exec setting', e); }
  }

  async function save(enabled) {
    try {
      await fetch('/api/auth/settings', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ shell_exec_enabled: enabled })
      });
      if (!enabled) {
        // Turning consent off clears every session's auto-approve elevation,
        // so re-enabling later requires fresh per-command approval again.
        try {
          await fetch('/api/shell/reset', { method: 'POST', credentials: 'same-origin' });
        } catch (e) { /* best-effort; next backend restart also resets */ }
      }
    } catch (e) { console.warn('Failed to save shell exec setting', e); }
  }

  document.addEventListener('DOMContentLoaded', () => {
    load();
    $('shell-exec-toggle')?.addEventListener('change', (e) => {
      const enabled = !!e.target.checked;
      reflect(enabled);
      save(enabled);
    });
  });
})();
