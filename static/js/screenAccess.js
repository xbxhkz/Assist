// Screen-access sidebar toggle: reflects and updates the admin-only
// `screen_access_enabled` setting that gates the desktop `capture_screen`
// tool. Mirrors the plain-script style of help.js. Defaults to off and is
// reset to off server-side on every restart (src/settings.py); this widget
// just displays and (when the user opts in) flips it back on.
(function () {
  function $(id) { return document.getElementById(id); }

  function reflect(enabled) {
    const toggle = $('screen-access-toggle');
    const indicator = $('screen-access-indicator');
    if (toggle) toggle.checked = !!enabled;
    if (indicator) indicator.style.display = enabled ? '' : 'none';
  }

  async function load() {
    try {
      const res = await fetch('/api/auth/settings', { credentials: 'same-origin' });
      if (!res.ok) return;
      const settings = await res.json();
      reflect(!!settings.screen_access_enabled);
    } catch (e) { console.warn('Failed to load screen access setting', e); }
  }

  async function save(enabled) {
    try {
      await fetch('/api/auth/settings', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ screen_access_enabled: enabled })
      });
    } catch (e) { console.warn('Failed to save screen access setting', e); }
  }

  document.addEventListener('DOMContentLoaded', () => {
    load();
    $('screen-access-toggle')?.addEventListener('change', (e) => {
      const enabled = !!e.target.checked;
      reflect(enabled);
      save(enabled);
    });
  });
})();
