// Input-control sidebar toggle: reflects and updates the admin-only
// `input_control_enabled` setting that gates the mouse/keyboard/UIA acting
// tools. Mirrors screenAccess.js. Defaults off and is reset off server-side
// on every restart (src/settings.py); this widget just displays and flips it.
(function () {
  function $(id) { return document.getElementById(id); }

  function reflect(enabled) {
    const toggle = $('input-control-toggle');
    const indicator = $('input-control-indicator');
    if (toggle) toggle.checked = !!enabled;
    if (indicator) indicator.style.display = enabled ? '' : 'none';
  }

  async function load() {
    try {
      const res = await fetch('/api/auth/settings', { credentials: 'same-origin' });
      if (!res.ok) return;
      const settings = await res.json();
      reflect(!!settings.input_control_enabled);
    } catch (e) { console.warn('Failed to load input control setting', e); }
  }

  async function save(enabled) {
    try {
      await fetch('/api/auth/settings', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_control_enabled: enabled })
      });
    } catch (e) { console.warn('Failed to save input control setting', e); }
  }

  document.addEventListener('DOMContentLoaded', () => {
    load();
    $('input-control-toggle')?.addEventListener('change', (e) => {
      const enabled = !!e.target.checked;
      reflect(enabled);
      save(enabled);
    });
  });
})();
