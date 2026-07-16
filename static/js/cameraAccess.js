// cameraAccess.js — the sidebar "Camera access" switch, backing the
// `camera_access_enabled` setting that gates the desktop `webcam_look` tool.
// Reset to off server-side on every restart (src/settings.py); this widget
// reflects and updates the persisted setting.
(function () {
  function $(id) { return document.getElementById(id); }

  function reflect(on) {
    const t = $('camera-access-toggle');
    if (t) t.checked = !!on;
    const dot = $('camera-access-indicator');
    if (dot) dot.style.display = on ? '' : 'none';
  }

  async function load() {
    try {
      const res = await fetch('/api/auth/settings', { credentials: 'same-origin' });
      const settings = await res.json();
      reflect(!!settings.camera_access_enabled);
    } catch (e) { /* leave default */ }
  }

  async function save(enabled) {
    try {
      await fetch('/api/auth/settings', {
        method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_access_enabled: enabled }),
      });
    } catch (e) { /* ignore */ }
    reflect(enabled);
  }

  document.addEventListener('DOMContentLoaded', () => {
    load();
    $('camera-access-toggle')?.addEventListener('change', (e) => {
      save(!!e.target.checked);
    });
  });
})();
