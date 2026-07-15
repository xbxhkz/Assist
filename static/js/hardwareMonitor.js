// Live hardware monitor: polls /api/hwfit/usage while the sidebar panel is open,
// keeps a ~60-sample ring buffer per metric, draws canvas sparklines. CSP-safe.
(function () {
  const POLL_MS = 1000;
  const WINDOW = 60;                 // samples kept per series
  let timer = null;
  const series = {};                 // key -> [values]
  const canvases = {};               // key -> {canvas, label}

  function $(id) { return document.getElementById(id); }

  function push(key, val) {
    const s = series[key] || (series[key] = []);
    s.push(val);
    if (s.length > WINDOW) s.shift();
  }

  function ensureRow(key, title) {
    if (canvases[key]) return canvases[key];
    const body = $('hwmon-body');
    const row = document.createElement('div');
    row.style.cssText = 'margin-bottom:6px;';
    const label = document.createElement('div');
    label.style.cssText = 'font-size:11px;opacity:0.8;margin-bottom:2px;';
    label.textContent = title;
    const cv = document.createElement('canvas');
    cv.width = 180; cv.height = 26;
    cv.style.cssText = 'width:100%;height:26px;display:block;background:rgba(127,127,127,0.12);border-radius:3px;';
    row.appendChild(label); row.appendChild(cv);
    body.appendChild(row);
    return (canvases[key] = { canvas: cv, label: label });
  }

  function draw(key) {
    const ref = canvases[key]; if (!ref) return;
    const s = series[key] || [];
    const ctx = ref.canvas.getContext('2d');
    const w = ref.canvas.width, h = ref.canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (s.length < 2) return;
    ctx.strokeStyle = '#50fa7b';       // literal — canvas 2d can't resolve CSS var()
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < s.length; i++) {
      const x = (i / (WINDOW - 1)) * w;
      const y = h - Math.max(0, Math.min(100, s[i])) / 100 * (h - 2) - 1;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function render(u) {
    push('cpu', u.cpu_percent || 0);
    ensureRow('cpu', 'CPU ' + Math.round(u.cpu_percent || 0) + '%').label.textContent =
      'CPU ' + Math.round(u.cpu_percent || 0) + '%';
    draw('cpu');

    push('ram', u.ram_percent || 0);
    ensureRow('ram', '').label.textContent =
      'RAM ' + (u.ram_used_gb || 0) + '/' + (u.ram_total_gb || 0) + ' GB (' + Math.round(u.ram_percent || 0) + '%)';
    draw('ram');

    (u.gpus || []).forEach((g) => {
      const vk = 'gpu' + g.index + 'vram';
      push(vk, g.vram_percent || 0);
      ensureRow(vk, '').label.textContent =
        'GPU' + g.index + ' VRAM ' + g.vram_used_gb + '/' + g.vram_total_gb + ' GB (' + Math.round(g.vram_percent) + '%)';
      draw(vk);
      const uk = 'gpu' + g.index + 'util';
      push(uk, g.util_percent || 0);
      ensureRow(uk, '').label.textContent = 'GPU' + g.index + ' util ' + Math.round(g.util_percent) + '%';
      draw(uk);
    });

    if (!(u.gpus || []).length && !canvases.nogpu) {
      const body = $('hwmon-body');
      const n = document.createElement('div');
      n.id = 'hwmon-nogpu'; n.style.cssText = 'font-size:11px;opacity:0.6;';
      n.textContent = 'No NVIDIA GPU detected';
      body.appendChild(n);
      canvases.nogpu = true;
    }
  }

  async function poll() {
    try {
      const res = await fetch('/api/hwfit/usage', { credentials: 'same-origin' });
      if (res.ok) render(await res.json());
    } catch (e) { /* skip this tick */ }
  }

  function start() { if (!timer) { poll(); timer = setInterval(poll, POLL_MS); } }
  function stop() { if (timer) { clearInterval(timer); timer = null; } }

  document.addEventListener('DOMContentLoaded', () => {
    const d = $('hwmon');
    if (!d) return;
    d.addEventListener('toggle', () => (d.open ? start() : stop()));
    if (d.open) start();
  });
})();
