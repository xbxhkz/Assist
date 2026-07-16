# EXL2 / MLX via External Servers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make EXL2 and MLX models first-class *connectable* models via one-click endpoint presets (TabbyAPI / `mlx_lm.server`), a Help-manual setup section, and an actionable Cookbook label — reusing the existing model-endpoint system.

**Architecture:** Frontend-only. Two preset buttons in the "Add a local model server" menu mirror the existing "Add Ollama" preset (fill `adm-epLocalUrl` with a default URL); a Help ▸ Manual `<details>` section documents running TabbyAPI (EXL2) and `mlx_lm.server` (MLX); the Cookbook's `_detectBackend` relabels EXL2/MLX to "External endpoint". No backend changes, no bundled runtimes.

**Tech Stack:** Vanilla ES-module JS (`static/js/admin.js`, `static/js/cookbook.js`), static HTML (`static/index.html`).

## Global Constraints

- **No backend changes, no bundled runtimes** (no exllamav2/mlx/CUDA-torch). Reuse `ModelEndpoint` + the existing `POST /api/model-endpoints` + probe flow.
- Preset default URLs are exactly: **TabbyAPI `http://127.0.0.1:5000/v1`**, **MLX `http://127.0.0.1:8080/v1`**.
- CSP-safe UI: `addEventListener` + `textContent` (no `innerHTML`-with-data, no inline `on*=` handlers). The card lives in the admin panel and is admin-only by its existing placement.
- Honest platform notes in the guidance: **EXL2 needs an NVIDIA GPU + CUDA (TabbyAPI); MLX is macOS/Apple-Silicon only** (`mlx_lm` won't run on Windows/NVIDIA).
- These are DOM/static-content changes in large existing modules that don't import cleanly under `node`; per the repo's established pattern for such files (hardware-monitor/voice/webcam UI), they are **live-verified**, not unit-tested. Stage specific files (never `git add -A`). Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Endpoint presets (TabbyAPI + MLX)

**Files:**
- Modify: `static/index.html` (2 buttons in `adm-epLocalMoreMenu` after `adm-epOllamaBtn` ~line 2501; card sub-text ~line 2519)
- Modify: `static/js/admin.js` (wire the 2 buttons after the `ollamaBtn` block ~line 1637)
- Test: none automated (DOM wiring in a non-`node`-importable module) — live-verified below.

**Interfaces:**
- Consumes (existing): `el(id)`, `_endpointMsg('local')` (returns `adm-epLocalMsg`), the input `adm-epLocalUrl`, the type select `adm-epLocalType`.

- [ ] **Step 1: Add the two preset menu buttons**

In `static/index.html`, inside `adm-epLocalMoreMenu`, immediately after the `adm-epOllamaBtn` button (~line 2501), add:

```html
                  <button class="admin-btn-sm adm-more-item" id="adm-epTabbyBtn" title="Fill the default TabbyAPI (EXL2) endpoint" style="background:none;border:0;border-radius:5px;padding:7px 9px;display:flex;align-items:center;gap:8px;width:100%;text-align:left;font-size:12px;font-weight:normal;color:var(--fg);cursor:pointer;"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>Add TabbyAPI (EXL2)</button>
                  <button class="admin-btn-sm adm-more-item" id="adm-epMlxBtn" title="Fill the default MLX (mlx_lm) endpoint" style="background:none;border:0;border-radius:5px;padding:7px 9px;display:flex;align-items:center;gap:8px;width:100%;text-align:left;font-size:12px;font-weight:normal;color:var(--fg);cursor:pointer;"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>Add MLX (mlx_lm)</button>
```

- [ ] **Step 2: Update the card sub-text**

In `static/index.html`, change the sub-text line (~line 2519):

```html
            <div class="admin-toggle-sub" style="margin:0 0 10px 2px;">Add a local model server (Ollama, llama.cpp, vLLM).</div>
```

to:

```html
            <div class="admin-toggle-sub" style="margin:0 0 10px 2px;">Add a local model server (Ollama, llama.cpp, vLLM, TabbyAPI/EXL2, MLX).</div>
```

- [ ] **Step 3: Wire the preset buttons in admin.js**

In `static/js/admin.js`, immediately after the `ollamaBtn` block closes (after its `}` near line 1637, before the `// Discover local models button` comment), add:

```javascript
  // EXL2 (TabbyAPI) and MLX (mlx_lm) endpoint presets — fill the default local
  // URL for that server; the user starts the server then clicks Add. See the
  // Help > Manual "EXL2 & MLX models" section for setup.
  const EXTERNAL_PRESETS = [
    { id: 'adm-epTabbyBtn', url: 'http://127.0.0.1:5000/v1', label: 'TabbyAPI (EXL2)' },
    { id: 'adm-epMlxBtn', url: 'http://127.0.0.1:8080/v1', label: 'MLX (mlx_lm)' },
  ];
  EXTERNAL_PRESETS.forEach((preset) => {
    const btn = el(preset.id);
    if (!btn) return;
    btn.addEventListener('click', () => {
      const input = el('adm-epLocalUrl');
      if (input) { input.value = preset.url; input.focus(); }
      const lt = el('adm-epLocalType');
      if (lt) lt.value = 'llm';
      const msg = _endpointMsg('local');
      if (msg) {
        msg.textContent = preset.label + ' URL filled — start the server, then click Add. See Help ▸ Manual for setup.';
        msg.className = 'adm-ep-inline-msg';
      }
    });
  });
```

- [ ] **Step 4: Self-review (no automated test — live-verified in Task 4's package step)**

Re-read the diff: the two buttons use the exact `adm-more-item` class/style as `adm-epOllamaBtn`; the JS uses `textContent` (no `innerHTML`), fills the exact URLs (`:5000/v1`, `:8080/v1`), sets the type to `llm`, and reuses `_endpointMsg('local')`. Confirm `admin.js` is loaded as `type="module"` in `static/index.html` (it is — grep `admin.js`), so the surrounding module code is fine; these additions use only existing helpers (`el`, `_endpointMsg`). Fix anything off.

- [ ] **Step 5: Commit**

```bash
git add static/index.html static/js/admin.js
git commit -m "feat(models): TabbyAPI (EXL2) + MLX endpoint presets

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Help ▸ Manual setup section

**Files:**
- Modify: `static/index.html` (a new `<details>` in `help-tab-manual`, before the "Troubleshooting" section)
- Test: none automated (static content) — live-verified.

- [ ] **Step 1: Add the manual section**

In `static/index.html`, in the Help manual accordion, immediately before the `<details>` whose summary is `Troubleshooting`, add:

```html
          <details>
            <summary style="cursor:pointer;font-weight:600;padding:6px 0;">EXL2 &amp; MLX models (external servers)</summary>
            <p>Assist serves <b>GGUF</b> models itself (Local Models). <b>EXL2</b> and <b>MLX</b> use GPU-vendor-specific runtimes, so you run a small OpenAI-compatible server for them and connect it as an endpoint (Settings → the model picker's <b>Add model endpoints</b>). Once added, the server's models show up in the model picker like any other.</p>
            <p><b>EXL2 — NVIDIA GPU + CUDA (Windows/Linux):</b> install <b>TabbyAPI</b> (<code>github.com/theroyallab/tabbyAPI</code>), run its <code>start.bat</code> (Windows) or <code>start.sh</code>, and put an <code>.exl2</code> model in its <code>models/</code> folder. It serves at <code>http://127.0.0.1:5000/v1</code>. Then use the <b>Add TabbyAPI (EXL2)</b> preset in the endpoints "…" menu and click <b>Add</b>.</p>
            <p><b>MLX — macOS / Apple Silicon only:</b> <code>pip install mlx-lm</code>, then <code>mlx_lm.server --model &lt;hf-repo-or-path&gt; --port 8080</code>. It serves at <code>http://127.0.0.1:8080/v1</code>. Then use the <b>Add MLX (mlx_lm)</b> preset → <b>Add</b>. (MLX does not run on Windows/NVIDIA.)</p>
          </details>
```

- [ ] **Step 2: Self-review**

Confirm the `<details>` is well-formed (open/close balanced), sits inside `help-tab-manual` before Troubleshooting, uses only static text (no data-interpolation), and the URLs/platform notes match Task 1's presets (`:5000/v1` EXL2, `:8080/v1` MLX; EXL2=NVIDIA/CUDA, MLX=Apple-only).

- [ ] **Step 3: Commit**

```bash
git add static/index.html
git commit -m "docs(help): EXL2 & MLX external-server setup section

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Cookbook "External endpoint" label for EXL2 / MLX

**Files:**
- Modify: `static/js/cookbook.js` (`_detectBackend`, ~line 411)
- Test: none automated (`cookbook.js` doesn't import cleanly under `node`) — live-verified.

**Interfaces:**
- Modifies `_detectBackend(model) -> {backend, label}` — keeps `backend: 'unsupported'` (so no downstream serve behavior changes), only makes the label actionable and adds EXL2 coverage.

- [ ] **Step 1: Relabel MLX + add EXL2**

In `static/js/cookbook.js`, replace the MLX branch (~line 411-413):

```javascript
  if (/\bmlx\b|mlx-|_mlx/i.test(_nm) || q.startsWith('MLX')) {
    return { backend: 'unsupported', label: 'Unsupported' };
  }
```

with:

```javascript
  if (/\bmlx\b|mlx-|_mlx/i.test(_nm) || q.startsWith('MLX')) {
    // MLX (mlx_lm, Apple Silicon) isn't served locally — connect it as an
    // external OpenAI-compatible endpoint (see Help > Manual).
    return { backend: 'unsupported', label: 'External endpoint' };
  }
  if (q.startsWith('EXL2') || /\bexl2\b|_exl2|exl2-/i.test(_nm)) {
    // EXL2 (exllamav2/TabbyAPI, CUDA) isn't served locally — connect it as an
    // external OpenAI-compatible endpoint (see Help > Manual).
    return { backend: 'unsupported', label: 'External endpoint' };
  }
```

- [ ] **Step 2: Self-review**

Confirm: `backend` stays `'unsupported'` for both (no change to the serve-command switch or any code keying off the backend value); only the human-readable `label` changes to `'External endpoint'`; the EXL2 branch is added right after MLX (before the AWQ/GGUF fall-through) so EXL2 repos no longer resolve to a misleading local-serve backend. Grep `cookbook.js` for other reads of `label === 'Unsupported'` — if any exist, note them for the reviewer (the label text changed).

- [ ] **Step 3: Commit**

```bash
git add static/js/cookbook.js
git commit -m "feat(cookbook): label EXL2/MLX models 'External endpoint' (not 'Unsupported')

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Package + live-verify

**Files:**
- Modify: `installer/Output/Assist-Setup.exe` (rebuilt; force-added)

**Interfaces:**
- Consumes: Tasks 1-3 (all in bundled `static/`).

- [ ] **Step 1: Build the installer**

Run: `powershell -ExecutionPolicy Bypass -File ./build-installer.ps1 -Fast`
Expected: ends with `Installer built: ...\installer\Output\Assist-Setup.exe`.

- [ ] **Step 2: Confirm the changes are in the frozen bundle**

Run:
```
grep -c "adm-epTabbyBtn" dist/Assist/_internal/static/index.html
grep -c "EXL2 &amp; MLX models" dist/Assist/_internal/static/index.html
grep -c "External endpoint" dist/Assist/_internal/static/js/cookbook.js
grep -c "127.0.0.1:5000/v1" dist/Assist/_internal/static/js/admin.js
```
Expected: each ≥ 1.

- [ ] **Step 3: Live-verify in the running app (manual)**

Reinstall, then as admin: open the model picker → **Add model endpoints** → the "…" menu shows **Add TabbyAPI (EXL2)** and **Add MLX (mlx_lm)**; clicking each fills `http://127.0.0.1:5000/v1` / `http://127.0.0.1:8080/v1` and shows the inline hint. Open **Help ▸ Manual** → the **"EXL2 & MLX models (external servers)"** section renders with the setup steps. In the **Cookbook**, browse an EXL2 or MLX repo → it shows the **"External endpoint"** label (not "Unsupported"). (Optionally, if you run a real TabbyAPI server, add it and confirm its model appears in the picker — the endpoint system itself is pre-existing.)

- [ ] **Step 4: Commit the installer**

```bash
git add -f installer/Output/Assist-Setup.exe
git commit -m "build: Assist-Setup.exe with EXL2/MLX endpoint presets + guidance

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- Frontend-only; no Python/backend files change and no new dependencies.
- All three feature tasks are DOM/static-content changes in large modules that don't import under `node`, so they're live-verified (Task 4) rather than unit-tested — consistent with how the hardware-monitor/voice/webcam UI was handled.
- The two preset URLs must be exactly `http://127.0.0.1:5000/v1` (TabbyAPI) and `http://127.0.0.1:8080/v1` (MLX); the Help section's URLs/platform notes must match.
- Task 3 keeps `backend: 'unsupported'` — do not change the backend value or the serve-command switch; only the label text changes.
