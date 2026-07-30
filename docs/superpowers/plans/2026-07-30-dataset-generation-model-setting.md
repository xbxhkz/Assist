# Dataset Generation Model Setting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an admin pin a specific model/endpoint for AI Studio dataset generation via a new Settings → AI card, instead of relying only on auto-detection.

**Architecture:** A new settings-key pair (`dataset_generation_endpoint_id`/`_model`) added to the existing admin settings schema; `_default_model_call` resolves that prefix first (falling through to Default/Utility/served-local exactly as today when unset); a new Settings card mirrors the existing Research Model card.

**Tech Stack:** Python 3.14 (FastAPI, existing settings module), ES5 browser JS (`settings.js`), pytest (`--import-mode=importlib`), `node --check`.

## Global Constraints

- `POST /api/auth/settings` only persists keys already present in `DEFAULT_SETTINGS` (`src/settings.py`) — both new keys MUST be added there with `""` defaults, or they silently won't save.
- Do NOT add the new keys to `_PER_USER_KEYS` (`src/settings.py`) — this is an admin-only feature, no per-user override needed.
- `_default_model_call`'s resolve-prefix change is the ONLY change to generation logic — do not touch `_served_local_endpoint`, `generate_rows`, or `generate_grounded`.
- The new Settings card mirrors the **Research Model** card exactly (two selects, no fallback-chain widget, "Same as chat" placeholder, auto-save on change) — not the heavier Utility/Default cards.
- pytest `--import-mode=importlib`; `node --check` gates `settings.js`; commit to `dev`; commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; stage only the named files (NEVER `git add -A`; never stage `installer/Output/Assist-Setup.exe`).

## File Structure

- **Modify `src/settings.py`** — add the two new keys to `DEFAULT_SETTINGS`. (Task 1)
- **Modify `routes/dataset_routes.py`** — `_default_model_call`'s first `resolve_endpoint` call uses the new prefix. (Task 1)
- **Modify `static/index.html`** — new "Dataset Generation Model" admin-card. (Task 2)
- **Modify `static/js/settings.js`** — new `initDatasetGenerationModel()` + its call-site wiring. (Task 2)
- **Tests:** `tests/test_dataset_generate_routes.py` (extend, Task 1), `tests/test_settings_dataset_generation.py` (new, Task 1), `tests/test_dataset_ui.py` (extend, Task 2).

---

### Task 1: Backend — settings key + resolve-prefix change

**Files:**
- Modify: `src/settings.py`
- Modify: `routes/dataset_routes.py`
- Test: `tests/test_settings_dataset_generation.py`, `tests/test_dataset_generate_routes.py` (extend)

**Interfaces:**
- Consumes: `resolve_endpoint(prefix, ..., owner=owner)` (`src/endpoint_resolver.py:271`) — unchanged signature.
- Produces: `DEFAULT_SETTINGS["dataset_generation_endpoint_id"] == ""`, `DEFAULT_SETTINGS["dataset_generation_model"] == ""`; `_default_model_call` resolves `"dataset_generation"` before falling back to `_served_local_endpoint`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_settings_dataset_generation.py`:

```python
from src.settings import DEFAULT_SETTINGS, _PER_USER_KEYS


def test_dataset_generation_keys_registered_with_empty_default():
    assert DEFAULT_SETTINGS.get("dataset_generation_endpoint_id") == ""
    assert DEFAULT_SETTINGS.get("dataset_generation_model") == ""


def test_dataset_generation_keys_not_per_user():
    # admin-only feature -- no per-user override needed
    assert "dataset_generation_endpoint_id" not in _PER_USER_KEYS
    assert "dataset_generation_model" not in _PER_USER_KEYS
```

Append to `tests/test_dataset_generate_routes.py`:

```python
def test_default_model_call_resolves_dataset_generation_prefix_first(monkeypatch):
    import src.endpoint_resolver as er
    calls = []

    def fake_resolve(prefix, *a, **k):
        calls.append(prefix)
        if prefix == "dataset_generation":
            return ("http://ep/v1/chat/completions", "pinned-model", {})
        return (None, None, None)

    monkeypatch.setattr(er, "resolve_endpoint", fake_resolve)
    # served-local fallback must NOT be consulted when the new prefix resolves
    monkeypatch.setattr(dr, "_served_local_endpoint", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("should not fall back to served-local when dataset_generation resolves")))

    class _FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "hi"}}]}

    class _FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k): return _FakeResp()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", lambda **k: _FakeClient())

    result = asyncio.run(dr._default_model_call("hi"))
    assert calls[0] == "dataset_generation"
    assert result == "hi"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_settings_dataset_generation.py tests/test_dataset_generate_routes.py -k "dataset_generation" --import-mode=importlib -q`
Expected: FAIL (keys not in `DEFAULT_SETTINGS`; `_default_model_call` still resolves `"default"` first).

- [ ] **Step 3: Implement**

In `src/settings.py`, add these two lines inside the `DEFAULT_SETTINGS = { ... }` dict, near the other `research_*`/`utility_*` entries (exact placement inside the dict doesn't matter — dict order isn't semantically load-bearing here):

```python
    "dataset_generation_endpoint_id": "",
    "dataset_generation_model": "",
```

Do **not** add these keys to `_PER_USER_KEYS`.

In `routes/dataset_routes.py`, change the first line of `_default_model_call`'s body from:

```python
    url, model, headers = resolve_endpoint("default", owner=owner)
```

to:

```python
    url, model, headers = resolve_endpoint("dataset_generation", owner=owner)
```

(Everything else in `_default_model_call` — the `_served_local_endpoint` fallback, the httpx call, error handling — is unchanged. `resolve_endpoint`'s own cascade (`src/endpoint_resolver.py:304-318`) already falls `"dataset_generation"` through to Utility then Default when unset, so this preserves today's behavior when the new setting is empty.)

- [ ] **Step 4: Run the tests + import smoke**

Run: `python -m pytest tests/test_settings_dataset_generation.py tests/test_dataset_generate_routes.py --import-mode=importlib -q`
Expected: PASS (2 new + all existing `test_dataset_generate_routes.py` tests, e.g. 8+ passed).
Run: `python -c "import app; print('OK')"`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/settings.py routes/dataset_routes.py tests/test_settings_dataset_generation.py tests/test_dataset_generate_routes.py
git commit -m "feat(dataset-tools): dataset_generation settings prefix (pin a model for generation)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Frontend — Settings card

**Files:**
- Modify: `static/index.html`
- Modify: `static/js/settings.js`
- Test: `tests/test_dataset_ui.py` (extend)

**Interfaces:**
- Consumes: `GET`/`POST /api/auth/settings` (existing); `_fetchModelEndpoints`, `_fillEndpointSelect`, `_fillModelSelect`, `_registerAiEndpointRefresh` (existing helpers in `settings.js`, already used by `initResearchSettings`/`initUtilityModel`).
- Produces: `#set-datasetGenEndpoint`, `#set-datasetGenModel`, `#set-datasetGenMsg` in `index.html`; `initDatasetGenerationModel()` in `settings.js`, called from the existing settings-init sequence.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dataset_ui.py`:

```python
def test_index_has_dataset_generation_model_card():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in ('id="set-datasetGenEndpoint"', 'id="set-datasetGenModel"', 'id="set-datasetGenMsg"'):
        assert el in html, f"{el} missing from index.html"


def test_settings_js_wires_dataset_generation_model():
    src = (ROOT / "static" / "js" / "settings.js").read_text(encoding="utf-8")
    for s in ('initDatasetGenerationModel', 'dataset_generation_endpoint_id', 'dataset_generation_model',
              'set-datasetGenEndpoint', 'set-datasetGenModel'):
        assert s in src, f"{s} missing from settings.js"
    assert "initDatasetGenerationModel();" in src  # actually called, not just defined
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_ui.py -k "dataset_generation_model" --import-mode=importlib -q`
Expected: FAIL (card + function missing).

- [ ] **Step 3: Add the Settings card to `static/index.html`**

Insert this `admin-card` inside `data-settings-panel="ai"`, immediately AFTER the Utility Model card (the one whose closing `</div>` follows the `#set-utilityFallbacks` fallback-chain block) and BEFORE the "Folder access" card:

```html
          <div class="admin-card">
            <h2><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:5px;opacity:0.6"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg>Dataset Generation Model</h2>
            <div class="admin-toggle-sub" style="margin-bottom:8px">Model used by AI Studio's "Generate (AI)" (Dataset Builder). Leave blank to use whatever model is currently served, or the chat model.</div>
            <div class="settings-col">
              <div class="settings-row">
                <label class="settings-label">Endpoint</label>
                <select id="set-datasetGenEndpoint" class="settings-select">
                  <option value="">Same as chat</option>
                </select>
              </div>
              <div class="settings-row">
                <label class="settings-label">Model</label>
                <span class="adm-model-logo" id="set-datasetGenModel-logo" style="display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;flex-shrink:0;opacity:0.9;color:var(--fg);"></span>
                <select id="set-datasetGenModel" class="settings-select">
                  <option value="">Same as chat</option>
                </select>
              </div>
              <div id="set-datasetGenMsg" style="font-size:11px;color:color-mix(in srgb, var(--fg) 45%, transparent);margin-top:2px;"></div>
            </div>
          </div>
```

- [ ] **Step 4: Add `initDatasetGenerationModel()` to `static/js/settings.js`**

Add this function immediately after `initUtilityModel()`'s closing `}` (before the `/* ── Teacher Model ── */` comment):

```javascript
/* ── Dataset Generation Model ── */
async function initDatasetGenerationModel() {
  var epSel = el('set-datasetGenEndpoint');
  var modelSel = el('set-datasetGenModel');
  var msg = el('set-datasetGenMsg');
  var endpoints = [];
  if (!epSel || !modelSel) return;

  try {
    endpoints = await _fetchModelEndpoints();
    _fillEndpointSelect(epSel, endpoints, epSel.value, true);
  } catch (e) { console.warn('Failed to load endpoints for dataset generation', e); }

  function refreshModels(selectedModel) {
    var epId = epSel.value;
    var ep = endpoints.find(function(e) { return e.id === epId; });
    _fillModelSelect(modelSel, ep ? ep.models : [], selectedModel, true);
  }

  try {
    var res = await fetch('/api/auth/settings', { credentials: 'same-origin' });
    var settings = await res.json();
    if (settings.dataset_generation_endpoint_id) epSel.value = settings.dataset_generation_endpoint_id;
    refreshModels(settings.dataset_generation_model || '');
  } catch (e) { console.warn('Failed to load dataset generation settings', e); }

  function showStatus() {
    if (epSel.value) {
      var epName = epSel.options[epSel.selectedIndex].textContent;
      var mName = modelSel.value ? modelSel.value.split('/').pop() : 'auto';
      msg.textContent = epName + ' / ' + mName;
    } else {
      msg.textContent = 'Using chat defaults';
    }
    msg.style.color = 'var(--fg)';
  }
  showStatus();

  async function saveDatasetGen() {
    try {
      await fetch('/api/auth/settings', { method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_generation_endpoint_id: epSel.value || '',
          dataset_generation_model: modelSel.value || ''
        })
      });
      msg.textContent = 'Saved'; msg.style.color = 'var(--fg)';
      setTimeout(showStatus, 1500);
    } catch (e) { msg.textContent = 'Failed to save'; msg.style.color = 'var(--red)'; }
  }

  epSel.addEventListener('change', function() { refreshModels(''); saveDatasetGen(); });
  modelSel.addEventListener('change', saveDatasetGen);

  _registerAiEndpointRefresh(function(nextEndpoints) {
    endpoints = nextEndpoints;
    _fillEndpointSelect(epSel, endpoints, epSel.value, true);
    refreshModels(modelSel.value);
  });
}
```

Wire it into the existing init sequence — add `initDatasetGenerationModel();` immediately after the `initUtilityModel();` call:

```javascript
  initUtilityModel();
  initDatasetGenerationModel();
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_dataset_ui.py --import-mode=importlib -q`
Expected: PASS (all existing + 2 new; the file's existing `node --check`-style syntax gate, if any, must also pass — if `settings.js` has no dedicated syntax test yet, run `node --check static/js/settings.js` manually and confirm exit 0).

- [ ] **Step 6: Commit**

```bash
git add static/index.html static/js/settings.js tests/test_dataset_ui.py
git commit -m "feat(dataset-tools): Dataset Generation Model settings card

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **The core insight:** `resolve_endpoint`'s existing cascade logic does all the fallback work for free — Task 1 is a two-line settings-schema addition plus a one-line prefix swap, not new resolution logic.
- **Never touch** `_served_local_endpoint`, `generate_rows`, or `generate_grounded` — this feature only changes which endpoint gets tried FIRST.
- **Manual GUI verification owed** by the user: set a specific model in the new card, confirm Generate uses it; clear it, confirm the cascade reverts to today's behavior.
- **No frozen rebuild needed to test** (frontend + Python + settings schema only) — reaches the installed app on the next installer rebuild.
