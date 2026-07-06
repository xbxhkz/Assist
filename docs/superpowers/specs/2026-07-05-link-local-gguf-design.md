# Link a local .gguf ("Browse for a model") — Design

**Date:** 2026-07-05
**Status:** Approved

## Goal

In the Local Models "add model" area, let the user **browse the machine for a
`.gguf` file** and add it as a **linked** model — one that appears in the Local
Models list and can be served/stopped like a downloaded model, without copying
the file. Removing a linked model only unlinks it (the file on disk is never
touched).

## Background

- The desktop shell is pywebview/WebView2. A web `<input type=file>` on Chromium
  does **not** expose the real filesystem path, so a native file dialog is
  required to obtain a path to serve from.
- `routes/localmodels_routes.py::_validate_model_path` currently accepts only
  paths **inside** `MODELS_DIR`. Serving a linked file requires allowing paths
  that the user has explicitly registered.
- `LocalModelManager.list_models()` returns `list_gguf_models(MODELS_DIR)` only.
- Auto-serve persists the served path to `last_model.json`, so a served linked
  model is re-served on next launch for free.

## Design

### 1. Native file picker (desktop only)

`launcher.py` exposes a `js_api` object on the window with one method:

```python
class _JsApi:
    _window = None
    def pick_gguf(self):
        """Open a native file dialog; return the chosen .gguf absolute path or ''."""
        try:
            paths = self._window.create_file_dialog(
                webview.OPEN_DIALOG, allow_multiple=False,
                file_types=('GGUF model (*.gguf)', 'All files (*.*)'))
            return paths[0] if paths else ''
        except Exception:
            return ''
```

Wired via `window = webview.create_window("Assist", origin, js_api=api)` then
`api._window = window`. Callable from JS as `await window.pywebview.api.pick_gguf()`.
Outside the desktop app (`window.pywebview` undefined) the Browse button is
hidden — this is a desktop-only affordance.

### 2. Linked-model registry — `src/localmodels/external.py` (new)

Persists a JSON list of absolute `.gguf` paths at
`<DATA_DIR>/external_models.json` (`os.path.dirname(MODELS_DIR)`).

- `add_external_model(path) -> dict`: realpath-normalize; require `.gguf` +
  `os.path.isfile`; do not add if the path is inside `MODELS_DIR` (already
  listed) or already registered; append + save; return
  `{"name", "path", "size", "external": True}`.
- `remove_external_model(path) -> None`: drop the realpath from the registry
  (no file deletion).
- `list_external_models() -> list[dict]`: for each registered path that still
  exists, return `{"name": basename, "path", "size", "external": True}`; **prune**
  registered paths whose file is gone (rewrite the registry without them).
- `is_registered_external(path) -> bool`: realpath is in the registry.

All file I/O is best-effort (corrupt/missing registry → empty list).

### 3. Merge into the model list

`LocalModelManager.list_models()` returns downloaded models (each tagged
`"external": False`) followed by `list_external_models()`, de-duplicated by
realpath. `/api/localmodels/models` shape is unchanged except each entry now
carries an `external` boolean.

### 4. Serve validation

`_validate_model_path` accepts a path that is `.gguf`, exists, and is **either**
inside `MODELS_DIR` **or** `is_registered_external(path)`. Arbitrary unregistered
paths are still rejected (unchanged 400).

### 5. Routes (`routes/localmodels_routes.py`)

- `POST /api/localmodels/add-external` `{path}` → `add_external_model`; 400 on a
  non-`.gguf` / missing file. Admin-guarded (inherits router dependency).
- `POST /api/localmodels/remove-external` `{path}` → `remove_external_model`.
- `/serve` unchanged except it now passes validation for linked paths.

### 6. UI (`static/js/localModels.js`)

- A **"Browse for a .gguf…"** button in the add-model area (near the HF search),
  rendered only when `window.pywebview?.api?.pick_gguf` exists. Click:
  `const p = await window.pywebview.api.pick_gguf(); if (p) { await api('/api/localmodels/add-external', {POST, body:{path:p}}); await refresh(); refreshPicker(); }`.
- List rows for `m.external === true`: show a small **"Linked"** badge; the
  right-hand button reads **"Remove"** and calls `/remove-external {path: m.path}`
  (with a confirm) instead of the file-deleting `/delete`.

## Scope / non-goals

- **In scope:** native picker, linked registry, list merge, serve validation,
  add/remove routes, UI button + linked rows.
- **Out of scope:** copying/importing files into `MODELS_DIR`; a path-paste
  fallback for non-desktop browsers (button simply hidden there); editing linked
  entries; recursive folder scanning.

## Testing

1. **Registry (`tests/test_localmodels_external.py`):** add returns a tagged
   entry and persists; add rejects non-`.gguf` and missing file; add skips a
   path inside `MODELS_DIR` and de-dupes; `list_external_models` prunes a
   registered path whose file was deleted; `remove_external_model` unlinks
   without deleting the file; `is_registered_external` reflects state.
2. **Serve validation:** `_validate_model_path` accepts a registered external
   path and still rejects an arbitrary outside path (monkeypatched MODELS_DIR +
   registry).
3. **List merge:** `list_models()` includes both downloaded (`external: False`)
   and linked (`external: True`) entries.
4. **Manual (desktop):** the Browse button opens a native dialog, the picked
   file appears as a Linked row, serves, and Remove unlinks it without deleting.

## Files touched

- Create: `src/localmodels/external.py`, `tests/test_localmodels_external.py`
- Modify: `src/localmodels/manager.py` (list_models merge), `launcher.py` (js_api),
  `routes/localmodels_routes.py` (validation + 2 routes), `static/js/localModels.js`
  (button + linked rows).
