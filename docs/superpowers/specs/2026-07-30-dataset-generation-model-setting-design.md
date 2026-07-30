# Dataset Generation Model Setting — Design

**Goal:** Let an admin pin a specific model/endpoint for AI Studio's dataset generation (free, grounded,
and upload-grounded), instead of relying purely on serve-time auto-detection (Default Chat Model → the
currently-served local model). Surfaced as a new "Dataset Generation Model" card in Settings → AI, next to
Default Chat Model / Utility Model / Research Model.

## The pattern being followed

The app already has this exact shape three times over — Default Chat Model, Utility Model, Research Model —
each a `{prefix}_endpoint_id` / `{prefix}_model` setting pair, resolved via
`resolve_endpoint(prefix, ..., owner=owner)` (`src/endpoint_resolver.py:271`), with a Settings UI card of an
Endpoint `<select>` + Model `<select>`, empty = "Same as chat", auto-saving on change. **Research Model** is
the closer precedent (no fallback-chain widget — just the two selects + a status line), since dataset
generation, like Research, is a single admin-facing purpose, not the primary chat path.

`resolve_endpoint` already cascades an unset purpose: anything other than `"utility"`/`"default"` falls
through to Utility, then to Default, if unset (`src/endpoint_resolver.py:304-318`). So giving dataset
generation its own prefix, `"dataset_generation"`, means: **unset → today's behavior unchanged** (Default →
this session's served-local-model fallback); **set → pinned to that specific endpoint/model**, bypassing
auto-detection entirely.

## Architecture

**Backend:**
- `src/settings.py` — add `"dataset_generation_endpoint_id": ""` and `"dataset_generation_model": ""` to
  `DEFAULT_SETTINGS` (required for `POST /api/auth/settings` to persist them — it only writes keys already
  present in that dict). **Not** added to `_PER_USER_KEYS` — the whole Dataset feature is admin-only, so a
  per-user override has no audience.
- `routes/dataset_routes.py` — `_default_model_call`'s first resolve call changes from
  `resolve_endpoint("default", owner=owner)` to `resolve_endpoint("dataset_generation", owner=owner)`. The
  existing `_served_local_endpoint` fallback (added this session) stays as the last resort when neither the
  new setting nor Default/Utility resolve anything.

**Frontend (`static/js/settings.js` + `static/index.html`):**
- A new admin-card in `data-settings-panel="ai"`, placed after the existing Utility Model card (before
  Folder access), titled "Dataset Generation Model", with `#set-datasetGenEndpoint` / `#set-datasetGenModel`
  selects (both starting with a `"Same as chat"` option) and a `#set-datasetGenMsg` status line — structurally
  identical to the Research Model card, minus the extra numeric-timeout inputs Research has.
- `initDatasetGenerationModel()` in `settings.js`, mirroring `initResearchSettings()`'s shape: fetch
  endpoints, populate selects from `GET /api/auth/settings`, auto-save `{dataset_generation_endpoint_id,
  dataset_generation_model}` on change via `POST /api/auth/settings`, refresh on `_registerAiEndpointRefresh`.
  Wired into the existing settings-init sequence next to `initUtilityModel()`.

## Data flow

Admin picks Endpoint/Model in Settings → AI → saved to `dataset_generation_endpoint_id`/`_model` → next
Generate click → `_default_model_call` resolves `"dataset_generation"` first → if set, calls that endpoint
directly; if unset, cascades to Utility → Default → the served-local-model fallback exactly as today.

## Error handling

No new error paths: this only changes *which* endpoint gets resolved first, not how a resolved (or
unresolved) call is handled — the existing never-raises/never-500 generation pipeline, and this session's
clear-error-message fixes, apply unchanged regardless of which endpoint answers.

## Testing

- `routes/dataset_routes.py`: a test confirming `_default_model_call` calls `resolve_endpoint("dataset_generation", ...)`
  as its first resolve attempt (via a fake/spy `resolve_endpoint`), and that when it returns a URL, no
  `_served_local_endpoint` fallback is attempted.
- `src/settings.py`: confirm the two new keys exist in `DEFAULT_SETTINGS` with `""` defaults, and are absent
  from `_PER_USER_KEYS`.
- Frontend: `node --check` on `settings.js`; text-guard tests on `index.html` for the new card's element ids
  and on `settings.js` for the new init function + its wiring call.
- Manual GUI verification owed: set a specific model in the new card, confirm Generate uses it regardless of
  what's served; clear it, confirm behavior reverts to today's cascade.

## Non-goals

- No fallback-chain widget for this setting (matches Research Model's simpler precedent, not Utility/Default's).
- No per-user override (admin-only feature).
- No change to the resolution cascade itself, `_served_local_endpoint`, or any generation logic beyond the
  one resolve-prefix swap.
