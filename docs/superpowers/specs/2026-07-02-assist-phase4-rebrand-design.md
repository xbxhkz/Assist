# Assist Phase 4 — "Assist" Rebrand (visible + env compatibility) (Design)

**Date:** 2026-07-02
**Status:** Approved for planning
**Depends on:** Phases 1–3 (all merged to `dev`).
**Parent project:** [[assist-native-windows-project]]. This is the final phase — the deep rebrand deferred since Phase 0 (exe/installer/window already say "Assist" as of Phase 2).

---

## Goal

Everything a user sees says **Assist**; `ASSIST_*` env vars are the documented names (with `ODYSSEUS_*` still honored); user data lives under `~/.assist` (auto-migrated from `~/.odysseus`). Internal Python/JS identifiers, comments, and log lines stay `odysseus` — out of scope by decision.

Success criteria:
- The web UI (titles, nav, page `<title>`, login, manifest) and rendered report/email templates read "Assist".
- README + user docs read "Assist"; the wordmark image is the provided `Assistlogo.png`; the app icon/favicon/PWA icons come from the provided `assistappicon.png`.
- Setting `ASSIST_X` works for every existing `ODYSSEUS_X` var; setting `ODYSSEUS_X` still works (backward compatible).
- On startup, an existing `~/.odysseus/data` is migrated to `~/.assist/data`; new installs use `~/.assist/data`.
- No behavioral regressions: the ~1,131 `getenv("ODYSSEUS_*")` call sites and ~335 tests are untouched (the shim makes rewrites unnecessary).

## Confirmed decisions

1. **Depth:** visible rebrand + env/data compatibility. Internal identifiers/comments/logs stay `odysseus`.
2. **Env vars:** a generic startup **mirror shim** (ASSIST_↔ODYSSEUS_) — `ASSIST_*` documented, `ODYSSEUS_*` still honored, **no `getenv` rewrites**. (Rejected: rewriting every call site — ~1,000 edits + huge test churn, no user benefit.)
3. **Data dir:** `~/.assist/data` with one-time auto-migration from `~/.odysseus/data`.
4. **Assets:** user provided `Assistlogo.png` (wordmark) and `assistappicon.png` (app icon, 1408×768 non-square → must be processed into square `.ico`/PWA/favicon).

## Scope of "user-facing" (what changes vs. what stays)

- **Change:** display strings in `static/index.html`, `static/login.html`, `static/manifest.json`, user-visible JS strings, rendered templates (`src/visual_report.py`, research report HTML), README + `docs/*.md` prose/headings, the wordmark + icon assets.
- **Keep (out of scope):** Python/JS variable/function/module names, code comments, `logger.*` messages, `ODYSSEUS_*` names *in code* (the shim bridges them), the GitHub repo URL (can't rename from here), and the demo screenshots `docs/odysseus-*.jpg` (they depict the old UI; refs may be renamed but the pixels stay).

## Components

### 1. Env-var compatibility shim — `src/brand_compat.py` (new)
A `mirror_brand_env(environ=os.environ)` function that, for every key, mirrors `ASSIST_*`↔`ODYSSEUS_*` using `setdefault` (so an explicitly-set target is never clobbered):

```python
def mirror_brand_env(environ=os.environ):
    for k, v in list(environ.items()):
        if k.startswith("ASSIST_"):
            environ.setdefault("ODYSSEUS_" + k[len("ASSIST_"):], v)
        elif k.startswith("ODYSSEUS_"):
            environ.setdefault("ASSIST_" + k[len("ODYSSEUS_"):], v)
```

Called **first thing** at process start — at the top of `launcher.py` (native) and imported/invoked at the top of `app.py` **before** `src.constants` (which reads env at import). Because every existing `getenv("ODYSSEUS_X")` still resolves, no other code changes. Fully unit-testable.

### 2. Data-dir migration — `src/runtime_paths.py` (modify)
- Default becomes `~/.assist/data` (frozen path); dev source path unchanged.
- A `_migrate_legacy_data_dir(new, legacy)` helper: if `new` doesn't exist and `legacy` (`~/.odysseus/data`) does, `os.rename(legacy, new)` (best-effort; on failure fall back to using `legacy`). Run once during `get_default_data_dir()`.
- `ASSIST_DATA_DIR` / `ODYSSEUS_DATA_DIR` both work via the shim (Component 1). Covered by `tests/test_runtime_paths.py` extensions.

### 3. Brand assets — `scripts/build_brand_assets.py` (new) + placement
- Process `assistappicon.png`: center-crop to square, resize, and write `static/icon.ico` (multi-size), `static/icons/icon-192.png`, `icon-512.png`, `icon-maskable-512.png`, and a favicon (Pillow — already a dependency). Committed outputs so the build/exe/PWA use them.
- Move `Assistlogo.png` → `docs/assist-wordmark.png`; update README to reference it.
- `Assist.spec` already uses `static/icon.ico`; regenerating that file in place makes the exe icon correct with no spec change.

### 4. User-facing strings — targeted display-text replacement
Replace "Odysseus" → "Assist" only where it renders to a user:
- `static/index.html`: `<title>`, the route→title map, nav/rail labels, modal headers, any visible "Odysseus" text.
- `static/manifest.json`: `name`, `short_name`.
- `static/login.html`: heading/branding.
- User-visible JS strings (e.g. toast/alert/label text) in `static/js/*.js` — display text only, not identifiers.
- Rendered templates: `src/visual_report.py`, research report HTML, and any email subject/body product name.
Each change is a string edit; identifiers/comments left alone.

### 5. Docs — `README.md` + `docs/*.md`
- Prose/headings "Odysseus" → "Assist"; wordmark image swap (Component 3).
- Leave the GitHub URL and the demo screenshots as-is (noted Minors).

## Data flow (env + data at startup)

1. Process starts → `mirror_brand_env()` runs → both `ASSIST_*` and `ODYSSEUS_*` resolve to the same values.
2. `src.constants` imports → reads `ODYSSEUS_*` as today (now also settable via `ASSIST_*`).
3. `get_default_data_dir()` resolves `~/.assist/data`, migrating `~/.odysseus/data` if present.

## Error handling
- **Shim:** pure dict mirroring; `setdefault` guarantees an explicitly-set `ODYSSEUS_X` *and* `ASSIST_X` (both set) keeps each as-is (no clobber). Never raises.
- **Data migration:** wrapped in try/except; on any failure (permissions, cross-volume), fall back to the legacy path so the app still starts and finds existing data.
- **Asset build:** the generator is a build-time script; committed outputs mean a missing Pillow at runtime is irrelevant.

## Testing
- **Unit:** `mirror_brand_env` (ASSIST→ODYSSEUS, ODYSSEUS→ASSIST, both-set no-clobber, non-brand keys untouched); `_migrate_legacy_data_dir` / `get_default_data_dir` (migrates when legacy present + new absent; no-op when new exists; frozen vs source path; env override wins via the shim).
- **Text-guard:** `static/index.html` `<title>` and `manifest.json` `name` say "Assist"; README references `assist-wordmark.png`; no user-facing "Odysseus" remains in the specific changed files (guard on those files, not the whole tree).
- **Regression:** fast-lane localmodels + runtime-path + env tests stay green (the shim must not disturb existing `getenv` behavior).
- **Manual (real gate):** launch the app — window/title/nav/login read "Assist", the new icon/favicon/wordmark render; set `ASSIST_PORT`/`ASSIST_DATA_DIR` and confirm they take effect; confirm an existing `~/.odysseus/data` migrates.

## Out of scope
- Renaming internal identifiers/modules/comments/log lines.
- The GitHub repo URL / `pewdiepie-archdaemon/odysseus` references.
- Re-screenshotting the demo images.
- Any code that reads `ODYSSEUS_*` directly stays (bridged by the shim).

## Risks & mitigations
- **A missed startup entrypoint leaves the shim un-run** (env not mirrored). Mitigation: invoke it at the top of BOTH `launcher.py` and `app.py` (module import time, before `constants`); a test imports `app` with only `ASSIST_*` set and asserts the corresponding `ODYSSEUS_*` resolves.
- **Data migration on a locked/cross-volume dir.** Mitigation: try/except → fall back to legacy path; never block startup.
- **Over-eager string replacement changing an identifier.** Mitigation: scope edits to display contexts per-file (Component 4 list), text-guard the key outputs, and rely on the existing test suite to catch behavioral breakage.
- **Non-square icon source.** Mitigation: center-crop to square before resizing in the asset script; visually verify in the manual gate.
