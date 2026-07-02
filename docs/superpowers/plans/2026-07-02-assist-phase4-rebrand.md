# Assist Phase 4 — "Assist" Rebrand — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Everything a user sees says "Assist"; `ASSIST_*` env vars are documented (with `ODYSSEUS_*` still honored); data lives under `~/.assist` (auto-migrated from `~/.odysseus`).

**Architecture:** A generic startup env-mirror shim bridges `ASSIST_*`↔`ODYSSEUS_*` (so no `getenv` call sites change), `runtime_paths` migrates the frozen data dir, a Pillow script regenerates brand assets from the provided source PNGs, and targeted display-string/doc edits complete the visible rebrand. Internal identifiers, comments, logs, and `localStorage` keys stay `odysseus`.

**Tech Stack:** Python 3.14, FastAPI, Pillow, vanilla JS/HTML, pytest.

## Global Constraints

- **Keep internal `odysseus`:** do NOT rename Python/JS identifiers, function/module names, code comments, `logger.*` messages, `localStorage` keys (`odysseus-theme`, `odysseus-ui-scale`, `odysseus-last-user`), the `X-Odysseus-Owner` HTTP header, or the GitHub repo URL. Change only user-facing display text, env-var *documented names*, the data dir, and brand assets.
- **Env shim, not rewrites:** bridge `ASSIST_*`↔`ODYSSEUS_*`; the ~1,131 `getenv("ODYSSEUS_*")` call sites and ~335 tests stay untouched.
- **Data safety:** frozen data dir becomes `~/.assist/data`, migrating `~/.odysseus/data` once; on failure fall back to the legacy path (never lose data, never block startup).
- **Assets from provided sources:** `assistappicon.png` (root) → `static/icon.ico` + PWA icons + favicon (center-crop square first); `Assistlogo.png` (root) → `docs/assist-wordmark.png`.
- **No internal module / `ODYSSEUS_*`-in-code renames.**
- **Test env:** pytest with `--import-mode=importlib`; new tests carry no `slow` marker.

## File Structure

- `src/brand_compat.py` (new) — env mirror shim. Task 1.
- `app.py` + `launcher.py` (modify) — invoke the shim first. Task 1.
- `src/runtime_paths.py` (modify) — data-dir migration. Task 2.
- `scripts/build_brand_assets.py` (new) + generated `static/icon.ico`, `static/icons/*.png`, `static/favicon.png`, `docs/assist-wordmark.png`. Task 3.
- `static/index.html`, `static/manifest.json`, `static/login.html`, `src/visual_report.py` (modify) — display strings. Task 4.
- `README.md` + `docs/*.md` (modify) — prose + wordmark. Task 5.
- Tests: `tests/test_brand_compat.py`, `tests/test_runtime_paths.py` (extend), `tests/test_brand_assets.py`, `tests/test_brand_strings.py`.

---

### Task 1: Env-var compatibility shim

**Files:**
- Create: `src/brand_compat.py`
- Modify: `app.py` (first import), `launcher.py` (first import)
- Test: `tests/test_brand_compat.py`

**Interfaces:**
- Produces: `mirror_brand_env(environ=None) -> None` (mirrors `ASSIST_*`↔`ODYSSEUS_*` via `setdefault`; runs once on module import against `os.environ`).

- [ ] **Step 1: Write failing tests**

Create `tests/test_brand_compat.py`:

```python
"""Env-var compatibility shim: ASSIST_* <-> ODYSSEUS_* mirroring."""
import os

import src.brand_compat as bc


def test_mirror_assist_to_odysseus():
    env = {"ASSIST_PORT": "9000"}
    bc.mirror_brand_env(env)
    assert env["ODYSSEUS_PORT"] == "9000"


def test_mirror_odysseus_to_assist():
    env = {"ODYSSEUS_DATA_DIR": "/x"}
    bc.mirror_brand_env(env)
    assert env["ASSIST_DATA_DIR"] == "/x"


def test_mirror_no_clobber_when_both_set():
    env = {"ASSIST_PORT": "1", "ODYSSEUS_PORT": "2"}
    bc.mirror_brand_env(env)
    assert env["ASSIST_PORT"] == "1"
    assert env["ODYSSEUS_PORT"] == "2"


def test_mirror_ignores_non_brand_keys():
    env = {"PATH": "/usr/bin", "HF_TOKEN": "x"}
    bc.mirror_brand_env(env)
    assert set(env) == {"PATH", "HF_TOKEN"}


def test_mirror_bridges_real_os_environ(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_SCRIPT_HOST", raising=False)
    monkeypatch.setenv("ASSIST_SCRIPT_HOST", "myhost")
    bc.mirror_brand_env()  # defaults to os.environ
    assert os.getenv("ODYSSEUS_SCRIPT_HOST") == "myhost"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_brand_compat.py -v --import-mode=importlib`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.brand_compat'`.

- [ ] **Step 3: Implement the shim**

Create `src/brand_compat.py`:

```python
"""Brand env-var compatibility (Phase 4).

The product's documented env vars are ASSIST_*; the legacy names are
ODYSSEUS_*. Instead of rewriting ~1000 getenv call sites, mirror the two
prefixes into each other at process start so either name resolves. Import
this module (and/or call mirror_brand_env()) BEFORE any module that reads env
at import time (src.constants / core.constants).
"""
import os


def mirror_brand_env(environ=None) -> None:
    """Mirror ASSIST_* <-> ODYSSEUS_* env vars. setdefault never clobbers a
    value the caller set explicitly under both prefixes."""
    env = os.environ if environ is None else environ
    for k, v in list(env.items()):
        if k.startswith("ASSIST_"):
            env.setdefault("ODYSSEUS_" + k[len("ASSIST_"):], v)
        elif k.startswith("ODYSSEUS_"):
            env.setdefault("ASSIST_" + k[len("ODYSSEUS_"):], v)


# Run once on import so a bare `import src.brand_compat` is enough to bridge.
mirror_brand_env()
```

- [ ] **Step 4: Wire the shim in first at startup**

In `app.py`, make this the VERY FIRST import (above the existing `from core.constants import ...` and all other imports), so env is mirrored before any constants module reads it:

```python
import src.brand_compat  # noqa: F401  -- mirror ASSIST_*<->ODYSSEUS_* before constants read env
```

In `launcher.py`, add the same line as the first import inside the file (top, before `import os`/`sys` is fine as long as it's before `from app import app` in `main`; putting it at module top is simplest):

```python
import src.brand_compat  # noqa: F401  -- bridge ASSIST_*/ODYSSEUS_* env at startup
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_brand_compat.py -v --import-mode=importlib`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add src/brand_compat.py app.py launcher.py tests/test_brand_compat.py
git commit -m "feat(brand): ASSIST_<->ODYSSEUS_ env compatibility shim"
```

---

### Task 2: Data-dir migration to ~/.assist

**Files:**
- Modify: `src/runtime_paths.py`
- Test: `tests/test_runtime_paths.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_frozen_data_dir(home) -> str` (migrates `<home>/.odysseus/data` → `<home>/.assist/data` once); `get_default_data_dir()` now returns the `~/.assist/data` path when frozen.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_runtime_paths.py`:

```python
import os
import src.runtime_paths as rp


def test_frozen_data_dir_migrates_legacy(tmp_path):
    legacy = tmp_path / ".odysseus" / "data"
    legacy.mkdir(parents=True)
    (legacy / "app.db").write_text("data")
    got = rp._frozen_data_dir(str(tmp_path))
    assert got == str(tmp_path / ".assist" / "data")
    assert (tmp_path / ".assist" / "data" / "app.db").read_text() == "data"
    assert not legacy.exists()  # moved, not copied


def test_frozen_data_dir_no_legacy_returns_assist(tmp_path):
    got = rp._frozen_data_dir(str(tmp_path))
    assert got == str(tmp_path / ".assist" / "data")


def test_frozen_data_dir_prefers_existing_assist(tmp_path):
    (tmp_path / ".assist" / "data").mkdir(parents=True)
    (tmp_path / ".odysseus" / "data").mkdir(parents=True)
    (tmp_path / ".odysseus" / "data" / "keep.txt").write_text("x")
    got = rp._frozen_data_dir(str(tmp_path))
    assert got == str(tmp_path / ".assist" / "data")
    assert (tmp_path / ".odysseus" / "data" / "keep.txt").exists()  # legacy untouched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_runtime_paths.py -k frozen_data_dir -v --import-mode=importlib`
Expected: FAIL — `AttributeError: module 'src.runtime_paths' has no attribute '_frozen_data_dir'`.

- [ ] **Step 3: Implement migration**

In `src/runtime_paths.py`, replace `get_default_data_dir` and add the helper:

```python
def _frozen_data_dir(home: str) -> str:
    """Resolve the frozen data dir under `home`, migrating a legacy
    `.odysseus/data` to `.assist/data` once. Falls back to the legacy path
    if a migration attempt fails but legacy data exists (never lose data)."""
    new = os.path.join(home, ".assist", "data")
    legacy = os.path.join(home, ".odysseus", "data")
    try:
        if not os.path.exists(new) and os.path.isdir(legacy):
            os.makedirs(os.path.dirname(new), exist_ok=True)
            os.rename(legacy, new)
    except Exception:
        if os.path.isdir(legacy):
            return legacy
    return new


def get_default_data_dir() -> str:
    """Return the default path to the data directory.

    In normal runs, this is a 'data' subdirectory under the app root. In frozen
    builds, it is a persistent user directory (~/.assist/data), migrated once
    from the legacy ~/.odysseus/data if present.
    """
    if getattr(sys, "frozen", False):
        return _frozen_data_dir(os.path.expanduser("~"))
    return os.path.join(get_app_root(), "data")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_runtime_paths.py -v --import-mode=importlib`
Expected: PASS (existing tests + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/runtime_paths.py tests/test_runtime_paths.py
git commit -m "feat(brand): migrate frozen data dir to ~/.assist (from ~/.odysseus)"
```

---

### Task 3: Brand assets

Generate the app icon / PWA icons / favicon from `assistappicon.png` and place the wordmark from `Assistlogo.png`.

**Files:**
- Create: `scripts/build_brand_assets.py`
- Generated (committed): `static/icon.ico`, `static/icons/icon-192.png`, `static/icons/icon-512.png`, `static/icons/icon-maskable-512.png`, `static/favicon.png`, `docs/assist-wordmark.png`
- Test: `tests/test_brand_assets.py`

**Interfaces:**
- Consumes: `assistappicon.png`, `Assistlogo.png` (repo root).
- Produces: the committed asset files above.

- [ ] **Step 1: Write failing tests**

Create `tests/test_brand_assets.py`:

```python
"""Brand assets exist and have the right shapes (generated from source PNGs)."""
import pathlib

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_app_icon_ico_exists():
    assert (ROOT / "static" / "icon.ico").is_file()


def test_pwa_icons_sizes():
    assert Image.open(ROOT / "static" / "icons" / "icon-192.png").size == (192, 192)
    assert Image.open(ROOT / "static" / "icons" / "icon-512.png").size == (512, 512)
    assert Image.open(ROOT / "static" / "icons" / "icon-maskable-512.png").size == (512, 512)


def test_wordmark_present():
    assert (ROOT / "docs" / "assist-wordmark.png").is_file()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_brand_assets.py -v --import-mode=importlib`
Expected: FAIL — `static/icon.ico` will actually exist (a 174-byte placeholder) so `test_app_icon_ico_exists` may pass, but `test_pwa_icons_sizes` will FAIL if the current icons aren't square-from-source and `test_wordmark_present` FAILs (`docs/assist-wordmark.png` absent). Confirm at least one failure before implementing.

- [ ] **Step 3: Create the asset generator**

Create `scripts/build_brand_assets.py`:

```python
"""Generate Assist brand assets from the provided source PNGs (Phase 4).

Center-crops assistappicon.png to a square, resizes, and writes the Windows
app icon (.ico), the PWA icons, and a favicon; copies Assistlogo.png into docs
as the wordmark. Run once at rebrand/build time; outputs are committed.
"""
import os

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _square(img):
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def main() -> int:
    src = os.path.join(ROOT, "assistappicon.png")
    img = _square(Image.open(src).convert("RGBA"))

    img.save(os.path.join(ROOT, "static", "icon.ico"),
             sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])

    icons = os.path.join(ROOT, "static", "icons")
    img.resize((192, 192)).save(os.path.join(icons, "icon-192.png"))
    img.resize((512, 512)).save(os.path.join(icons, "icon-512.png"))
    img.resize((512, 512)).save(os.path.join(icons, "icon-maskable-512.png"))
    img.resize((32, 32)).save(os.path.join(ROOT, "static", "favicon.png"))

    logo = os.path.join(ROOT, "Assistlogo.png")
    if os.path.isfile(logo):
        Image.open(logo).save(os.path.join(ROOT, "docs", "assist-wordmark.png"))

    print("brand assets generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Generate the assets**

Run: `python scripts/build_brand_assets.py`
Expected: prints `brand assets generated`; the six output files now exist. (Pillow is already a dependency.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_brand_assets.py -v --import-mode=importlib`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit (including the generated binaries + source PNGs)**

```bash
git add scripts/build_brand_assets.py tests/test_brand_assets.py \
        assistappicon.png Assistlogo.png \
        static/icon.ico static/icons/icon-192.png static/icons/icon-512.png \
        static/icons/icon-maskable-512.png static/favicon.png docs/assist-wordmark.png
git commit -m "feat(brand): generate Assist app icon, PWA icons, favicon, wordmark"
```

---

### Task 4: User-facing display strings

Replace visible "Odysseus" → "Assist" in the UI and rendered templates. **Do NOT** touch `localStorage` keys, JS identifiers (`_odysseusLoadTime`, `odysseusInitMermaid`), or comments.

**Files:**
- Modify: `static/index.html`, `static/manifest.json`, `static/login.html`, `src/visual_report.py`
- Test: `tests/test_brand_strings.py`

**Interfaces:**
- Produces: user-facing "Assist" strings; `localStorage`/identifier preservation.

- [ ] **Step 1: Write failing/guard tests**

Create `tests/test_brand_strings.py`:

```python
"""Guards: user-facing text says 'Assist'; persistence keys/identifiers stay 'odysseus'."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_index_titles_say_assist():
    html = _read("static/index.html")
    assert "<title>Assist Chat</title>" in html
    assert "— Assist'" in html                 # route→title map entries
    assert "Message Assist..." in html          # composer placeholder
    assert ">Odysseus<" not in html             # no bare visible Odysseus label


def test_index_keeps_persistence_keys_and_identifiers():
    html = _read("static/index.html")
    assert "'odysseus-theme'" in html           # localStorage key preserved
    assert "odysseus-ui-scale" in html
    assert "_odysseusLoadTime" in html          # JS identifier preserved
    assert "odysseusInitMermaid" in html


def test_manifest_and_login_say_assist():
    man = _read("static/manifest.json")
    assert '"name": "Assist"' in man
    assert '"short_name": "Assist"' in man
    login = _read("static/login.html")
    assert "<title>Assist — Login</title>" in login
    assert "<span>Assist</span>" in login
    assert "'odysseus-theme'" in login          # login persistence key preserved


def test_report_template_says_assist():
    vr = _read("src/visual_report.py")
    assert "Assist &mdash; Deep Research Report" in vr
    assert "Generated by Assist Deep Research" in vr
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_brand_strings.py -v --import-mode=importlib`
Expected: FAIL — the display strings still say "Odysseus".

- [ ] **Step 3: Edit `static/index.html` (display text only)**

Make these exact replacements (leave every other `odysseus` occurrence — localStorage keys, `_odysseusLoadTime`, `odysseusInitMermaid`, comments — unchanged):

- `<title>Odysseus Chat</title>` → `<title>Assist Chat</title>`
- Each route→title map entry `'X — Odysseus'` → `'X — Assist'` (the 8 lines: Calendar, Notes, Cookbook, Email, Memory, Gallery, Tasks, Library).
- `(titles[path] || 'Odysseus')` → `(titles[path] || 'Assist')` (both occurrences on the `name:`/`short_name:` lines).
- `<label>Odysseus Logo</label>` → `<label>Assist Logo</label>`
- `<span class="sidebar-brand-title">Odysseus</span>` → `...>Assist</span>`
- `<h1 class="a11y-visually-hidden">Odysseus</h1>` → `...>Assist</h1>`
- `<span id="current-meta">Odysseus Chat</span>` → `...>Assist Chat</span>`
- welcome-name `...</svg>Odysseus</div>` → `...</svg>Assist</div>`
- `placeholder="Message Odysseus..."` → `placeholder="Message Assist..."`
- `<span class="vis-label">Odysseus <span class="vis-hint">Brand name</span></span>` → `...>Assist <span ...`
- admin note `...clickable links back to Odysseus inside...` → `...back to Assist inside...`

- [ ] **Step 4: Edit `static/manifest.json`**

Change `"name": "Odysseus"` → `"name": "Assist"` and `"short_name": "Odysseus"` → `"short_name": "Assist"`.

- [ ] **Step 5: Edit `static/login.html` (display only)**

- `<title>Odysseus — Login</title>` → `<title>Assist — Login</title>`
- `<span>Odysseus</span>` → `<span>Assist</span>`
- Leave `localStorage.getItem('odysseus-theme')`, `'odysseus-last-user'`, and the comments unchanged.

- [ ] **Step 6: Edit `src/visual_report.py` (rendered report display)**

- `Odysseus &mdash; Deep Research Report` → `Assist &mdash; Deep Research Report`
- `Generated by Odysseus Deep Research` → `Generated by Assist Deep Research`
- Leave the `// original Odysseus tab...` comment unchanged.

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_brand_strings.py -v --import-mode=importlib`
Expected: PASS (4 tests).

- [ ] **Step 8: Commit**

```bash
git add static/index.html static/manifest.json static/login.html src/visual_report.py tests/test_brand_strings.py
git commit -m "feat(brand): user-facing UI + report strings say Assist"
```

---

### Task 5: Docs + README

**Files:**
- Modify: `README.md`, `docs/*.md`
- Test: extend `tests/test_brand_strings.py`

**Interfaces:**
- Produces: README/docs product name "Assist"; wordmark points at `docs/assist-wordmark.png`; an env-var compatibility note.

- [ ] **Step 1: Add failing guard tests**

Append to `tests/test_brand_strings.py`:

```python
def test_readme_rebranded():
    r = _read("README.md")
    assert "docs/assist-wordmark.png" in r
    assert 'alt="Odysseus"' not in r
    assert "ASSIST_" in r  # env-var compatibility note mentions the new prefix
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_brand_strings.py::test_readme_rebranded -v --import-mode=importlib`
Expected: FAIL — README still references `docs/odysseus-wordmark.png` / `alt="Odysseus"` and has no `ASSIST_` note.

- [ ] **Step 3: Rebrand README**

In `README.md`:
- Change the wordmark line `<img src="docs/odysseus-wordmark.png" alt="Odysseus" width="238">` → `<img src="docs/assist-wordmark.png" alt="Assist" width="238">`.
- Replace standalone capitalized product name `Odysseus` → `Assist` in prose/headings (leave lowercase `odysseus` in URLs/paths like the GitHub link and repology badge, and leave `ODYSSEUS_` env examples).
- Add a short note under configuration: `Environment variables use the ` + "`ASSIST_*`" + ` prefix (e.g. ` + "`ASSIST_PORT`, `ASSIST_DATA_DIR`" + `). The legacy ` + "`ODYSSEUS_*`" + ` names are still honored for backward compatibility.`

- [ ] **Step 4: Rebrand the other docs**

For each file in `docs/*.md` (excluding `docs/superpowers/**`), replace the standalone capitalized word `Odysseus` → `Assist` (product name in prose/headings). Leave lowercase `odysseus` (URLs/paths/filenames) and `ODYSSEUS_` env-var names as-is. Command:

```bash
for f in docs/*.md; do sed -i 's/\bOdysseus\b/Assist/g' "$f"; done
```

(`\bOdysseus\b` matches only the exact-case product word, not `ODYSSEUS_` or lowercase `odysseus`.)

- [ ] **Step 5: Run the guard test**

Run: `python -m pytest tests/test_brand_strings.py -v --import-mode=importlib`
Expected: PASS (all brand-string tests, including `test_readme_rebranded`).

- [ ] **Step 6: Commit**

```bash
git add README.md docs/*.md tests/test_brand_strings.py
git commit -m "docs(brand): rebrand README + docs to Assist + env-var note"
```

---

## Appendix: Manual smoke test (real acceptance gate)

1. Launch the app → the browser tab, window title, sidebar brand, welcome screen, and composer placeholder read **Assist**; the new favicon/app icon/wordmark render.
2. Set `ASSIST_PORT=7100` (with no `ODYSSEUS_PORT`) → the app binds 7100 (shim bridge works); set `ODYSSEUS_PORT` → still works.
3. On a frozen build with an existing `~/.odysseus/data`, launch → data is now under `~/.assist/data` and prior sessions/settings are intact.
4. Generate a Deep Research report → the header/footer read "Assist".

## Self-Review

**Spec coverage:**
- Env shim (spec C1) → Task 1 (`brand_compat.py` + app/launcher wiring). ✓
- Data-dir migration (C2) → Task 2. ✓
- Brand assets (C3) → Task 3. ✓
- User-facing strings (C4) → Task 4 (index/manifest/login/report). ✓
- Docs (C5) → Task 5. ✓
- "Keep internal odysseus" constraint → guarded by Task 4's `test_index_keeps_persistence_keys_and_identifiers` and the scoped edits. ✓
- Testing (unit shim/migration + text-guards + manual smoke) → each task + appendix. ✓

**Placeholder scan:** No TBD/TODO; every code/edit step is explicit. The doc `sed` uses `\bOdysseus\b` (exact-case) so it can't touch `ODYSSEUS_`/lowercase paths. Manual/visual steps are labeled and backed by text-guards + the smoke appendix. ✓

**Type consistency:** `mirror_brand_env(environ=None)` signature matches across Task 1 impl, tests, and the app/launcher import. `_frozen_data_dir(home)` / `get_default_data_dir()` consistent between Task 2 impl and tests. Asset output paths (`static/icon.ico`, `static/icons/icon-192.png` etc., `docs/assist-wordmark.png`) identical between Task 3 generator, its tests, and Task 5's README reference. The display strings asserted in Task 4/5 guards exactly match the edit instructions. Persistence keys guarded as preserved (`'odysseus-theme'`, `odysseus-ui-scale`, `_odysseusLoadTime`, `odysseusInitMermaid`). ✓
