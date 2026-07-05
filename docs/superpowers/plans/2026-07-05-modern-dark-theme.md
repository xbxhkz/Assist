# Graphite Modern-Dark Theme + Rounded Windows — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle Assist's default `dark` theme into the "Graphite" modern-dark palette and give modals/cards/controls consistent rounded corners via radius tokens, without touching the other presets, the light theme, or custom themes.

**Architecture:** The default palette lives in two synced places — `:root` in `static/style.css` and `THEMES.dark` in `static/js/theme.js`. Rounding is added as three `:root` radius tokens routed through the shared base selectors (`.modal-content`, base controls). A one-time load migration upgrades an *untouched* saved `dark` snapshot so existing default users see the change.

**Tech Stack:** Vanilla CSS + ES-module JS (no build step). Tests are pytest text-guards over the source files (repo convention — no JS test runner).

## Global Constraints

- Graphite tokens (exact): `--bg:#13151A`, `--panel:#1C1F27`, `--fg:#E6E9EE`, `--border:#2A2F3A`, `--red:#45C4B0`.
- `:root` and `THEMES.dark` MUST hold identical values for the 5 tokens.
- Do NOT modify: the other 16 `THEMES` presets, `:root.light`, custom themes, docked/mobile-sheet modal radius overrides (radius `0` / `14px 14px 0 0`), or the OS/WebView2 window frame.
- Radius tokens: `--radius-window:16px`, `--radius-card:12px`, `--radius-control:8px`.

---

### Task 1: Graphite palette + syntax surfaces

**Files:**
- Modify: `static/style.css` `:root` block (palette lines `--bg/--fg/--panel/--border/--red`, and `--hl-bg`, `--hl-fg`)
- Modify: `static/js/theme.js` `THEMES.dark` (line ~12)
- Test: `tests/test_theme_palette_sync.py`

**Interfaces:**
- Produces: the Graphite `:root` defaults and `THEMES.dark`, consumed by Task 2 (migration reads the new `THEMES.dark`) and Task 3 (radius tokens live in the same `:root`).

- [ ] **Step 1: Write the failing test** — `tests/test_theme_palette_sync.py`

```python
"""Guard: the default theme's :root palette (style.css) and THEMES.dark
(theme.js) must stay identical, and be the approved Graphite values."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSS = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
JS = (ROOT / "static" / "js" / "theme.js").read_text(encoding="utf-8")

GRAPHITE = {"bg": "#13151A", "panel": "#1C1F27", "fg": "#E6E9EE",
            "border": "#2A2F3A", "red": "#45C4B0"}

def _root_var(name: str) -> str:
    # first :root definition wins (the default dark theme block)
    m = re.search(r"--%s:\s*(#[0-9a-fA-F]{6})" % name, CSS)
    assert m, f"--{name} not found in style.css"
    return m.group(1).upper()

def _themes_dark(key: str) -> str:
    block = re.search(r"dark:\s*\{([^}]*)\}", JS).group(1)
    m = re.search(r"%s:\s*'(#[0-9a-fA-F]{6})'" % key, block)
    assert m, f"{key} not found in THEMES.dark"
    return m.group(1).upper()

def test_root_matches_graphite():
    for css_name, key in [("bg", "bg"), ("panel", "panel"), ("fg", "fg"),
                          ("border", "border"), ("red", "red")]:
        assert _root_var(css_name) == GRAPHITE[key], css_name

def test_themes_dark_matches_root():
    for css_name, key in [("bg", "bg"), ("panel", "panel"), ("fg", "fg"),
                          ("border", "border"), ("red", "red")]:
        assert _themes_dark(key) == _root_var(css_name), key
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_theme_palette_sync.py --import-mode=importlib -q`
Expected: FAIL (current `:root --bg` is `#282c34`, not `#13151A`).

- [ ] **Step 3: Update `static/style.css` `:root`** — replace the five palette values and the two surface-dependent syntax vars:

```css
  --bg: #13151A;
  --fg: #E6E9EE;
  --panel: #1C1F27;
  --border: #2A2F3A;
  --red: #45C4B0;
```
and in the syntax block:
```css
  --hl-bg: #171A21;
  --hl-fg: #E6E9EE;
```
(Leave `--green`, `--warn`, all `--color-*`, and the other `--hl-*` token hues unchanged. Do NOT touch the `:root.light` block.)

- [ ] **Step 4: Update `static/js/theme.js` `THEMES.dark`**

```javascript
  dark:       { bg:'#13151A', fg:'#E6E9EE', panel:'#1C1F27', border:'#2A2F3A', red:'#45C4B0' },
```
(Leave every other entry in `THEMES` exactly as-is.)

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_theme_palette_sync.py --import-mode=importlib -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add static/style.css static/js/theme.js tests/test_theme_palette_sync.py
git commit -m "feat(theme): Graphite palette for the default dark theme"
```

---

### Task 2: Radius tokens + rounded windows/controls

**Files:**
- Modify: `static/style.css` — add tokens to `:root`; route `.modal-content` (~line 5284) and base controls (~line 1796)
- Test: `tests/test_theme_radius_tokens.py`

**Interfaces:**
- Consumes: the `:root` block from Task 1 (tokens added to the same block).

- [ ] **Step 1: Write the failing test** — `tests/test_theme_radius_tokens.py`

```python
"""Guard: radius tokens exist and the shared window/control selectors use
them (so rounding is consistent and tunable, not scattered literals)."""
import re
from pathlib import Path

CSS = (Path(__file__).resolve().parent.parent / "static" / "style.css").read_text(encoding="utf-8")

def test_radius_tokens_defined():
    for tok, val in [("--radius-window", "16px"), ("--radius-card", "12px"),
                     ("--radius-control", "8px")]:
        assert re.search(rf"{tok}:\s*{val}", CSS), tok

def test_modal_content_uses_window_token():
    block = re.search(r"\.modal-content \{(.*?)\}", CSS, re.S).group(1)
    assert "border-radius:var(--radius-window)" in block.replace(" ", "")

def test_base_controls_use_control_token():
    block = re.search(r"input, textarea, button, select \{(.*?)\}", CSS, re.S).group(1)
    assert "border-radius:var(--radius-control)" in block.replace(" ", "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_theme_radius_tokens.py --import-mode=importlib -q`
Expected: FAIL (tokens not defined; selectors use literal `10px` / `4px`).

- [ ] **Step 3: Add the tokens to `static/style.css` `:root`** (after the core palette, before the syntax block):

```css
  /* Window rounding tokens (see docs/superpowers/specs/2026-07-05-modern-dark-theme-design.md) */
  --radius-window: 16px;   /* modals / dialogs */
  --radius-card: 12px;     /* cards, menus, nested panels */
  --radius-control: 8px;   /* buttons, inputs, selects, chips */
```

- [ ] **Step 4: Route the shared base selectors** — change the base `.modal-content` rule's `border-radius:10px;` to `border-radius:var(--radius-window);` (the rule at ~line 5284 that also sets `background:var(--panel)`), and the base `input, textarea, button, select` rule's `border-radius: 4px;` (~line 1796) to `border-radius: var(--radius-control);`. Do NOT change the docked/mobile-sheet `.modal-content` overrides that set radius to `0` or `14px 14px 0 0`.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_theme_radius_tokens.py --import-mode=importlib -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add static/style.css tests/test_theme_radius_tokens.py
git commit -m "feat(theme): radius tokens; round modals and base controls"
```

---

### Task 3: Upgrade untouched saved `dark` snapshots

**Files:**
- Modify: `static/js/theme.js` — add a migration helper and call it before the initial `applyColors(currentColors)` (~line 757–758)
- Test: `tests/test_theme_dark_migration.py`

**Interfaces:**
- Consumes: `THEMES.dark` (Task 1). The migration compares a saved snapshot against the *previous* dark palette (hardcoded in the helper) and, if unchanged, swaps in the new `THEMES.dark`.

- [ ] **Step 1: Write the failing test** — `tests/test_theme_dark_migration.py`

```python
"""Guard: theme.js carries a one-time migration that upgrades an UNTOUCHED
saved 'dark' snapshot to the new palette, and leaves tweaked/other themes
alone. (Text-guard — repo has no JS test runner.)"""
import re
from pathlib import Path

JS = (Path(__file__).resolve().parent.parent / "static" / "js" / "theme.js").read_text(encoding="utf-8")

def test_migration_helper_present():
    assert "PREV_DARK" in JS
    # the previous dark palette it compares against, verbatim
    assert "#282c34" in JS and "#9cdef2" in JS and "#355a66" in JS
    assert re.search(r"function _migrateDarkSnapshot", JS)

def test_migration_is_invoked_before_initial_apply():
    # helper is called (name appears at least twice: definition + call)
    assert JS.count("_migrateDarkSnapshot") >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_theme_dark_migration.py --import-mode=importlib -q`
Expected: FAIL (`_migrateDarkSnapshot` not defined).

- [ ] **Step 3: Add the migration helper in `static/js/theme.js`** (near the other module-level helpers, above the init that reads `getSaved()`):

```javascript
// One-time upgrade: when the built-in "dark" theme was restyled to Graphite,
// a user who had previously selected the OLD dark keeps a frozen snapshot in
// localStorage and would never see the new palette. If their snapshot is the
// UNTOUCHED old dark (they never hand-tweaked a color), swap in the new one.
// A customized snapshot (any color differs) is left exactly as the user set it.
const PREV_DARK = { bg:'#282c34', fg:'#9cdef2', panel:'#111111', border:'#355a66', red:'#e06c75' };
function _migrateDarkSnapshot(saved) {
  if (!saved || saved.name !== 'dark' || !saved.colors) return saved;
  const c = saved.colors;
  const untouched = ['bg','fg','panel','border','red'].every(
    k => (c[k] || '').toLowerCase() === PREV_DARK[k]);
  if (untouched) {
    saved.colors = { ...c, ...THEMES.dark };
    try { _saveFull(saved.name, saved.colors); } catch (e) {}
  }
  return saved;
}
```

- [ ] **Step 4: Invoke it where the saved theme is first read for the initial apply** — at the init that does `const saved = getSaved();` then `const currentColors = saved ? saved.colors : THEMES[DEFAULT_THEME]; applyColors(currentColors);` (~line 757), insert immediately after `getSaved()`:

```javascript
  saved = _migrateDarkSnapshot(saved);
```
(If `saved` is declared `const`, change that one declaration to `let saved`. Ensure `_saveFull` is in scope at the helper's call site; if it is defined later as a nested function, inline the persistence with the module-level `save(name, colors, opts)` used elsewhere instead.)

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_theme_dark_migration.py --import-mode=importlib -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add static/js/theme.js tests/test_theme_dark_migration.py
git commit -m "feat(theme): migrate untouched saved dark snapshot to Graphite"
```

---

### Task 4: Frozen-app visual verification

**Files:** none (verification only)

- [ ] **Step 1:** Copy updated static into the bundle and launch:
```bash
cp static/style.css static/js/theme.js dist/Assist/_internal/static/style.css dist/Assist/_internal/static/js/theme.js 2>/dev/null || true
# (copy each to its matching path under dist/Assist/_internal/static/)
```
- [ ] **Step 2:** Launch `dist/Assist/Assist.exe`, open the app, confirm: default look is Graphite (dark slate bg, teal accent, near-white text); modals (Local Models, Settings) have 16px rounded corners; buttons/inputs are subtly rounded (8px).
- [ ] **Step 3:** In Settings → Theme, select `midnight` — confirm it is unchanged; select the light theme — confirm unchanged. Re-select `dark` — confirm Graphite.
- [ ] **Step 4:** Rebuild the installer (`ISCC.exe //DMyAppVersion=1.1.0 installer/Assist.iss`) once visuals are confirmed.

---

## Self-Review

**Spec coverage:** palette (Task 1) ✓; syntax surfaces (Task 1) ✓; radius tokens + rounded windows/controls (Task 2) ✓; untouched-snapshot migration (Task 3) ✓; non-goals respected (docked/mobile overrides and other presets untouched — called out in Task 2/Global Constraints) ✓; verification (Task 4) ✓.

**Placeholder scan:** none — all edits give exact selectors/values.

**Type/name consistency:** `_migrateDarkSnapshot`, `PREV_DARK`, `THEMES.dark`, `--radius-window/card/control` used consistently across tasks and tests.

**Known residual:** `--radius-card` is defined and available but only lightly applied (message bubbles are already 12–18px rounded; cards/menus can be routed opportunistically during Task 2 Step 4 without new literals). This is intentional YAGNI — the high-impact windows are modals + controls.
