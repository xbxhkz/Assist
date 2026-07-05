# Modern Dark Theme ("Graphite") + Rounded Windows — Design

**Date:** 2026-07-05
**Status:** Approved (direction: Graphite)

## Goal

Restyle Assist's **default theme** into a modern, neutral dark look ("Graphite"),
and give the app's **windows** (modals, side panels, cards) consistent rounded
corners. The other 16 built-in presets, the light theme, and user custom themes
are left untouched.

## Background — how theming works today

- **Two sources of truth for the default palette**, which must stay in sync:
  1. `static/style.css` `:root { --bg; --fg; --panel; --border; --red; … }` — the
     CSS default / pre-JS fallback.
  2. `static/js/theme.js` `THEMES.dark = { bg, fg, panel, border, red }` — the
     preset that `applyColors()` writes to the CSS variables at runtime.
     `DEFAULT_THEME = 'dark'`.
- **Selected themes are saved as a color snapshot.** On load
  (`theme.js`), `const currentColors = saved ? saved.colors : THEMES[DEFAULT_THEME]`.
  So a user who has previously selected "Dark" carries a *frozen copy* of the old
  palette in `localStorage['odysseus-theme'].colors`; editing `THEMES.dark` alone
  will not reach them until they re-select the theme.
- **Border-radius is hardcoded** throughout `static/style.css` (values 4/6/8/10px,
  and larger ad-hoc values) — there is no radius token. Radius is global CSS, not
  part of the 5-token palette, so it applies across all themes.
- The desktop shell is a WebView2/pywebview window; on Windows 11 the OS already
  rounds the outer window frame. "Rounded windows" here means the **in-app**
  surfaces, not the OS frame.

## Design

### 1. Graphite palette (the 5 theme tokens)

| Token      | Old (dark)   | New (Graphite) | Role |
|------------|--------------|----------------|------|
| `--bg`     | `#282c34`    | `#13151A`      | app background |
| `--panel`  | `#111111`    | `#1C1F27`      | elevated surface (panels, cards, bubbles, inputs) |
| `--fg`     | `#9cdef2`    | `#E6E9EE`      | primary text — neutral near-white (color now carried by the accent, not the text) |
| `--border` | `#355a66`    | `#2A2F3A`      | subtle low-contrast separators |
| `--red`    | `#e06c75`    | `#45C4B0`      | accent (`--red` is the app's accent slot: send button, active nav, scrollbar, links) — a refined teal that evolves the app's current cyan |

Neutral `--fg` (rather than a tinted cyan) is intentional: the accent carries the
color and text stays legible for long sessions. WCAG AA is met: `#E6E9EE` on
`#13151A` ≈ 15:1; `#E6E9EE` on `#1C1F27` ≈ 13:1; accent `#45C4B0` on `#13151A`
≈ 8:1.

### 2. Syntax-highlight vars (default `:root` only)

Adjust the two surface-dependent tokens so code blocks sit on the new palette;
keep the token hues (they read well on dark):
- `--hl-bg`: `#1e2228` → `#171A21`
- `--hl-fg`: `#9cdef2` → `#E6E9EE`

### 3. Radius tokens + "rounded windows"

Introduce three tokens in `:root` and route the app's containers through them
(replacing the relevant hardcoded values):

```
--radius-window:  16px;   /* modals / dialogs, primary panels & sidebar container */
--radius-card:    12px;   /* cards, list items, message bubbles, dropdown menus, nested panels */
--radius-control:  8px;   /* buttons, inputs, selects, chips, small controls */
```

- **Windows/modals** (e.g. the Cookbook, Local Models, Settings dialogs and the
  main panel/sidebar shells) → `--radius-window`.
- **Cards / bubbles / menus** → `--radius-card`.
- **Buttons / inputs / selects / chips** → `--radius-control` (a subtle bump from
  today's 4–6px, not fully pill-shaped).

Tokens (not scattered literals) keep rounding consistent and tunable, and apply
across every theme uniformly. Values are theme-independent (rounding is not part
of the palette).

### 4. Making the restyle actually reach the current default user

`:root` and `THEMES.dark` are updated together. To avoid a stale snapshot hiding
the change, add a **safe, one-time migration** on theme load:

- If a saved theme exists, its `name === 'dark'`, and its `colors` **deep-equals
  the previous `THEMES.dark`** (i.e. the user never hand-tweaked it), replace the
  saved `colors` with the new `THEMES.dark` and re-save.
- If the saved colors differ (the user customized), **leave them alone**.
- New users (no saved theme) get Graphite directly via `THEMES[DEFAULT_THEME]`.

This guarantees untouched default users adopt Graphite with no action, while
never clobbering a customization.

## Scope / non-goals

- **In scope:** `:root` default palette + `:root` syntax `--hl-bg/--hl-fg` +
  radius tokens & their application to windows/cards/controls + `THEMES.dark` +
  the untweaked-snapshot migration.
- **Out of scope (do not touch):** the other 16 presets in `THEMES`, `:root.light`,
  user custom themes, the OS/WebView2 window frame, and any functional/JS behavior
  beyond the migration guard. Buttons/inputs get only a subtle radius bump, not a
  pill restyle.

## Testing / verification

1. **Palette sync guard (unit):** a text/JS test asserting `THEMES.dark` equals the
   `:root` default values, so the two sources can't drift.
2. **Migration (unit):** untweaked saved `dark` snapshot → upgraded to new palette;
   a hand-tweaked snapshot → left unchanged; a non-`dark` saved theme → untouched.
3. **Existing theme tests** continue to pass (no preset/custom regressions).
4. **Visual check in the frozen app:** default look is Graphite; modals/cards/panels
   are rounded per the token values; a non-default preset (e.g. `midnight`) is
   unchanged; the light theme is unchanged.

## Files touched

- `static/style.css` — `:root` palette + `--hl-bg/--hl-fg` + new `--radius-*`
  tokens; replace targeted hardcoded `border-radius` values on windows/cards/controls.
- `static/js/theme.js` — `THEMES.dark` new values + the untweaked-`dark`-snapshot
  migration on load.
- Tests — palette-sync guard + migration tests.
