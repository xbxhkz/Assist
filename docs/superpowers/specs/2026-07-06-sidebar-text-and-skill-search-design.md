# Sidebar Text Size + In-App GitHub Skill Search — Design

**Date:** 2026-07-06
**Status:** Approved

Two independent changes requested together.

---

## Part A — Larger sidebar text

**Goal:** Make **all** sidebar text a little bigger — chat/session names, section
headers (Chats, Email, …), the Search / New Chat / Assistant buttons, and the
brand title — so the sidebar is easier to read.

**Approach:** The sidebar's text sizes are set across several rules (base +
responsive media queries + density modes), mostly landing around 13–14px. Add
one scoped block that bumps the sidebar text elements up by ~1.5px with enough
specificity to win, without touching the density/responsive logic:

```css
/* Larger sidebar text (readability) */
.sidebar .list-item .grow,
.sidebar .section-header-flex .section-title,
.sidebar .section-header-flex h4,
.sidebar-brand-title,
#sidebar-search-btn .grow,
#sidebar-new-chat-btn .grow,
.sidebar-assistant-entry .grow { font-size: 15px; }
```

The exact px is tunable — the user confirms visually and we adjust one number.

**Non-goals:** the main chat area, modals, or non-sidebar text. **Test:** a
text-guard asserting the sidebar font-size block exists.

---

## Part B — In-app GitHub skill search

**Goal:** Search GitHub for `SKILL.md` files from inside the app and install one
with a click, instead of browsing GitHub in a web browser and pasting a link.

**Reuses:** the existing `POST /api/skills/import-from-url` +
`services/memory/skill_importer.fetch_skill_bundle` (accepts github.com blob /
raw / skills.sh URLs). The new work is the **search** that feeds it.

### B1. GitHub token setting (optional)

A new optional server-side setting `github_token` (admin-only, stored with the
app's other settings). Purposes:
- Unlocks GitHub **code search** (unauthenticated code search is not allowed).
- Raises the GitHub API rate limit (60/hr → 5000/hr) for search + downloads.

Read via a helper `get_github_token()`; used by both the search and (optionally)
the importer's requests. Never returned to the client in plaintext (the settings
GET returns only whether it is set).

### B2. Search backend — `services/memory/skill_search.py` (new)

`search_skills(query: str, token: str | None, limit: int = 15) -> list[dict]`
returning a unified shape per result:
`{ "name", "description", "repo", "stars", "url", "source" }`
where `url` is a GitHub link `fetch_skill_bundle` can import, and `source` is
`"code"` or `"repo"`.

- **With token (code search):**
  `GET https://api.github.com/search/code?q=filename:SKILL.md <query>` with
  `Authorization: Bearer <token>`. Each item → `url` = its `html_url` (the
  SKILL.md blob), `repo` = `repository.full_name`, `name` = repo/last path
  segment. Finds `SKILL.md` anywhere in public repos.
- **Without token (repo search):**
  `GET https://api.github.com/search/repositories?q=<query> skill in:name,description,readme sort:stars`.
  For the top N repos, probe for a top-level skill file at
  `raw.githubusercontent.com/<full_name>/<default_branch>/SKILL.md` (HEAD/GET);
  include repos where it exists, `url` = the repo's `blob/<branch>/SKILL.md`,
  `stars` = `stargazers_count`. Covers root-level `SKILL.md`; nested files need a
  token (documented limitation).
- **Errors:** network / rate-limit / bad-token → raise a typed error the route
  turns into a clear message (e.g. "GitHub rate limit — add a token in Settings").
  All GitHub hosts validated (reuse `skill_importer` host allowlist ethos).

### B3. Route — `routes/skills_routes.py`

`POST /api/skills/discover` `{query}` (admin-guarded) → `get_github_token()` →
`search_skills(query, token)` → `{ "results": [...], "token_set": bool }`.
Install reuses the existing `POST /api/skills/import-from-url {url}`.

### B4. UI — the Skills area (`static/`)

- A **search box + Search button** ("Search GitHub for skills…").
- Results list: each row shows **name · repo · ⭐stars · description** and an
  **Add** button → `POST /skills/import-from-url {url:result.url}` → on success,
  refresh the installed-skills list + toast "Added <name>".
- A subtle hint when `token_set` is false: "Add a GitHub token in Settings for
  full file-level search." A field to set the token lives in Settings.

### B5. Testing

- `search_skills` with a token → parses a mocked `search/code` response into the
  unified shape (injected HTTP client).
- Without a token → parses a mocked `search/repositories` response + a mocked
  raw-`SKILL.md` probe; repos without a `SKILL.md` are excluded.
- Rate-limit / error mapping raises the typed error.
- `get_github_token()` reads the setting; the settings GET never leaks the value.

## Scope / non-goals

- No new download mechanism (reuse `import-from-url`).
- No auth beyond the existing admin guard; the token is optional.
- No-token search is repo-level (root `SKILL.md`); full file search needs a token.

## Files touched

- Create: `services/memory/skill_search.py`, `tests/test_skill_search.py`
- Modify: `routes/skills_routes.py` (`/discover` + `get_github_token`), the
  settings store/route (add `github_token`), `static/` skills UI + a Settings
  field, `static/style.css` (sidebar text block).
