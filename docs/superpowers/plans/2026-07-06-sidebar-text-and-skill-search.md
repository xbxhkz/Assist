# Sidebar Text + GitHub Skill Search — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Enlarge all sidebar text, and add an in-app GitHub `SKILL.md` search that installs via the existing importer (with an optional GitHub token unlocking code search).

**Architecture:** A pure `skill_search.search_skills(query, token, http=…)` (injectable HTTP for tests) hits GitHub's code-search API when a token is set, else repo-search + a root-`SKILL.md` probe; a `/api/skills/discover` route wires it with `get_github_token()`; the UI reuses `import-from-url` to install. `github_token` is a normal admin setting like the existing `*_api_key`s. Sidebar text is one scoped CSS block.

**Tech Stack:** FastAPI, httpx, vanilla JS/CSS. Tests: pytest with injected fakes.

## Global Constraints

- `github_token` is optional; stored in `data/settings.json` like `brave_api_key` (admin-only Settings).
- Reuse `POST /api/skills/import-from-url` for install; do NOT add a new downloader.
- Unified search result shape: `{"name","description","repo","stars","url","source"}` where `source ∈ {"code","repo"}` and `url` is a github.com/raw URL importable by `fetch_skill_bundle`.
- No-token mode finds only root-level `SKILL.md` (documented limitation).

---

### Task 1: Larger sidebar text

**Files:** Modify `static/style.css`; Test `tests/test_sidebar_text_size.py`

- [ ] **Step 1 — failing test** `tests/test_sidebar_text_size.py`:
```python
from pathlib import Path
CSS = (Path(__file__).resolve().parent.parent / "static" / "style.css").read_text(encoding="utf-8")
def test_sidebar_text_enlarged():
    assert "/* Larger sidebar text" in CSS
    assert ".sidebar .list-item .grow" in CSS
```
- [ ] **Step 2 — run, expect FAIL:** `python -m pytest tests/test_sidebar_text_size.py --import-mode=importlib -q`
- [ ] **Step 3 — implement:** append to `static/style.css`:
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
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit:** `git add static/style.css tests/test_sidebar_text_size.py && git commit -m "feat(ui): larger sidebar text"`

---

### Task 2: GitHub token setting

**Files:** Modify `src/settings.py` (DEFAULT_SETTINGS + helper); Test `tests/test_github_token_setting.py`

**Interfaces:** Produces `src.settings.get_github_token() -> str`.

- [ ] **Step 1 — failing test**:
```python
import src.settings as s
def test_github_token_default_empty(monkeypatch):
    monkeypatch.setattr(s, "load_settings", lambda: dict(s.DEFAULT_SETTINGS))
    assert s.get_github_token() == ""
def test_github_token_read(monkeypatch):
    monkeypatch.setattr(s, "load_settings", lambda: {**s.DEFAULT_SETTINGS, "github_token": "ghp_x"})
    assert s.get_github_token() == "ghp_x"
```
- [ ] **Step 2 — run, expect FAIL** (`get_github_token`/`github_token` absent).
- [ ] **Step 3 — implement:** in `src/settings.py` add `"github_token": "",` to `DEFAULT_SETTINGS`, and:
```python
def get_github_token() -> str:
    """Optional GitHub PAT — unlocks code search + higher API rate limits."""
    return (get_setting("github_token", "") or "").strip()
```
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit.**

---

### Task 3: Search backend `services/memory/skill_search.py`

**Files:** Create `services/memory/skill_search.py`, `tests/test_skill_search.py`

**Interfaces:** Produces `search_skills(query: str, token: str = "", limit: int = 15, http=None) -> list[dict]` and `class SkillSearchError(Exception)`. `http` is a callable `(method, url, headers) -> (status:int, json:dict)`; default uses httpx.

- [ ] **Step 1 — failing tests** `tests/test_skill_search.py`:
```python
import pytest
from services.memory import skill_search as ss

def _http(routes):
    def call(method, url, headers=None):
        for frag, resp in routes.items():
            if frag in url:
                return resp
        return (404, {})
    return call

def test_code_search_with_token_maps_items():
    routes = {"search/code": (200, {"items": [
        {"name": "SKILL.md", "path": "skills/foo/SKILL.md",
         "html_url": "https://github.com/o/r/blob/main/skills/foo/SKILL.md",
         "repository": {"full_name": "o/r", "description": "d", "stargazers_count": 5}}]})}
    out = ss.search_skills("foo", token="ghp_x", http=_http(routes))
    assert out[0]["source"] == "code"
    assert out[0]["repo"] == "o/r"
    assert out[0]["url"].endswith("/SKILL.md")

def test_repo_search_no_token_includes_only_repos_with_skill_md():
    routes = {
        "search/repositories": (200, {"items": [
            {"full_name": "o/has", "description": "d", "stargazers_count": 3, "default_branch": "main",
             "html_url": "https://github.com/o/has"},
            {"full_name": "o/no", "description": "d", "stargazers_count": 1, "default_branch": "main",
             "html_url": "https://github.com/o/no"}]}),
        "o/has/main/SKILL.md": (200, {}),   # raw probe: exists
        "o/no/main/SKILL.md": (404, {}),    # raw probe: missing
    }
    out = ss.search_skills("foo", token="", http=_http(routes))
    repos = [r["repo"] for r in out]
    assert "o/has" in repos and "o/no" not in repos
    assert out[0]["source"] == "repo"

def test_rate_limit_raises():
    routes = {"search/repositories": (403, {"message": "rate limit"})}
    with pytest.raises(ss.SkillSearchError):
        ss.search_skills("foo", token="", http=_http(routes))
```
- [ ] **Step 2 — run, expect FAIL** (module missing).
- [ ] **Step 3 — implement** `services/memory/skill_search.py`:
```python
"""Search GitHub for SKILL.md files. With a token: code search (files anywhere).
Without: repo search + a root-SKILL.md probe. Returns a unified shape the
existing import-from-url flow can install."""
import urllib.parse

API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"


class SkillSearchError(Exception):
    pass


def _default_http(method, url, headers=None):
    import httpx
    with httpx.Client(timeout=10, follow_redirects=True) as c:
        r = c.request(method, url, headers=headers or {})
        try:
            body = r.json()
        except Exception:
            body = {}
        return r.status_code, body


def _headers(token):
    h = {"Accept": "application/vnd.github+json", "User-Agent": "Assist"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _fail(status, body):
    msg = (body or {}).get("message", "") if isinstance(body, dict) else ""
    if status in (403, 429) or "rate limit" in msg.lower():
        raise SkillSearchError("GitHub rate limit — add a GitHub token in Settings for higher limits.")
    if status == 401:
        raise SkillSearchError("GitHub token is invalid.")
    raise SkillSearchError(f"GitHub search failed ({status}).")


def _code_search(query, token, limit, http):
    q = urllib.parse.quote(f"filename:SKILL.md {query}")
    status, body = http("GET", f"{API}/search/code?q={q}&per_page={limit}", _headers(token))
    if status != 200:
        _fail(status, body)
    out = []
    for it in (body.get("items") or [])[:limit]:
        repo = (it.get("repository") or {})
        out.append({
            "name": (it.get("path") or it.get("name") or "SKILL.md").split("/")[-2:][0]
                    if "/" in (it.get("path") or "") else repo.get("full_name", "skill"),
            "description": repo.get("description") or "",
            "repo": repo.get("full_name") or "",
            "stars": int(repo.get("stargazers_count") or 0),
            "url": it.get("html_url") or "",
            "source": "code",
        })
    return out


def _repo_search(query, limit, http):
    q = urllib.parse.quote(f"{query} skill in:name,description,readme")
    status, body = http("GET", f"{API}/search/repositories?q={q}&sort=stars&per_page={limit}", _headers(""))
    if status != 200:
        _fail(status, body)
    out = []
    for it in (body.get("items") or [])[:limit]:
        full = it.get("full_name") or ""
        branch = it.get("default_branch") or "main"
        pstatus, _ = http("GET", f"{RAW}/{full}/{branch}/SKILL.md", {"User-Agent": "Assist"})
        if pstatus != 200:
            continue
        out.append({
            "name": full.split("/")[-1],
            "description": it.get("description") or "",
            "repo": full,
            "stars": int(it.get("stargazers_count") or 0),
            "url": f"https://github.com/{full}/blob/{branch}/SKILL.md",
            "source": "repo",
        })
    return out


def search_skills(query: str, token: str = "", limit: int = 15, http=None) -> list:
    query = (query or "").strip()
    if not query:
        return []
    http = http or _default_http
    return _code_search(query, token, limit, http) if token else _repo_search(query, limit, http)
```
- [ ] **Step 4 — run, expect PASS** (3 tests).
- [ ] **Step 5 — commit.**

---

### Task 4: `/discover` route

**Files:** Modify `routes/skills_routes.py` (add route near `/search`, line ~1651)

- [ ] **Step 1 — implement** inside `setup_skills_routes`, before `return router`:
```python
    @router.post("/discover")
    async def discover_skills(request: Request):
        require_admin(request)
        from services.memory.skill_search import search_skills, SkillSearchError
        from src.settings import get_github_token
        body = await request.json()
        query = (body.get("query") or "").strip()
        if not query:
            raise HTTPException(400, "query is required")
        token = get_github_token()
        try:
            results = await _asyncio.to_thread(search_skills, query, token)
        except SkillSearchError as e:
            raise HTTPException(502, str(e))
        return {"results": results, "token_set": bool(token)}
```
(Confirm `_asyncio` is imported in the file; if it's `asyncio`, use that.)
- [ ] **Step 2 — verify import/app boots:** `python -c "import routes.skills_routes"` (Expected: no error).
- [ ] **Step 3 — commit.**

---

### Task 5: Settings token field + Skills search UI

**Files:** Modify the Settings UI (add a `github_token` text field alongside the other API keys) and the Skills UI (search box + results + Add). Exact selectors located during implementation (`static/js/settings.js`, the skills modal in `static/index.html` / its JS).

- [ ] **Step 1** — Settings: add a "GitHub token (optional)" input bound to the `github_token` setting, saved through the existing settings-save path (same as `brave_api_key`).
- [ ] **Step 2** — Skills UI: a search box + "Search GitHub" button → `POST /api/skills/discover {query}` → render rows (name · repo · ⭐ · description + **Add**). **Add** → `POST /api/skills/import-from-url {url}` → on ok, refresh installed skills + toast. Show the "add a token for full search" hint when `token_set` is false.
- [ ] **Step 3** — manual check in the frozen app; **commit.**

---

### Task 6: Rebuild + verify

- [ ] Kill any `Assist`/`llama-server`, `python -m PyInstaller --noconfirm --clean Assist.spec`, then ISCC. Boot; confirm `/api/skills/discover` responds (repo mode without a token) and the sidebar text is larger.

## Self-Review

- **Spec coverage:** sidebar (T1) ✓; token setting (T2) ✓; search backend both modes (T3) ✓; route (T4) ✓; UI + Settings field (T5) ✓; rebuild (T6) ✓.
- **Placeholders:** none in T1–T4 (full code). T5 UI selectors are located at implementation time (existing-UI integration), not invented.
- **Consistency:** `search_skills(query, token, limit, http)`, `SkillSearchError`, result shape, `get_github_token()` consistent across tasks.
