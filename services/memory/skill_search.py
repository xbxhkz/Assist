"""Search GitHub for SKILL.md files, for the in-app skill finder.

With a token: GitHub code search (finds SKILL.md anywhere in public repos).
Without: repo search + a root-SKILL.md probe (root-level only). Returns a
unified shape the existing /skills/import-from-url flow can install:
    {"name", "description", "repo", "stars", "url", "source"}
`http` is injectable for tests: (method, url, headers) -> (status, json_dict).
"""
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
        raise SkillSearchError(
            "GitHub rate limit reached — add a GitHub token in Settings for higher limits.")
    if status == 401:
        raise SkillSearchError("GitHub token is invalid — check it in Settings.")
    raise SkillSearchError(f"GitHub search failed ({status}).")


def _code_search(query, token, limit, http):
    q = urllib.parse.quote(f"filename:SKILL.md {query}")
    status, body = http("GET", f"{API}/search/code?q={q}&per_page={limit}", _headers(token))
    if status != 200:
        _fail(status, body)
    out = []
    for it in (body.get("items") or [])[:limit]:
        repo = it.get("repository") or {}
        parts = [p for p in (it.get("path") or "").split("/") if p]
        name = parts[-2] if len(parts) >= 2 else (repo.get("full_name") or "skill")
        out.append({
            "name": name,
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
