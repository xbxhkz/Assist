"""Live Hugging Face GGUF catalog (Phase 3b).

Search GGUF model repos and list a repo's .gguf files. All network access is
behind an injectable `get_json(url, headers)` so parsing/URL/token logic is
unit-testable without network. No Cookbook imports.
"""
import os
from urllib.parse import quote

_HF = "https://huggingface.co"
_ALLOWED_SORT = {"downloads", "likes", "lastModified", "trendingScore"}


def _hf_token() -> str:
    return (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()


def _hf_headers() -> dict:
    tok = _hf_token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _default_get_json(url: str, headers: dict):
    import httpx
    r = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
    r.raise_for_status()
    return r.json()


def search_gguf_models(query: str = "", sort: str = "downloads",
                       limit: int = 30, get_json=None) -> list:
    """Return GGUF model repos matching `query`, most-downloaded first."""
    get_json = get_json or _default_get_json
    if sort not in _ALLOWED_SORT:
        sort = "downloads"
    url = (f"{_HF}/api/models?filter=gguf&sort={sort}&direction=-1"
           f"&limit={int(limit)}&search={quote(query or '')}")
    try:
        data = get_json(url, _hf_headers())
    except Exception:
        return []
    out = []
    for m in data or []:
        repo = m.get("id") or m.get("modelId")
        if repo:
            out.append({"repo": repo,
                        "downloads": int(m.get("downloads") or 0),
                        "likes": int(m.get("likes") or 0)})
    return out


def list_repo_gguf_files(repo: str, get_json=None) -> list:
    """Return the .gguf files in `repo` as [{filename, size, url}]."""
    get_json = get_json or _default_get_json
    url = f"{_HF}/api/models/{repo}/tree/main?recursive=1"
    try:
        data = get_json(url, _hf_headers())
    except Exception:
        return []
    out = []
    for e in data or []:
        path = e.get("path") or ""
        if e.get("type") == "file" and path.lower().endswith(".gguf"):
            out.append({
                "filename": os.path.basename(path),
                "size": int(e.get("size") or 0),
                "url": f"{_HF}/{repo}/resolve/main/{path}",
            })
    return out
