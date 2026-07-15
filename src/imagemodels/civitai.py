"""Civitai LoRA search + download-URL/token helpers. `get` is injectable so
search is unit-testable without network."""
_API = "https://civitai.com/api/v1/models"


def download_url_with_token(url: str, token: str) -> str:
    if not token:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}token={token}"


def _flatten(item: dict) -> dict:
    versions = item.get("modelVersions") or []
    v = versions[0] if versions else {}
    files = v.get("files") or []
    f = next((x for x in files if x.get("primary")), files[0] if files else {})
    return {
        "id": item.get("id"),
        "name": item.get("name") or "",
        "base_model": v.get("baseModel") or "",
        "trigger_words": v.get("trainedWords") or [],
        "version_id": v.get("id"),
        "download_url": f.get("downloadUrl") or v.get("downloadUrl") or "",
        "file_name": f.get("name") or "",
        "size_kb": f.get("sizeKB") or 0,
    }


def _default_get(url, params, headers):
    import httpx
    r = httpx.get(url, params=params, headers=headers, timeout=30,
                  follow_redirects=True)
    r.raise_for_status()
    return r.json()


def search(query, *, limit=20, token=None, get=None) -> list:
    get = get or _default_get
    params = {"types": "LORA", "query": query or "", "limit": max(1, min(int(limit), 100))}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    data = get(_API, params, headers) or {}
    return [_flatten(it) for it in (data.get("items") or [])]
