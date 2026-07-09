"""Global file search: Windows Search index first, bounded os.walk fallback.
Metadata only — reading a hit still goes through the confined read tool."""
import fnmatch
import os


def _meta(path):
    try:
        st = os.stat(path)
        return {"path": path, "size": st.st_size, "modified": int(st.st_mtime)}
    except OSError:
        return None


def search(query, *, roots, ext=None, all_drives=False, max_results=200,
           searcher=None, walker=os.walk, is_sensitive=None, max_scan=50000):
    if is_sensitive is None:
        from src.tool_execution import _is_sensitive_path as is_sensitive
    q = (query or "").strip()
    hits = []

    # Primary: Windows Search index.
    if searcher is None:
        searcher = _default_searcher
    try:
        paths = searcher(q, ext, max_results)
    except Exception:
        paths = None
    if paths is not None:
        for p in paths:
            if is_sensitive(p):
                continue
            m = _meta(p)
            if m:
                hits.append(m)
            if len(hits) >= max_results:
                break
        return hits

    # Fallback: bounded walk.
    ql = q.lower()
    extl = ("." + ext.lower().lstrip(".")) if ext else None
    scanned = 0
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dp, dns, fns in walker(root):
            dns[:] = [d for d in dns if not is_sensitive(os.path.join(dp, d))]
            for fn in fns:
                scanned += 1
                matched = ((not ql or ql in fn.lower() or fnmatch.fnmatch(fn.lower(), ql))
                           and (not extl or fn.lower().endswith(extl)))
                if matched:
                    full = os.path.join(dp, fn)
                    if not is_sensitive(full):
                        m = _meta(full)
                        if m:
                            hits.append(m)
                if len(hits) >= max_results or scanned >= max_scan:
                    return hits
    return hits


def _default_searcher(query, ext, max_results):  # pragma: no cover (Windows-only)
    """Query the Windows Search index via ADO. Returns None when unavailable
    so the caller falls back to walking. PLAN-TIME: pin adodbapi vs a minimal
    comtypes call and bundle-verify; until then this returns None (walk-only)."""
    return None
