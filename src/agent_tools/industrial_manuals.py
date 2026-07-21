"""Admin-only equipment-manual tools: ingest_equipment_manual (add/list/remove a
manual in the dedicated KB) and search_equipment_manual (page-cited retrieval).
Both handlers NEVER raise — every failure returns {"error": ...}. The ManualStore
is injectable so the tool logic is unit-testable without ChromaDB."""
import json
import os


async def ingest_equipment_manual(content, ctx, *, store=None):
    if store is None:
        from src.industrial.manual_store import get_manual_store
        store = get_manual_store()
    if store is None:
        return {"error": "ingest_equipment_manual: manual store unavailable"}
    try:
        args = json.loads(content) if content and content.strip() else {}
    except (ValueError, TypeError):
        return {"error": "ingest_equipment_manual: arguments must be valid JSON"}
    if not isinstance(args, dict):
        return {"error": "ingest_equipment_manual: arguments must be a JSON object"}

    action = args.get("action", "add")
    try:
        if action == "add":
            path = args.get("path")
            if not isinstance(path, str) or not path:
                return {"error": "ingest_equipment_manual: 'path' (string) is required"}
            title = args.get("title")
            if title is not None and not isinstance(title, str):
                return {"error": "ingest_equipment_manual: 'title' must be a string"}
            if not os.path.isfile(path):
                return {"error": f"ingest_equipment_manual: no such file: {path}"}
            res = store.ingest_file(path, title=title)
            if isinstance(res, dict) and res.get("error"):
                return {"error": f"ingest_equipment_manual: {res['error']}"}
            return {"output": res}
        if action == "list":
            return {"output": {"manuals": store.list_manuals()}}
        if action == "remove":
            manual_id = args.get("manual_id")
            if not isinstance(manual_id, str) or not manual_id:
                return {"error": "ingest_equipment_manual: 'manual_id' (string) is required"}
            return {"output": store.remove_manual(manual_id)}
        return {"error": f"ingest_equipment_manual: unknown action {action!r}"}
    except Exception as e:
        return {"error": f"ingest_equipment_manual: {e}"}


class IngestEquipmentManualTool:
    async def execute(self, content, ctx):
        return await ingest_equipment_manual(content, ctx)


async def search_equipment_manual(content, ctx, *, store=None):
    if store is None:
        from src.industrial.manual_store import get_manual_store
        store = get_manual_store()
    if store is None:
        return {"error": "search_equipment_manual: manual store unavailable"}
    try:
        args = json.loads(content) if content and content.strip() else {}
    except (ValueError, TypeError):
        return {"error": "search_equipment_manual: arguments must be valid JSON"}
    if not isinstance(args, dict):
        return {"error": "search_equipment_manual: arguments must be a JSON object"}

    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return {"error": "search_equipment_manual: 'query' (string) is required"}
    k = args.get("k", 5)
    if isinstance(k, bool) or not isinstance(k, int):
        k = 5
    k = max(1, min(k, 20))
    manual_id = args.get("manual_id")
    if manual_id is not None and not isinstance(manual_id, str):
        return {"error": "search_equipment_manual: 'manual_id' must be a string"}

    try:
        hits = store.search(query, k=k, manual_id=manual_id)
    except Exception as e:
        return {"error": f"search_equipment_manual: {e}"}

    if not hits:
        return {"output": {"citations": "", "results": [],
                           "message": "No matching manual passage found."}}

    lines = []
    results = []
    for h in hits:
        title = h.get("title") or "manual"
        page = h.get("page")
        snippet = (h.get("snippet") or "").strip()
        snippet_short = snippet if len(snippet) <= 300 else snippet[:300] + "…"
        if isinstance(page, int) and page >= 1:
            label = f"{title}, p.{page}"
        else:
            label = f"{title} (excerpt {(h.get('chunk_id') or 0) + 1})"
        lines.append(f'**{label}** — "{snippet_short}"')
        results.append({"title": title, "page": page, "source": h.get("source"),
                        "snippet": snippet, "score": h.get("score")})
    return {"output": {"citations": "\n".join(lines), "results": results}}


class SearchEquipmentManualTool:
    async def execute(self, content, ctx):
        return await search_equipment_manual(content, ctx)
