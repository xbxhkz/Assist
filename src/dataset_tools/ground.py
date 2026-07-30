"""Document-grounded synthetic generation: chunk source text and generate training
rows grounded in each chunk (reusing generate_rows). Pure + injectable model_call.
Never raises. The saved row stays clean — the source label rides in the report only."""
from src.dataset_tools.generate import generate_rows


def chunk_document(pages, *, max_chars=2000):
    """`pages`: iterable of (page_no, page_text). Emit [{"source","text"}] — one or
    more chunks per non-blank page (split to <= max_chars), labeled 'p.<n>'.
    Never raises."""
    out = []
    try:
        step = max(200, int(max_chars))
    except Exception:  # noqa: BLE001
        step = 2000
    try:
        for item in pages:
            try:
                page_no, text = item
            except Exception:  # noqa: BLE001
                continue
            text = (text if isinstance(text, str) else "").strip()
            if not text:
                continue
            label = f"p.{page_no}" if isinstance(page_no, int) and page_no >= 1 else "source"
            for i in range(0, len(text), step):
                piece = text[i:i + step].strip()
                if piece:
                    out.append({"source": label, "text": piece})
    except Exception:  # noqa: BLE001
        pass
    return out


async def generate_grounded(chunks, fmt, count, *, model_call, existing=None,
                            brief="", per_chunk=4, batch_size=4, max_chunks=200):
    """Walk chunks; per chunk call generate_rows with the chunk as `context`, tag
    each candidate row with its source, and dedup across chunks by threading the
    growing accepted list as generate_rows's `existing`. Stop at `count` valid rows
    or when chunks are exhausted. Never raises."""
    try:
        count = max(0, int(count))
    except Exception:  # noqa: BLE001
        count = 0
    try:
        max_chunks = max(0, int(max_chunks))
    except Exception:  # noqa: BLE001
        max_chunks = 200
    chunk_list = chunks if isinstance(chunks, list) else []
    acc = [r for r in (existing if isinstance(existing, list) else []) if isinstance(r, dict)]
    staged, produced, used, err = [], 0, 0, None
    for chunk in chunk_list[:max_chunks]:
        if produced >= count:
            break
        used += 1
        try:
            ctext = chunk.get("text") if isinstance(chunk, dict) else None
            source = chunk.get("source") if isinstance(chunk, dict) else None
            rep = await generate_rows(fmt, min(per_chunk, count - produced), brief,
                                      existing=acc, model_call=model_call,
                                      batch_size=batch_size, context=ctext)
        except Exception as e:  # noqa: BLE001 -- airtight never-raises
            err = f"grounded generation failed: {e}"
            break
        for c in (rep.get("rows") if isinstance(rep, dict) else []) or []:
            item = dict(c) if isinstance(c, dict) else {"row": None, "valid": False,
                                                        "error": "bad candidate", "duplicate": False}
            item["source"] = source
            staged.append(item)
            if item.get("valid") and not item.get("duplicate"):
                acc.append(item.get("row"))
                produced += 1
        if isinstance(rep, dict) and rep.get("error"):
            err = rep["error"]
            break
    report = {
        "rows": staged,
        "valid": sum(1 for c in staged if c.get("valid") and not c.get("duplicate")),
        "invalid": sum(1 for c in staged if not c.get("valid")),
        "duplicates": sum(1 for c in staged if c.get("duplicate")),
        "requested": count,
        "produced": produced,
        "chunks_used": used,
    }
    if err:
        report["error"] = err
    return report


def _default_pdf_pages(path):
    from src.industrial.manual_store import _pdf_pages
    return _pdf_pages(path)


def _default_read_text(path, ext):
    from src.industrial.manual_store import _read_document_text
    return _read_document_text(path, ext)


def library_chunks(store, query, *, k=8):
    """Retrieve top-k manual passages for `query` -> grounding chunks labeled
    '<title>, p.<page>'. Never raises; [] on any failure."""
    try:
        if store is None or not isinstance(query, str) or not query.strip():
            return []
        try:
            k = max(1, min(int(k), 20))
        except Exception:  # noqa: BLE001
            k = 8
        hits = store.search(query, k=k)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for h in (hits if isinstance(hits, list) else []):
        if not isinstance(h, dict):
            continue
        text = h.get("snippet")
        if not isinstance(text, str) or not text.strip():
            continue
        title = h.get("title") or "manual"
        page = h.get("page")
        label = f"{title}, p.{page}" if isinstance(page, int) and page >= 1 else str(title)
        out.append({"source": label, "text": text.strip()})
    return out


def document_chunks(path, ext, *, pdf_pages=None, read_text=None, filename=None, max_chars=2000):
    """Extract an uploaded document into grounding chunks (PDF via pdf_pages ->
    (page,text); else read_text -> str), labeled '<filename> p.<n>'. Never raises;
    [] on unsupported/unreadable input."""
    name = filename or "document"
    try:
        e = (ext or "").lower()
        if e == ".pdf":
            pages = list((pdf_pages or _default_pdf_pages)(path))
        else:
            text = (read_text or _default_read_text)(path, e)
            if not isinstance(text, str) or not text.strip():
                return []
            pages = [(1, text)]
    except Exception:  # noqa: BLE001
        return []
    chunks = chunk_document(pages, max_chars=max_chars)
    for c in chunks:
        c["source"] = f"{name} {c['source']}"
    return chunks
