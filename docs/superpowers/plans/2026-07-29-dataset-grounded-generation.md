# Document-Grounded Synthetic Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Dataset Builder's "Generate (AI)" card so generated training rows can be **grounded** in real source text — either an uploaded document (PDF/text, chunked by page) or the ingested equipment-manual library (passages retrieved by topic) — with each staged row showing its source page while the saved row stays clean.

**Architecture:** A thin, never-raising grounding layer (`src/dataset_tools/ground.py`) sits on top of sub-project 2's `generate_rows`: `chunk_document` turns `(page_no, text)` pairs into labeled chunks; `generate_grounded` walks chunks, calls `generate_rows` per chunk with the chunk as a new `context` argument, tags each staged row with its source, and dedups across chunks by threading the growing accepted list as `generate_rows`'s `existing`. Two adapters build chunks from the manual library (`ManualStore.search`) or an uploaded document (`_pdf_pages`/`_read_document_text`). Two admin routes (`/generate/grounded` JSON, `/generate/upload` multipart) wire it up; the Generate card gains a source selector.

**Tech Stack:** Python 3.14 (stdlib + FastAPI + httpx + python-multipart, already bundled), ES-module browser JS, pytest (`--import-mode=importlib`), `node --check`.

## Global Constraints

- Main app is Python 3.14 and MUST NEVER import torch/peft/bitsandbytes/transformers/gguf/sentencepiece/trl/datasets/accelerate. `ground.py` imports only `generate_rows` at module top; the `manual_store` helpers (`_pdf_pages`, `_read_document_text`, `get_manual_store`) are imported LAZILY inside functions so `import ground` stays light.
- **Backward compatibility:** adding `context=None` to `build_generation_prompt`/`generate_rows` must leave their output byte-identical to today when `context` is None/empty (sub-project 2's tests must stay green).
- **Never-raises:** `chunk_document`, `generate_grounded`, `library_chunks`, `document_chunks` never raise on ANY input (bad pages, non-list chunks, a `model_call` that raises, a store that raises, unreadable file).
- **Never-500:** `POST /api/datasets/generate/grounded` and `POST /api/datasets/generate/upload` return HTTP 200 with a report on every input; failures (no query, no hits, unsupported/corrupt file, no model endpoint) ride the report's `error` field. NO `HTTPException` in either generate route. The upload route wraps its body in try/except and removes the temp file in `finally`.
- **Admin-gated:** both routes ride the existing `setup_dataset_routes()` router `dependencies=[Depends(require_admin)]`. Extend it — no second router.
- **Clean saved rows:** the `source` label rides in the staging report only (`report["rows"][i]["source"]`); it is NEVER baked into the row dict. "Add valid rows" merges the clean `row` values.
- `count` clamped `[1, MAX_GENERATE]` (MAX_GENERATE=200, reuse the sub-project-2 constant); `k` clamped `[1, 20]`.
- Every generated row is validated by `normalize_row` (inside `generate_rows`) before it can be staged valid.
- Frontend HTML-escapes every model/source string via the existing `esc()` before `innerHTML`.
- pytest `--import-mode=importlib`; `node --check` gates `dataset.js`; commit to `dev`; commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; stage only the named files (NEVER `git add -A`; never stage `installer/Output/Assist-Setup.exe`).

## File Structure

- **Modify `src/dataset_tools/generate.py`** — add `context=None` to `build_generation_prompt` (grounding instruction) and `generate_rows` (thread it through). (Task 1)
- **Create `src/dataset_tools/ground.py`** — `chunk_document` + `generate_grounded` (Task 2); `library_chunks` + `document_chunks` adapters + `_default_pdf_pages`/`_default_read_text` (Task 3).
- **Modify `routes/dataset_routes.py`** — `/generate/grounded` + `/generate/upload` (Task 4).
- **Modify `static/index.html` + `static/js/dataset.js`** — source selector + branched `generate()` + source in staging (Task 5).
- **Modify `static/index.html`** — Help sentence (Task 6).
- **Tests:** `tests/test_dataset_generate.py` (extend, Task 1), `tests/test_dataset_ground.py` (Tasks 2+3), `tests/test_dataset_ground_routes.py` (Task 4), `tests/test_dataset_ui.py` (extend, Tasks 5+6).

---

### Task 1: `context` grounding param on the generator

**Files:**
- Modify: `src/dataset_tools/generate.py`
- Test: `tests/test_dataset_generate.py` (extend)

**Interfaces:**
- Produces: `build_generation_prompt(fmt, count, brief, seed_rows=None, context=None)` and `generate_rows(..., context=None)`. When `context` is a non-empty str, the prompt instructs "answerable ONLY from this source text". `context` None/empty → identical to today.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dataset_generate.py`:

```python
def test_prompt_context_grounds_and_is_backward_compatible():
    base_sys, base_usr = build_generation_prompt("text", 3, "b", None)
    # no context (explicit None) == prior behavior
    n_sys, n_usr = build_generation_prompt("text", 3, "b", None, None)
    assert (n_sys, n_usr) == (base_sys, base_usr)
    assert "Source text" not in base_usr
    # with context: grounding instruction in system + the passage in the user message
    g_sys, g_usr = build_generation_prompt("text", 3, "b", None, "PUMP TRIP CODE E12 = overpressure")
    assert "ONLY" in g_sys and "source" in g_sys.lower()
    assert "E12 = overpressure" in g_usr and "Source text" in g_usr


def test_generate_rows_threads_context_into_prompt():
    seen = {}
    async def fake(prompt, system=None):
        seen["prompt"] = prompt
        return '{"text": "a"}'
    import asyncio
    asyncio.run(generate_rows("text", 1, "", model_call=fake, batch_size=1,
                              context="GROUNDING SOURCE XYZ"))
    assert "GROUNDING SOURCE XYZ" in seen["prompt"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_generate.py -k "context" --import-mode=importlib -q`
Expected: FAIL (context param not accepted).

- [ ] **Step 3: Implement**

In `src/dataset_tools/generate.py`, replace the `build_generation_prompt` signature line and body top:

```python
def build_generation_prompt(fmt, count, brief, seed_rows=None, context=None):
    """Build (system, user) messages instructing the model to emit JSONL rows of
    the target shape. When `context` is a non-empty passage, ground generation in
    it. Deterministic, string-only, never raises."""
    fmt = fmt if isinstance(fmt, str) and fmt in _SHAPE_DESC else "text"
    ctx = context.strip() if isinstance(context, str) else ""
    system = (
        "You generate high-quality fine-tuning data for a language model. "
        "Output ONLY JSONL: one JSON object per line, no prose, no markdown, no code fences. "
        f"Each object MUST have {_SHAPE_DESC[fmt]}. Example line: {_SHAPE_EXAMPLE[fmt]}"
    )
    if ctx:
        system += (" Every example MUST be answerable using ONLY the provided source text; "
                   "do NOT introduce facts that are not present in it.")
```

Then, in the same function, insert the source block into `parts` — add it immediately AFTER the `if b.strip():` block (after the "Topic / instructions" append) and BEFORE the `examples = []` line:

```python
    if ctx:
        parts.append("Source text (generate examples ONLY from this):\n" + ctx)
```

Then update `generate_rows` — change its signature and the `build_generation_prompt` call. Signature:

```python
async def generate_rows(fmt, count, brief, *, seed_rows=None, existing=None,
                        model_call, batch_size=10, max_attempts=None, context=None):
```

And the prompt-build call inside the `while` loop — find:

```python
        _system, _user = build_generation_prompt(fmt, min(batch_size, count - accepted), brief, seed_rows)
```

replace with:

```python
        _system, _user = build_generation_prompt(fmt, min(batch_size, count - accepted), brief, seed_rows, context)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_dataset_generate.py --import-mode=importlib -q`
Expected: PASS (all prior tests + 2 new; the backward-compat test proves no regression).

- [ ] **Step 5: Commit**

```bash
git add src/dataset_tools/generate.py tests/test_dataset_generate.py
git commit -m "feat(dataset-tools): optional context grounding param on the generator

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `ground.py` — chunk_document + generate_grounded

**Files:**
- Create: `src/dataset_tools/ground.py`
- Test: `tests/test_dataset_ground.py`

**Interfaces:**
- Consumes: `generate_rows(..., context=...)` (Task 1).
- Produces: `chunk_document(pages, *, max_chars=2000) -> [{"source","text"}]`; `async generate_grounded(chunks, fmt, count, *, model_call, existing=None, brief="", per_chunk=4, batch_size=4, max_chunks=200) -> dict` (report with `rows:[{row,valid,error,duplicate,source}], valid, invalid, duplicates, requested, produced, chunks_used, error?`). Both never raise.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_ground.py`:

```python
import asyncio
from src.dataset_tools.ground import chunk_document, generate_grounded


def _run(coro):
    return asyncio.run(coro)


def test_chunk_document_labels_and_splits():
    chunks = chunk_document([(1, "hello"), (2, ""), (3, "x" * 4500)], max_chars=2000)
    assert chunks[0] == {"source": "p.1", "text": "hello"}          # page 1
    assert all(c["source"] != "p.2" for c in chunks)                # blank page skipped
    p3 = [c for c in chunks if c["source"] == "p.3"]
    assert len(p3) == 3 and all(len(c["text"]) <= 2000 for c in p3)  # long page split


def test_chunk_document_never_raises_on_garbage():
    assert chunk_document(None) == []
    assert chunk_document([("bad",), 42, (1, 5)]) == []  # non-str text / bad tuples skipped


def test_generate_grounded_tags_source_and_stops_at_count():
    async def fake(prompt, system=None):
        # echo a row that embeds which source text it saw so we can assert grounding
        return '{"text": "row"}'
    chunks = [{"source": "p.1", "text": "AAA"}, {"source": "p.2", "text": "BBB"}]
    rep = _run(generate_grounded(chunks, "text", 1, model_call=fake, per_chunk=4, batch_size=4))
    assert rep["produced"] == 1 and rep["requested"] == 1
    assert rep["rows"][0]["source"] == "p.1"          # tagged
    assert rep["chunks_used"] == 1                     # stopped after count reached


def test_generate_grounded_dedups_across_chunks():
    async def fake(prompt, system=None):
        return '{"text": "same"}'                      # every chunk yields the same row
    chunks = [{"source": "p.1", "text": "A"}, {"source": "p.2", "text": "B"}]
    rep = _run(generate_grounded(chunks, "text", 5, model_call=fake, per_chunk=1, batch_size=1))
    assert rep["produced"] == 1 and rep["duplicates"] >= 1  # 2nd chunk's row is a dup


def test_generate_grounded_model_error_and_empty_never_raise():
    async def boom(prompt, system=None):
        raise RuntimeError("no endpoint")
    rep = _run(generate_grounded([{"source": "p.1", "text": "A"}], "text", 3, model_call=boom))
    assert rep["produced"] == 0 and "error" in rep
    rep2 = _run(generate_grounded([], "text", 3, model_call=boom))
    assert rep2["produced"] == 0 and rep2["chunks_used"] == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_ground.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.dataset_tools.ground'`).

- [ ] **Step 3: Implement**

Create `src/dataset_tools/ground.py`:

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_dataset_ground.py --import-mode=importlib -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dataset_tools/ground.py tests/test_dataset_ground.py
git commit -m "feat(dataset-tools): grounding core (chunk_document + generate_grounded)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Source adapters — library + document

**Files:**
- Modify: `src/dataset_tools/ground.py`
- Test: `tests/test_dataset_ground.py` (extend)

**Interfaces:**
- Consumes: `chunk_document` (Task 2); `ManualStore.search(query, k) -> [{"title","page","snippet",...}]` and the module-level `_pdf_pages`/`_read_document_text` (imported lazily).
- Produces: `library_chunks(store, query, *, k=8) -> [{"source","text"}]`; `document_chunks(path, ext, *, pdf_pages=None, read_text=None, filename=None, max_chars=2000) -> [{"source","text"}]`. Both never raise; return `[]` on unusable input.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dataset_ground.py`:

```python
from src.dataset_tools.ground import library_chunks, document_chunks


class _FakeStore:
    def __init__(self, hits): self._hits = hits
    def search(self, query, k=8): return self._hits


def test_library_chunks_labels_title_and_page():
    store = _FakeStore([
        {"title": "VFD Manual", "page": 42, "snippet": "E12 = overpressure trip"},
        {"title": "VFD Manual", "page": None, "snippet": "general notes"},
        {"title": "X", "page": 1, "snippet": "   "},           # blank -> skipped
    ])
    chunks = library_chunks(store, "overpressure", k=8)
    assert chunks[0] == {"source": "VFD Manual, p.42", "text": "E12 = overpressure trip"}
    assert chunks[1]["source"] == "VFD Manual"                 # no page
    assert all(c["text"].strip() for c in chunks) and len(chunks) == 2


def test_library_chunks_never_raises():
    assert library_chunks(None, "q") == []
    assert library_chunks(_FakeStore("not-a-list"), "q") == []
    assert library_chunks(_FakeStore([]), "") == []           # empty query


def test_document_chunks_pdf_and_text_and_unsupported():
    def fake_pdf(path):
        return [(1, "page one text"), (2, "page two text")]
    pdf = document_chunks("f.pdf", ".pdf", pdf_pages=fake_pdf, filename="Guide.pdf")
    assert pdf[0] == {"source": "Guide.pdf p.1", "text": "page one text"}
    def fake_read(path, ext):
        return "plain body"
    txt = document_chunks("f.txt", ".txt", read_text=fake_read, filename="notes.txt")
    assert txt[0]["source"] == "notes.txt p.1" and txt[0]["text"] == "plain body"
    def none_read(path, ext):
        return None
    assert document_chunks("f.xyz", ".xyz", read_text=none_read) == []   # unsupported
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_ground.py -k "library or document_chunks" --import-mode=importlib -q`
Expected: FAIL (adapters not defined).

- [ ] **Step 3: Implement**

Append to `src/dataset_tools/ground.py`:

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_dataset_ground.py --import-mode=importlib -q`
Expected: PASS (5 from Task 2 + 3 new = 8).

- [ ] **Step 5: Commit**

```bash
git add src/dataset_tools/ground.py tests/test_dataset_ground.py
git commit -m "feat(dataset-tools): grounding adapters (manual library + uploaded document)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Admin grounded + upload routes

**Files:**
- Modify: `routes/dataset_routes.py`
- Test: `tests/test_dataset_ground_routes.py`

**Interfaces:**
- Consumes: `generate_grounded`, `library_chunks`, `document_chunks` (Tasks 2-3); `_default_model_call`, `MAX_GENERATE` (already in the module); `get_manual_store` (lazy).
- Produces: `POST /api/datasets/generate/grounded` (JSON) and `POST /api/datasets/generate/upload` (multipart) on the existing admin router. Both return HTTP 200 + a report; never 500.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_ground_routes.py`:

```python
import io
import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.dataset_routes as dr


def _client(monkeypatch, *, chunks=None, fake_call=None):
    monkeypatch.setattr(dr, "require_admin", lambda: None)
    async def _default(prompt, system=None, owner=None):
        return '{"text": "a"}'
    monkeypatch.setattr(dr, "_default_model_call", fake_call or _default)
    if chunks is not None:
        monkeypatch.setattr(dr, "library_chunks", lambda store, query, k=8: chunks)
        monkeypatch.setattr(dr, "document_chunks",
                            lambda path, ext, pdf_pages=None, read_text=None, filename=None, max_chars=2000: chunks)
    app = FastAPI(); app.include_router(dr.setup_dataset_routes())
    return TestClient(app)


def test_grounded_happy(monkeypatch):
    c = _client(monkeypatch, chunks=[{"source": "M, p.1", "text": "AAA"}])
    r = c.post("/api/datasets/generate/grounded", json={"format": "text", "count": 1, "query": "x"})
    assert r.status_code == 200
    j = r.json()
    assert j["produced"] == 1 and j["rows"][0]["source"] == "M, p.1"


def test_grounded_requires_query(monkeypatch):
    c = _client(monkeypatch, chunks=[{"source": "M, p.1", "text": "AAA"}])
    r = c.post("/api/datasets/generate/grounded", json={"format": "text", "count": 1, "query": "  "})
    assert r.status_code == 200 and "error" in r.json() and r.json()["produced"] == 0


def test_grounded_no_hits(monkeypatch):
    c = _client(monkeypatch, chunks=[])
    r = c.post("/api/datasets/generate/grounded", json={"format": "text", "count": 1, "query": "zzz"})
    assert r.status_code == 200 and "error" in r.json()


def test_upload_happy(monkeypatch):
    c = _client(monkeypatch, chunks=[{"source": "Guide.pdf p.1", "text": "AAA"}])
    files = {"file": ("Guide.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")}
    data = {"format": "text", "count": "1", "brief": "", "existing": json.dumps([])}
    r = c.post("/api/datasets/generate/upload", files=files, data=data)
    assert r.status_code == 200 and r.json()["produced"] == 1
    assert r.json()["rows"][0]["source"] == "Guide.pdf p.1"


def test_upload_unextractable(monkeypatch):
    c = _client(monkeypatch, chunks=[])   # extractor yields nothing
    files = {"file": ("x.bin", io.BytesIO(b"\x00\x01"), "application/octet-stream")}
    r = c.post("/api/datasets/generate/upload", files=files, data={"format": "text", "count": "1"})
    assert r.status_code == 200 and "error" in r.json() and r.json()["produced"] == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_ground_routes.py --import-mode=importlib -q`
Expected: FAIL (routes not defined).

- [ ] **Step 3: Implement**

In `routes/dataset_routes.py`, add to the imports at the top (after the existing `from src.dataset_tools.generate import generate_rows`):

```python
from src.dataset_tools.ground import generate_grounded, library_chunks, document_chunks
```

And change the FastAPI import line at the top to also bring in the multipart helpers:

```python
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
```

Then add both endpoints inside `setup_dataset_routes()`, immediately AFTER the existing `@router.post("/generate")` block:

```python
    @router.post("/generate/grounded")
    async def generate_grounded_route(body: dict = Body(...)):
        fmt = body.get("format") or body.get("fmt") or "text"
        try:
            count = int(body.get("count", 10))
        except (TypeError, ValueError, OverflowError):
            count = 10
        count = max(1, min(count, MAX_GENERATE))
        query = body.get("query")
        if not isinstance(query, str) or not query.strip():
            return {"rows": [], "valid": 0, "invalid": 0, "duplicates": 0,
                    "requested": count, "produced": 0, "chunks_used": 0,
                    "error": "a topic query is required"}
        try:
            k = int(body.get("k", 8))
        except (TypeError, ValueError, OverflowError):
            k = 8
        from src.industrial.manual_store import get_manual_store
        chunks = library_chunks(get_manual_store(), query, k=k)
        if not chunks:
            return {"rows": [], "valid": 0, "invalid": 0, "duplicates": 0,
                    "requested": count, "produced": 0, "chunks_used": 0,
                    "error": "no matching manual passages found"}
        return await generate_grounded(chunks, fmt, count, model_call=_default_model_call,
                                       existing=body.get("existing"), brief=body.get("brief", ""))

    @router.post("/generate/upload")
    async def generate_upload_route(file: UploadFile = File(...),
                                    format: str = Form("text"), count: str = Form("10"),
                                    brief: str = Form(""), existing: str = Form("")):
        import json as _json
        import os as _os
        import tempfile as _tempfile
        try:
            cnt = int(count)
        except (TypeError, ValueError, OverflowError):
            cnt = 10
        cnt = max(1, min(cnt, MAX_GENERATE))
        try:
            ex = _json.loads(existing) if existing else None
        except (ValueError, TypeError):
            ex = None
        empty = {"rows": [], "valid": 0, "invalid": 0, "duplicates": 0,
                 "requested": cnt, "produced": 0, "chunks_used": 0}
        fname = file.filename or "document"
        ext = _os.path.splitext(fname)[1].lower()
        tmp = None
        try:
            data = await file.read()
            fd, tmp = _tempfile.mkstemp(suffix=ext or ".bin")
            with _os.fdopen(fd, "wb") as fh:
                fh.write(data)
            chunks = document_chunks(tmp, ext, filename=fname)
            if not chunks:
                return {**empty, "error": "could not extract text from the uploaded file"}
            return await generate_grounded(chunks, format, cnt, model_call=_default_model_call,
                                           existing=ex, brief=brief)
        except Exception as e:  # noqa: BLE001 -- never 500
            return {**empty, "error": f"upload failed: {e}"}
        finally:
            if tmp:
                try:
                    _os.remove(tmp)
                except OSError:
                    pass
```

Note: `generate_grounded` never raises, so the grounded route needs no try/except; the upload route wraps its file I/O because extraction/temp-file handling CAN raise, and must still return 200 + error.

- [ ] **Step 4: Run the tests + import smoke**

Run: `python -m pytest tests/test_dataset_ground_routes.py --import-mode=importlib -q`
Expected: PASS (5 passed).
Run: `python -c "import app; print('OK')"`
Expected: `OK` (confirms `python-multipart` is available for Form/File — it is, the app already has upload routes).

- [ ] **Step 5: Commit**

```bash
git add routes/dataset_routes.py tests/test_dataset_ground_routes.py
git commit -m "feat(dataset-tools): admin grounded + upload generation routes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Frontend source selector + branched generate

**Files:**
- Modify: `static/index.html`
- Modify: `static/js/dataset.js`
- Test: `tests/test_dataset_ui.py` (extend)

**Interfaces:**
- Consumes: `/api/datasets/generate/grounded`, `/api/datasets/generate/upload` (Task 4); the existing `generate()`/`renderStaging()`/`staged`/`$`/`esc`/`api` in `dataset.js`.
- Produces: `#dataset-gen-source` selector + `#dataset-gen-file` + `#dataset-gen-query` in the Generate card; `generate()` branches by source; `renderStaging` shows each row's `source`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dataset_ui.py`:

```python
def test_index_has_grounding_source_controls():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in ('id="dataset-gen-source"', 'id="dataset-gen-file"', 'id="dataset-gen-query"'):
        assert el in html, f"{el} missing from index.html"


def test_dataset_js_branches_grounded_sources():
    src = (ROOT / "static" / "js" / "dataset.js").read_text(encoding="utf-8")
    for s in ('/api/datasets/generate/upload', '/api/datasets/generate/grounded',
              'dataset-gen-source', 'c.source', 'FormData'):
        assert s in src, f"{s} missing from dataset.js"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_ui.py -k "grounding or branches" --import-mode=importlib -q`
Expected: FAIL.

- [ ] **Step 3: Add the source controls to `static/index.html`**

In the Generate card (the `admin-card` containing `#dataset-generate`), insert these three controls immediately AFTER the `#dataset-gen-count` `<label>...</label>` line and BEFORE the `<button id="dataset-generate"...>` line:

```html
        <label>Source
          <select id="dataset-gen-source">
            <option value="none">None (free)</option>
            <option value="upload">Uploaded document</option>
            <option value="library">Manual library</option>
          </select>
        </label>
        <input id="dataset-gen-file" type="file" accept=".pdf,.txt,.md,.docx" style="display:none">
        <input id="dataset-gen-query" placeholder="topic to pull from your manuals" style="width:100%;display:none">
```

- [ ] **Step 4: Update `static/js/dataset.js`**

Replace the entire `generate()` function (the one that POSTs to `/api/datasets/generate`) with this branched version:

```javascript
async function generate() {
  const src = $('dataset-gen-source') ? $('dataset-gen-source').value : 'none';
  const brief = $('dataset-gen-brief') ? $('dataset-gen-brief').value.trim() : '';
  const count = $('dataset-gen-count') ? (parseInt($('dataset-gen-count').value, 10) || 10) : 10;
  const fmt = $('dataset-format') ? $('dataset-format').value : 'text';
  const btn = $('dataset-generate');
  const out = $('dataset-gen-staging');
  if (btn) { btn.disabled = true; btn.textContent = 'Generating…'; }
  if (out) out.innerHTML = '<div style="opacity:0.6">Generating… this can take a minute.</div>';
  try {
    let rep;
    if (src === 'upload') {
      const f = $('dataset-gen-file');
      if (!f || !f.files || !f.files[0]) { throw new Error('Choose a file to upload.'); }
      const fd = new FormData();
      fd.append('file', f.files[0]);
      fd.append('format', fmt); fd.append('count', String(count));
      fd.append('brief', brief); fd.append('existing', JSON.stringify(rows));
      rep = await api('/api/datasets/generate/upload', { method: 'POST', body: fd });
    } else if (src === 'library') {
      const q = $('dataset-gen-query') ? $('dataset-gen-query').value.trim() : '';
      rep = await api('/api/datasets/generate/grounded', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format: fmt, count: count, brief: brief, query: q, existing: rows }),
      });
    } else {
      rep = await api('/api/datasets/generate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format: fmt, count: count, brief: brief,
                               existing: rows, seed_rows: rows.slice(0, 3) }),
      });
    }
    staged = rep.rows || [];
    renderStaging(rep);
  } catch (e) {
    if (out) out.textContent = 'Generate failed: ' + e.message;
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Generate'; }
  }
}
```

Replace the `renderStaging()` function's per-row map and summary line to show the source. Replace the whole `renderStaging` function with:

```javascript
function renderStaging(rep) {
  const out = $('dataset-gen-staging'); if (!out) return;
  if (rep.error && !(rep.rows || []).length) { out.innerHTML = '<div style="color:#c00">' + esc(rep.error) + '</div>'; return; }
  const items = (staged || []).map(function (c) {
    const mark = c.valid ? (c.duplicate ? '⚠' : '✓') : '✗';
    const srcnote = c.source ? ' <span style="opacity:0.6">[' + esc(c.source) + ']</span>' : '';
    const note = c.duplicate ? ' (duplicate)' : (c.error ? ' — ' + esc(c.error) : '');
    return '<div>' + mark + ' ' + esc(JSON.stringify(c.row)).slice(0, 140) + srcnote + note + '</div>';
  }).join('');
  const meta = (rep.chunks_used != null) ? (rep.chunks_used + ' chunk(s)')
             : (rep.attempts != null ? rep.attempts + ' attempt(s)' : '');
  out.innerHTML = '<div>produced ' + rep.produced + ' of ' + rep.requested +
    (meta ? ' · ' + meta : '') + (rep.error ? ' · <span style="color:#c00">' + esc(rep.error) + '</span>' : '') +
    '</div>' + items + '<button class="btn" id="dataset-gen-add">Add valid rows</button>';
  const add = $('dataset-gen-add');
  if (add) add.addEventListener('click', addGenerated);
}
```

Wire the source selector's show/hide in `init()` — add this alongside the existing `#dataset-generate` wiring (before the `Modals.register(...)` line):

```javascript
  const srcSel = $('dataset-gen-source');
  if (srcSel) srcSel.addEventListener('change', function () {
    const v = srcSel.value;
    const f = $('dataset-gen-file'); if (f) f.style.display = (v === 'upload') ? '' : 'none';
    const q = $('dataset-gen-query'); if (q) q.style.display = (v === 'library') ? '' : 'none';
  });
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_dataset_ui.py --import-mode=importlib -q`
Expected: PASS (existing UI tests + 2 new; `test_dataset_js_syntax`'s `node --check` confirms `dataset.js` parses).

- [ ] **Step 6: Commit**

```bash
git add static/index.html static/js/dataset.js tests/test_dataset_ui.py
git commit -m "feat(dataset-tools): grounding source selector + branched generate + source in staging

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Finalize — Help section + full suite

**Files:**
- Modify: `static/index.html` (Datasets Help section)
- Test: `tests/test_dataset_ui.py` (extend); run the whole dataset suite + `import app`

- [ ] **Step 1: Add the Help entry test**

Append to `tests/test_dataset_ui.py`:

```python
def test_help_mentions_grounded_generation():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert "grounded in a document or your manuals" in html
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_ui.py::test_help_mentions_grounded_generation --import-mode=importlib -q`
Expected: FAIL.

- [ ] **Step 3: Add the sentence to the Datasets Help section**

In `static/index.html`, find the Help-manual `Datasets (AI Studio)` `<details>` block. It already contains a "Generate with AI." paragraph (from sub-project 2). Immediately AFTER that `<p><b>Generate with AI.</b> ...</p>` paragraph, add this paragraph inside the same `<details>`:

```html
            <p><b>Ground it in your documents.</b> Under the Generate <b>Source</b> selector, pick <b>Uploaded document</b> to upload a PDF or text file, or <b>Manual library</b> to pull from your ingested equipment manuals by topic — rows are then <b>grounded in a document or your manuals</b>, and each staged row shows the page it came from so you can check it. Saved rows stay clean (no citation text).</p>
```

- [ ] **Step 4: Run the full dataset suite + import smoke**

Run: `python -m pytest tests/test_dataset_generate.py tests/test_dataset_ground.py tests/test_dataset_generate_routes.py tests/test_dataset_ground_routes.py tests/test_dataset_normalize.py tests/test_dataset_validate.py tests/test_dataset_store.py tests/test_dataset_routes.py tests/test_dataset_core_js.py tests/test_dataset_ui.py --import-mode=importlib -q`
Expected: PASS (all green).
Run: `python -c "import app; print('OK')"`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add static/index.html tests/test_dataset_ui.py
git commit -m "docs(dataset-tools): Help note for document-grounded generation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Backward compatibility (Task 1):** `context=None` MUST leave `build_generation_prompt`/`generate_rows` byte-identical to sub-project 2. The regression test asserts the no-context prompt is unchanged — keep it green.
- **Never-raises (Tasks 2-3):** `chunk_document`, `generate_grounded`, `library_chunks`, `document_chunks` return a value on ANY input. `generate_grounded` reuses `generate_rows`'s dedup by passing the growing accepted list as `existing`.
- **Never-500 (Task 4):** both routes return 200 + report; the grounded route needs no try/except (generate_grounded never raises); the upload route wraps file I/O and cleans the temp file in `finally`. The route tests monkeypatch `require_admin` (sibling convention) — the gate is proven by the router construction, verify by reading.
- **Clean rows:** `source` rides in the report only; `addGenerated()` merges the clean `row`. Never bake the citation into the row.
- **No heavy-dep leak / no rebuild:** `ground.py` imports `generate_rows` at top and `manual_store` helpers lazily; the routes add `python-multipart` (already bundled). Frontend + Python only — no frozen rebuild needed to test; the feature reaches users on the next installer rebuild.
- **Manual GUI verification owed** by the user: ingest a manual (or upload a PDF) → open Dataset → Generate with a grounding source → confirm rows match the cited pages → Add valid rows → Save.
- **Scope:** upload (PDF/text) + library retrieval, clean rows, source in staging only. NO citation-baking, NO OCR, NO per-manual filter UI, NO answerability verification (all deferred).
```
