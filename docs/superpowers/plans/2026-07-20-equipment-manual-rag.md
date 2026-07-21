# Equipment-Manual RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated, admin-managed equipment-manual knowledge base plus two admin-only agent tools — `ingest_equipment_manual` (add/list/remove) and `search_equipment_manual` (returns page-cited passages) — so the agent can quote a manual while diagnosing.

**Architecture:** A focused `ManualStore` class owns an isolated ChromaDB collection base `equipment_manuals`, built via the existing `src/embedding_lanes.build_embedding_lanes` (offline FastEmbed + optional custom lane, hybrid `query_lanes`+`dedupe_results`). PDFs are extracted per-page (via already-bundled `pypdf`) so chunks carry a page number; each page's text is chunked with the existing sentence-aware splitter (promoted to a module-level function). Two thin, never-raises agent tools sit over the store and are registered like the industrial siblings.

**Tech Stack:** Python 3.14, ChromaDB, FastEmbed (offline, bundled), pypdf (bundled), pytest.

## Global Constraints

- **No new dependencies** and **no `Assist.spec` change** — `pypdf` and FastEmbed are already bundled/collected.
- **Dedicated collection base** `equipment_manuals` via `build_embedding_lanes("equipment_manuals")` — isolated from the personal-docs `odysseus_rag`.
- **`manual_id = "man_" + sha256(abspath(source))[:12]`** (keyed on the absolute source path → re-ingest is idempotent). **`doc_id = "man_" + sha256(f"{manual_id}\x00{page}\x00{chunk_id}")[:16]`**.
- **Chroma metadata cannot store `None`** → a non-paginated chunk stores `page = 0` (sentinel). On read, `page >= 1` → "p.N"; else → "excerpt {chunk_id+1}".
- **Both tools admin-only** — in `NON_ADMIN_BLOCKED_TOOLS`. `search_equipment_manual` is read-only → also in `PLAN_MODE_READONLY_TOOLS`. `ingest_equipment_manual` mutates the KB → **NOT** in `PLAN_MODE_READONLY_TOOLS`; add it to `_PLAN_MODE_KNOWN_MUTATORS` (fail-closed backstop).
- **Handlers never raise** — non-JSON/non-object content, missing/wrong-shape arg (isinstance-guarded; reject `bool` for the int `k`), unknown action, unsupported/unreadable file, empty extraction, store unavailable, or an inner raise → each returns `{"error": …}`.
- **Tool handler contract:** `SomeTool().execute(content: str, ctx: dict) -> dict`; `content` is JSON; `ctx` carries `owner`; return `{"output": …}` or `{"error": …}`.
- **Domain:** both tools join the existing `"desktop"` set in `agent_loop._DOMAIN_TOOL_MAP` (co-located with `diagnose_equipment`, the primary pairing). No new domain key (a new key would require a matching `_DOMAIN_RULES` entry).
- pytest `--import-mode=importlib`. Commit directly to `dev`. Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Stage only the files each task names — **never** `git add -A`; never stage `installer/Output/Assist-Setup.exe`, `assistappicon.png`, or `assistlogo.png` (pre-existing dirty artifacts).
- ~220 unrelated pre-existing test failures exist elsewhere — run only the test files each task names.

---

### Task 1: `ManualStore` + module-level chunker

**Files:**
- Modify: `src/rag_vector.py` (promote `_split_into_chunks` to a reusable module-level function)
- Create: `src/industrial/manual_store.py`
- Test: `tests/test_manual_store.py`

**Interfaces:**
- Consumes: `src.embedding_lanes.build_embedding_lanes`, `query_lanes`, `dedupe_results`; `src.personal_docs.read_text_file`, `extract_office_text`; `src.markitdown_runtime.MARKITDOWN_EXTS`; `pypdf.PdfReader`.
- Produces:
  - `src.rag_vector.split_into_chunks(text, chunk_size=1000, overlap=200) -> list[str]`
  - `class ManualStore(base_name="equipment_manuals", lanes=None)` with: `healthy` (property), `ingest_file(path, title=None, *, extract_pages=None) -> dict`, `list_manuals() -> list[dict]`, `remove_manual(manual_id) -> dict`, `search(query, k=5, manual_id=None) -> list[dict]`.
  - `src.industrial.manual_store.get_manual_store() -> ManualStore | None`

- [ ] **Step 1: Write the failing test for the module-level chunker**

Create `tests/test_manual_store.py` with this first test:

```python
def test_split_into_chunks_module_level_splits_long_text():
    from src.rag_vector import split_into_chunks
    text = "This is a sentence. " * 200  # ~4000 chars
    chunks = split_into_chunks(text, chunk_size=1000, overlap=200)
    assert len(chunks) > 1
    assert all(len(c) <= 1200 for c in chunks)  # chunk_size + a little slack
    assert split_into_chunks("") == []
    assert split_into_chunks("short") == ["short"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_manual_store.py::test_split_into_chunks_module_level_splits_long_text --import-mode=importlib -q`
Expected: FAIL (`ImportError: cannot import name 'split_into_chunks'`).

- [ ] **Step 3: Promote `_split_into_chunks` to a module-level function**

In `src/rag_vector.py`, the current method body (`def _split_into_chunks(self, text, chunk_size=1000, overlap=200)`, around lines 609–660) uses no `self`. Move its body verbatim to a new **module-level** function placed just above the `class VectorRAG` line, and make the method delegate:

```python
def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Sentence-boundary-aware chunking (shared by VectorRAG and ManualStore)."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for sentence in sentences:
        sent_len = len(sentence)
        if sent_len > chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_len = 0
            for start in range(0, sent_len, chunk_size - overlap):
                chunks.append(sentence[start:start + chunk_size])
            continue
        if current_len + sent_len + 1 > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            overlap_sentences: List[str] = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s) + 1
            current_chunk = overlap_sentences
            current_len = sum(len(s) for s in current_chunk) + max(0, len(current_chunk) - 1)
        current_chunk.append(sentence)
        current_len += sent_len + (1 if current_len > 0 else 0)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks if chunks else [text]
```

Then replace the method body with a one-line delegation (keep the method for backward compat):

```python
    def _split_into_chunks(
        self, text: str, chunk_size: int = 1000, overlap: int = 200
    ) -> List[str]:
        return split_into_chunks(text, chunk_size, overlap)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest tests/test_manual_store.py::test_split_into_chunks_module_level_splits_long_text --import-mode=importlib -q`
Expected: PASS.

Also confirm no RAG regression:
Run: `python -m pytest tests/test_rag_remove_directory_scope.py --import-mode=importlib -q`
Expected: PASS (unchanged behavior — the method now delegates).

- [ ] **Step 5: Write the failing tests for `ManualStore`**

Append to `tests/test_manual_store.py`:

```python
import pytest
from tests.helpers.embedding_lanes import FakeChroma, FakeEmbedder, patch_chroma


@pytest.fixture
def store(monkeypatch):
    """A ManualStore whose lanes run on an in-memory FakeChroma + a fake embedder,
    exercising the real build_embedding_lanes/query_lanes machinery with no disk
    or model load. Only the fastembed lane exists (custom lane forced unavailable)."""
    import src.embedding_lanes as el
    patch_chroma(monkeypatch, FakeChroma())
    monkeypatch.setattr(el, "_build_fastembed_client", lambda: FakeEmbedder(8, "fake-embed", ""))

    def _no_custom():
        raise RuntimeError("no custom lane in tests")
    monkeypatch.setattr(el, "_build_custom_client", _no_custom)

    from src.industrial.manual_store import ManualStore
    return ManualStore(base_name="equipment_manuals_test")


def _fake_pages(pairs):
    def _extract(_source):
        return list(pairs)
    return _extract


def test_ingest_pdf_sets_per_page_metadata(store):
    res = store.ingest_file(
        "C:/manuals/vfd.pdf", title="VFD Manual",
        extract_pages=_fake_pages([(1, "F0002 overcurrent during accel."),
                                   (2, "Check motor cabling and load.")]),
    )
    assert res["manual_id"].startswith("man_")
    assert res["title"] == "VFD Manual"
    assert res["pages"] == 2
    assert res["chunks_indexed"] >= 2
    hits = store.search("overcurrent cabling", k=20, manual_id=res["manual_id"])
    assert {h["page"] for h in hits} == {1, 2}
    assert all(h["title"] == "VFD Manual" for h in hits)


def test_ingest_text_file_page_is_none(store, tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("Belt tension spec is 40 N. Grease every 500 hours.", encoding="utf-8")
    res = store.ingest_file(str(p), title="Notes")
    assert res["chunks_indexed"] >= 1
    hits = store.search("grease", k=20, manual_id=res["manual_id"])
    assert hits and all(h["page"] is None for h in hits)


def test_remove_manual_deletes_only_that_manual(store):
    a = store.ingest_file("C:/m/a.pdf", title="A", extract_pages=_fake_pages([(1, "alpha")]))
    b = store.ingest_file("C:/m/b.pdf", title="B", extract_pages=_fake_pages([(1, "bravo")]))
    assert store.remove_manual(a["manual_id"])["removed_count"] >= 1
    assert store.search("alpha", k=20, manual_id=a["manual_id"]) == []
    assert store.search("bravo", k=20, manual_id=b["manual_id"])  # B untouched
    ids = {m["manual_id"] for m in store.list_manuals()}
    assert b["manual_id"] in ids and a["manual_id"] not in ids


def test_reingest_same_file_is_idempotent(store):
    pages = _fake_pages([(1, "alpha overcurrent"), (2, "bravo cabling")])
    store.ingest_file("C:/m/a.pdf", title="A", extract_pages=pages)
    count_after_first = store._lanes[0].count()
    store.ingest_file("C:/m/a.pdf", title="A", extract_pages=pages)
    assert store._lanes[0].count() == count_after_first


def test_search_and_list_safe_when_empty(store):
    assert store.search("anything") == []
    assert store.list_manuals() == []
```

- [ ] **Step 6: Run them to verify they fail**

Run: `python -m pytest tests/test_manual_store.py --import-mode=importlib -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.industrial.manual_store'`).

- [ ] **Step 7: Implement `ManualStore`**

Create `src/industrial/manual_store.py`:

```python
"""Equipment-manual knowledge base: a dedicated ChromaDB collection (isolated from
the personal-docs RAG) of ingested manuals/datasheets, searchable with page-level
citations. Built on the existing offline embedding-lane machinery. READ operations
never raise; every method degrades to an empty/error result when the store is
unhealthy. Admin-only at the tool layer."""
import hashlib
import logging
import os

from src.embedding_lanes import build_embedding_lanes, dedupe_results, query_lanes
from src.rag_vector import split_into_chunks

logger = logging.getLogger(__name__)

MANUAL_COLLECTION = "equipment_manuals"
_PLAIN_EXTS = {".txt", ".md", ".json"}


def _pdf_pages(source):
    """Yield (1-based page_number, page_text) for each page of a text-layer PDF."""
    from pypdf import PdfReader
    reader = PdfReader(source)
    for i, page in enumerate(reader.pages):
        yield (i + 1, page.extract_text() or "")


def _read_document_text(source, ext):
    from src.personal_docs import extract_office_text, read_text_file
    from src.markitdown_runtime import MARKITDOWN_EXTS
    if ext in _PLAIN_EXTS:
        return read_text_file(source)
    if ext in MARKITDOWN_EXTS:
        return extract_office_text(source)
    return None  # unsupported


class ManualStore:
    def __init__(self, base_name: str = MANUAL_COLLECTION, lanes=None):
        self.base_name = base_name
        self._lanes = lanes if lanes is not None else build_embedding_lanes(base_name)

    @property
    def healthy(self) -> bool:
        return bool(self._lanes)

    # ------------------------------------------------------------------
    def ingest_file(self, path, title=None, *, extract_pages=None) -> dict:
        if not self.healthy:
            return {"error": "manual store unavailable"}
        source = os.path.abspath(path)
        ext = os.path.splitext(source)[1].lower()
        title = title or os.path.splitext(os.path.basename(source))[0]
        manual_id = "man_" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]

        if ext == ".pdf":
            try:
                pages = list((extract_pages or _pdf_pages)(source))
            except Exception as e:
                return {"error": f"could not read PDF: {e}"}
        else:
            text = _read_document_text(source, ext)
            if text is None:
                return {"error": f"unsupported manual type: {ext or '(none)'}"}
            pages = [(0, text)]  # 0 = non-paginated sentinel

        chunks_indexed = 0
        page_numbers = set()
        for page, text in pages:
            if not text or not text.strip():
                continue
            page_val = int(page) if isinstance(page, int) else 0
            if page_val >= 1:
                page_numbers.add(page_val)
            for chunk_id, chunk in enumerate(split_into_chunks(text)):
                meta = {
                    "manual_id": manual_id, "title": title, "source": source,
                    "page": page_val, "chunk_id": chunk_id, "kind": "manual",
                }
                if self._add(chunk, meta):
                    chunks_indexed += 1

        if chunks_indexed == 0:
            return {"error": "no extractable text in manual"}
        return {
            "manual_id": manual_id, "title": title,
            "pages": len(page_numbers), "chunks_indexed": chunks_indexed,
        }

    def _add(self, text: str, meta: dict) -> bool:
        doc_id = "man_" + hashlib.sha256(
            f"{meta['manual_id']}\x00{meta['page']}\x00{meta['chunk_id']}".encode("utf-8")
        ).hexdigest()[:16]
        wrote = False
        for lane in self._lanes:
            try:
                if lane.collection.get(ids=[doc_id]).get("ids"):
                    wrote = True
                    continue
                lane.collection.add(
                    ids=[doc_id], embeddings=lane.encode([text]),
                    documents=[text], metadatas=[meta],
                )
                wrote = True
            except Exception as e:
                logger.warning("manual _add failed in %s lane: %s", lane.name, e)
        return wrote

    # ------------------------------------------------------------------
    def list_manuals(self) -> list:
        if not self.healthy:
            return []
        try:
            data = self._lanes[0].collection.get(include=["metadatas"])
        except Exception as e:
            logger.warning("list_manuals failed: %s", e)
            return []
        manuals = {}
        for meta in data.get("metadatas") or []:
            if not isinstance(meta, dict):
                continue
            mid = meta.get("manual_id")
            if not mid:
                continue
            m = manuals.setdefault(mid, {
                "manual_id": mid, "title": meta.get("title"),
                "source": meta.get("source"), "chunk_count": 0,
            })
            m["chunk_count"] += 1
        return list(manuals.values())

    def remove_manual(self, manual_id) -> dict:
        if not self.healthy:
            return {"removed_count": 0}
        removed = set()
        for lane in self._lanes:
            try:
                res = lane.collection.get(where={"manual_id": manual_id}, include=[])
                ids = res.get("ids") or []
                if ids:
                    lane.collection.delete(ids=ids)
                    removed.update(ids)
            except Exception as e:
                logger.warning("remove_manual failed in %s lane: %s", lane.name, e)
        return {"removed_count": len(removed)}

    # ------------------------------------------------------------------
    def search(self, query, k: int = 5, manual_id=None) -> list:
        if not self.healthy or not query or not isinstance(query, str):
            return []
        where = {"manual_id": manual_id} if manual_id else None
        candidates = []
        try:
            for lane, results in query_lanes(
                self._lanes, query,
                n_results=lambda lane: min(max(k, 20), lane.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            ):
                for idx in range(len(results["ids"][0])):
                    meta = results["metadatas"][0][idx] or {}
                    distance = results["distances"][0][idx]
                    page = meta.get("page")
                    page = page if isinstance(page, int) and page >= 1 else None
                    candidates.append({
                        "id": results["ids"][0][idx],
                        "title": meta.get("title"),
                        "page": page,
                        "source": meta.get("source"),
                        "snippet": results["documents"][0][idx],
                        "chunk_id": meta.get("chunk_id"),
                        "score": round(1.0 - distance, 4),
                    })
        except Exception as e:
            logger.warning("manual search failed: %s", e)
            return []
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return dedupe_results(candidates, limit=k)


_manual_store = None


def get_manual_store():
    """Lazy singleton. Returns a ManualStore (possibly unhealthy — its methods then
    return empty/error results) or None if construction raised."""
    global _manual_store
    if _manual_store is None:
        try:
            _manual_store = ManualStore()
        except Exception as e:
            logger.error("get_manual_store init failed: %s", e)
            return None
    return _manual_store
```

Note: `src/industrial/__init__.py` already exists (from the live-data feature). If it does not, create an empty one.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `python -m pytest tests/test_manual_store.py --import-mode=importlib -q`
Expected: PASS (6 passed).

- [ ] **Step 9: Commit**

```bash
git add src/rag_vector.py src/industrial/manual_store.py tests/test_manual_store.py
git commit -m "feat(industrial): ManualStore — dedicated equipment-manual KB (page-cited)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `ingest_equipment_manual` tool

**Files:**
- Create: `src/agent_tools/industrial_manuals.py`
- Test: `tests/test_ingest_equipment_manual.py`

**Interfaces:**
- Consumes: `src.industrial.manual_store.get_manual_store` (default store); a store exposing `ingest_file(path, title=None)`, `list_manuals()`, `remove_manual(manual_id)`.
- Produces: `async ingest_equipment_manual(content, ctx, *, store=None) -> dict`; `class IngestEquipmentManualTool` with `async execute(self, content, ctx) -> dict`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingest_equipment_manual.py`:

```python
import asyncio
import json

import src.agent_tools.industrial_manuals as im


def _run(coro):
    return asyncio.run(coro)


class FakeStore:
    def __init__(self):
        self.ingested = []
        self.removed = []
    def ingest_file(self, path, title=None):
        self.ingested.append((path, title))
        return {"manual_id": "man_abc", "title": title or "x", "pages": 2, "chunks_indexed": 5}
    def list_manuals(self):
        return [{"manual_id": "man_abc", "title": "VFD", "source": "C:/m/vfd.pdf", "chunk_count": 5}]
    def remove_manual(self, manual_id):
        self.removed.append(manual_id)
        return {"removed_count": 3}


def _exec(content, store=None, ctx=None):
    store = store or FakeStore()
    out = _run(im.ingest_equipment_manual(content, ctx or {"owner": "admin"}, store=store))
    return out, store


def test_add_happy_path(tmp_path):
    f = tmp_path / "vfd.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    out, store = _exec(json.dumps({"action": "add", "path": str(f), "title": "VFD"}))
    assert out["output"]["manual_id"] == "man_abc"
    assert store.ingested == [(str(f), "VFD")]


def test_add_missing_file_is_error():
    out, _ = _exec(json.dumps({"action": "add", "path": "C:/nope/missing.pdf"}))
    assert "error" in out


def test_add_non_str_path_is_error():
    out, _ = _exec(json.dumps({"action": "add", "path": 5}))
    assert "error" in out


def test_add_null_byte_path_never_raises():
    out, _ = _exec(json.dumps({"action": "add", "path": "C:/x" + chr(0) + ".pdf"}))
    assert "error" in out


def test_list_action():
    out, _ = _exec(json.dumps({"action": "list"}))
    assert out["output"]["manuals"][0]["manual_id"] == "man_abc"


def test_remove_action():
    out, store = _exec(json.dumps({"action": "remove", "manual_id": "man_abc"}))
    assert out["output"]["removed_count"] == 3 and store.removed == ["man_abc"]


def test_remove_missing_id_is_error():
    out, _ = _exec(json.dumps({"action": "remove"}))
    assert "error" in out


def test_bad_json_is_error():
    out, _ = _exec("not json")
    assert "error" in out


def test_unknown_action_is_error():
    out, _ = _exec(json.dumps({"action": "explode"}))
    assert "error" in out


def test_store_raising_never_raises():
    class Boom(FakeStore):
        def list_manuals(self):
            raise RuntimeError("chroma down")
    out, _ = _exec(json.dumps({"action": "list"}), store=Boom())
    assert "error" in out and "chroma down" in out["error"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_ingest_equipment_manual.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.agent_tools.industrial_manuals'`).

- [ ] **Step 3: Implement the ingest tool**

Create `src/agent_tools/industrial_manuals.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ingest_equipment_manual.py --import-mode=importlib -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add src/agent_tools/industrial_manuals.py tests/test_ingest_equipment_manual.py
git commit -m "feat(industrial): ingest_equipment_manual tool (admin, never-raises)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `search_equipment_manual` tool

**Files:**
- Modify: `src/agent_tools/industrial_manuals.py` (append the search handler + class)
- Test: `tests/test_search_equipment_manual.py`

**Interfaces:**
- Consumes: `src.industrial.manual_store.get_manual_store`; a store exposing `search(query, k=5, manual_id=None) -> list[dict]` where each hit is `{title, page, source, snippet, chunk_id, score}` (`page` is an int ≥ 1 or `None`).
- Produces: `async search_equipment_manual(content, ctx, *, store=None) -> dict`; `class SearchEquipmentManualTool` with `async execute(self, content, ctx) -> dict`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_search_equipment_manual.py`:

```python
import asyncio
import json

import src.agent_tools.industrial_manuals as im


def _run(coro):
    return asyncio.run(coro)


class FakeStore:
    def __init__(self, hits=None):
        self._hits = hits if hits is not None else [
            {"title": "VFD Manual", "page": 42, "source": "C:/m/vfd.pdf",
             "snippet": "F0002 = overcurrent during acceleration.", "chunk_id": 3, "score": 0.9},
            {"title": "Notes", "page": None, "source": "C:/m/notes.txt",
             "snippet": "Grease every 500 hours.", "chunk_id": 0, "score": 0.5},
        ]
        self.calls = []
    def search(self, query, k=5, manual_id=None):
        self.calls.append((query, k, manual_id))
        return self._hits


def _exec(content, store=None, ctx=None):
    store = store or FakeStore()
    out = _run(im.search_equipment_manual(content, ctx or {"owner": "admin"}, store=store))
    return out, store


def test_search_returns_page_citation():
    out, store = _exec(json.dumps({"query": "overcurrent"}))
    cites = out["output"]["citations"]
    assert "VFD Manual, p.42" in cites
    assert "Notes (excerpt 1)" in cites  # page-less -> excerpt (chunk_id 0 -> 1)
    assert out["output"]["results"][0]["page"] == 42
    assert store.calls == [("overcurrent", 5, None)]


def test_k_clamped_and_bool_rejected():
    out, store = _exec(json.dumps({"query": "x", "k": True}))
    assert store.calls[0][1] == 5  # bool k ignored -> default 5
    _exec_out, store2 = _exec(json.dumps({"query": "x", "k": 999}), store=FakeStore())
    assert store2.calls[0][1] == 20  # clamped to max 20


def test_manual_id_filter_passed_through():
    out, store = _exec(json.dumps({"query": "x", "manual_id": "man_abc"}))
    assert store.calls[0][2] == "man_abc"


def test_missing_query_is_error():
    out, _ = _exec(json.dumps({"k": 5}))
    assert "error" in out


def test_non_str_query_is_error():
    out, _ = _exec(json.dumps({"query": 5}))
    assert "error" in out


def test_no_results_is_clean_output_not_error():
    out, _ = _exec(json.dumps({"query": "nothing"}), store=FakeStore(hits=[]))
    assert "error" not in out
    assert out["output"]["results"] == []
    assert "No matching" in out["output"]["message"]


def test_bad_json_is_error():
    out, _ = _exec("not json")
    assert "error" in out


def test_store_raising_never_raises():
    class Boom(FakeStore):
        def search(self, query, k=5, manual_id=None):
            raise RuntimeError("chroma down")
    out, _ = _exec(json.dumps({"query": "x"}), store=Boom())
    assert "error" in out and "chroma down" in out["error"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_search_equipment_manual.py --import-mode=importlib -q`
Expected: FAIL (`AttributeError: module ... has no attribute 'search_equipment_manual'`).

- [ ] **Step 3: Implement the search tool**

Append to `src/agent_tools/industrial_manuals.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_search_equipment_manual.py --import-mode=importlib -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/agent_tools/industrial_manuals.py tests/test_search_equipment_manual.py
git commit -m "feat(industrial): search_equipment_manual tool (page-cited, never-raises)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Register both tools + bundle check

**Files:**
- Modify: `src/agent_tools/__init__.py`, `src/tool_schemas.py`, `src/tool_security.py`, `src/tool_index.py`, `src/agent_loop.py`
- Test: `tests/test_manuals_registration.py`

**Interfaces:**
- Consumes: `IngestEquipmentManualTool`, `SearchEquipmentManualTool` from `src.agent_tools.industrial_manuals`.
- Produces: both tools registered at every builtin-tool surface; `search_equipment_manual` gated read-only-in-plan, `ingest_equipment_manual` gated as a mutator; both admin-only.

Read each file's current structure first; place new entries beside the `read_equipment` / `diagnose_equipment` siblings (grep for them — line numbers below are approximate).

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_manuals_registration.py`:

```python
def test_registered_handlers_and_tags():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    from src.agent_tools.industrial_manuals import (
        IngestEquipmentManualTool, SearchEquipmentManualTool)
    assert TOOL_HANDLERS["ingest_equipment_manual"].__self__.__class__ is IngestEquipmentManualTool
    assert TOOL_HANDLERS["search_equipment_manual"].__self__.__class__ is SearchEquipmentManualTool
    assert {"ingest_equipment_manual", "search_equipment_manual"} <= TOOL_TAGS


def test_admin_and_plan_gating():
    import src.tool_security as ts
    for name in ("ingest_equipment_manual", "search_equipment_manual"):
        assert name in ts.NON_ADMIN_BLOCKED_TOOLS
    # search is read-only in plan mode; ingest is a mutator (must NOT be readonly)
    assert "search_equipment_manual" in ts.PLAN_MODE_READONLY_TOOLS
    assert "ingest_equipment_manual" not in ts.PLAN_MODE_READONLY_TOOLS
    assert "ingest_equipment_manual" in ts._PLAN_MODE_KNOWN_MUTATORS
    # and the derived plan-mode denylist blocks ingest but not search
    disabled = ts.plan_mode_disabled_tools()
    assert "ingest_equipment_manual" in disabled
    assert "search_equipment_manual" not in disabled


def test_schema_index_sections_domain():
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    import src.tool_index as ti
    import src.agent_loop as al
    names = {s["function"]["name"] for s in FUNCTION_TOOL_SCHEMAS if s.get("type") == "function"}
    for name in ("ingest_equipment_manual", "search_equipment_manual"):
        assert name in names
        assert name in ti.BUILTIN_TOOL_DESCRIPTIONS
        assert name in al.TOOL_SECTIONS
        assert name in al._DOMAIN_TOOL_MAP["desktop"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_manuals_registration.py --import-mode=importlib -q`
Expected: FAIL (not registered yet).

- [ ] **Step 3: Register handlers + tags** (`src/agent_tools/__init__.py`)

Add an import beside the other `from .` imports:
```python
from .industrial_manuals import IngestEquipmentManualTool, SearchEquipmentManualTool
```
In `TOOL_HANDLERS`, beside `"read_equipment"`:
```python
    "ingest_equipment_manual": IngestEquipmentManualTool().execute,
    "search_equipment_manual": SearchEquipmentManualTool().execute,
```
In `TOOL_TAGS`, add both names: `"ingest_equipment_manual", "search_equipment_manual"`.

- [ ] **Step 4: Add function-tool schemas** (`src/tool_schemas.py`, in `FUNCTION_TOOL_SCHEMAS` beside `read_equipment`)

```python
    {
        "type": "function",
        "function": {
            "name": "ingest_equipment_manual",
            "description": "Admin: add an equipment manual / datasheet (PDF, txt, md, docx, pptx, xlsx, epub) to the searchable manuals knowledge base, or list/remove entries. PDFs are indexed per page for page-cited retrieval. READ-ONLY of the source file; writes only to the local manuals index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "list", "remove"], "description": "add (default), list, or remove"},
                    "path": {"type": "string", "description": "add: local file path of the manual"},
                    "title": {"type": "string", "description": "add: optional display title (defaults to the filename)"},
                    "manual_id": {"type": "string", "description": "remove: the manual_id returned by add/list"}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_equipment_manual",
            "description": "Search the equipment-manual knowledge base and return passages WITH CITATIONS (manual title + page). Use it to look up a fault code, spec, or procedure — e.g. after diagnose_equipment surfaces a code or read_equipment returns an anomalous value — and cite the manual + page in your answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "what to look up (fault code, component, symptom, spec)"},
                    "k": {"type": "integer", "description": "max passages (default 5, 1..20)"},
                    "manual_id": {"type": "string", "description": "optional: restrict to one manual"}
                },
                "required": ["query"]
            }
        }
    },
```

- [ ] **Step 5: Gating** (`src/tool_security.py`)

- In `NON_ADMIN_BLOCKED_TOOLS` (beside `"read_equipment"`): add `"ingest_equipment_manual",` and `"search_equipment_manual",`.
- In `PLAN_MODE_READONLY_TOOLS` (beside `"read_equipment"`): add **only** `"search_equipment_manual",`.
- In `_PLAN_MODE_KNOWN_MUTATORS` (beside `"run_workflow",`): add `"ingest_equipment_manual",`.

- [ ] **Step 6: Tool index descriptions** (`src/tool_index.py`, in `BUILTIN_TOOL_DESCRIPTIONS`)

```python
    "ingest_equipment_manual": "Admin: add/list/remove an equipment manual or datasheet (PDF/office/text) in the searchable manuals knowledge base; PDFs indexed per page.",
    "search_equipment_manual": "Search ingested equipment manuals and return passages with citations (manual title + page) — for fault codes, specs, procedures, wiring, and datasheets.",
```

- [ ] **Step 7: Prompt sections + domain** (`src/agent_loop.py`)

In `TOOL_SECTIONS` (beside `"read_equipment"`):
```python
    "ingest_equipment_manual": "- ```ingest_equipment_manual``` — Admin: add/list/remove a manual in the KB. Args (JSON): add → {\"action\":\"add\",\"path\":\"C:/manuals/vfd.pdf\",\"title\":\"VFD Manual\"}; list → {\"action\":\"list\"}; remove → {\"action\":\"remove\",\"manual_id\":\"man_...\"}.",
    "search_equipment_manual": "- ```search_equipment_manual``` — Search ingested equipment manuals; returns passages WITH CITATIONS (title + page). Args (JSON): {\"query\":\"F0002 overcurrent\",\"k\":5}. Cite the manual + page in your answer; pair with diagnose_equipment / read_equipment.",
```
In `_DOMAIN_TOOL_MAP["desktop"]` (the set at ~line 306, which already holds `diagnose_equipment`): add `"ingest_equipment_manual", "search_equipment_manual"`.

- [ ] **Step 8: Run the parity test + import smoke**

Run: `python -m pytest tests/test_manuals_registration.py --import-mode=importlib -q`
Expected: PASS (3 passed).
Run: `python -c "import app"`
Expected: no error.

- [ ] **Step 9: Run the full feature suite (no regression)**

Run: `python -m pytest tests/test_manual_store.py tests/test_ingest_equipment_manual.py tests/test_search_equipment_manual.py tests/test_manuals_registration.py --import-mode=importlib -q`
Expected: PASS (27 passed).

- [ ] **Step 10: Commit**

```bash
git add src/agent_tools/__init__.py src/tool_schemas.py src/tool_security.py src/tool_index.py src/agent_loop.py tests/test_manuals_registration.py
git commit -m "feat(industrial): register ingest/search_equipment_manual (admin, plan-gated)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **All TDD.** `ManualStore` tests run the real `build_embedding_lanes`/`query_lanes`/`dedupe_results` machinery over an in-memory `FakeChroma` + `FakeEmbedder` (see `tests/helpers/embedding_lanes.py`) — no disk, no model load, no hardware. Tool tests inject a `FakeStore`.
- **No new dependencies, no `Assist.spec` change** — `pypdf` + FastEmbed are already bundled. A frozen boot-check (after a rebuild) confirms both tools import in the bundle; expected clean.
- **Never-raises is a hard requirement** — the null-byte-path test in Task 2 guards the same class of leak found in `read_equipment` (an OS call that raises *before* the try). Keep `os.path.isfile` inside the handler's `try` (it is).
- **Gating is a security requirement** — both tools admin-only; `ingest_equipment_manual` must stay OUT of `PLAN_MODE_READONLY_TOOLS` and IN `_PLAN_MODE_KNOWN_MUTATORS`. Do not weaken a parity assertion to make it pass; if a module name moved, fix the registration AND the assertion.
- **Owed by the user (not automated):** ingesting a *real* manual PDF and judging answer quality with a served chat model; a frozen boot-check after the next rebuild.
- **Scope:** the read/ingest foundation only. Do NOT build a UI upload panel, chat auto-injection, OCR of scanned PDFs, per-user manual libraries, or cross-manual synthesis (all explicit non-goals).
