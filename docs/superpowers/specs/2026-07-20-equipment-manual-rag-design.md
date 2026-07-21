# Equipment-Manual RAG — Design

**Goal:** Give the Industrial Assistant a dedicated, admin-managed knowledge base of
equipment manuals / datasheets / fault-code tables, plus two admin-only agent tools —
one to **ingest** a manual, one to **search** it and return **cited** passages (e.g.
"VFD Manual, p.42") — so the agent can quote a manual while diagnosing a fault. Pairs
with the shipped `diagnose_equipment` (vision) and `read_equipment` (live-data) tools.

**Scope:** One cohesive slice — a dedicated ChromaDB collection plus two agent tools
(ingest + cited search). Not a new RAG engine; it reuses the app's existing vector
infrastructure.

---

## Background — what this builds on

The app already has a full offline RAG stack; this feature is an industrial-flavored,
**cited** slice over it, not net-new RAG:

- **`VectorRAG`** (`src/rag_vector.py`) — ChromaDB, hybrid search (vector 0.7 + keyword
  0.3), sentence-aware chunking (~1000 chars / 200 overlap), owner-scoped metadata, one
  shared collection `odysseus_rag`.
- **`embedding_lanes`** (`src/embedding_lanes.py`) — `build_embedding_lanes(base_name)`
  returns retrieval lanes (an optional custom HTTP endpoint + the bundled **offline
  FastEmbed** model), each a `EmbeddingLane` with `.encode()`, `.collection` (a ChromaDB
  collection), `.count()`. `query_lanes(...)` + `dedupe_results(...)` implement hybrid
  multi-lane retrieval. ChromaDB fixes a collection's embedding dimension on first
  insert, so each lane gets its own suffixed collection (`<base>_fastembed`,
  `<base>_custom`) — this is why a **dedicated base name** cleanly isolates a new KB.
- **`PersonalDocsManager`** (`src/personal_docs.py`) — indexes tracked directories;
  `extract_pdf_text()` uses **`pypdf`** (already a dependency) but concatenates all
  pages with **no page markers**. Page-level citation therefore needs per-page
  extraction, which this feature adds.
- **Retrieval reaches the agent only via auto-injection today** (`retrieve_personal` →
  `[filename :: vector search]`). There is **no callable knowledge-search tool** and **no
  citation surfaced to the user**. `search_chats` exists for chat history but nothing for
  documents. This gap is exactly what this feature fills.
- **Sibling agent tools** — `diagnose_equipment` / `read_equipment` are admin-only
  builtin tools registered at ~7 surfaces; their gating and never-raises discipline are
  the precedent this feature follows.

**No new dependencies.** `pypdf` (PDF text) and FastEmbed (offline embeddings) are
already bundled. This is buildable and testable fully headlessly.

## Architecture

One focused store class over the existing embedding infra, plus two thin agent tools:

- **`src/industrial/manual_store.py`** — `ManualStore`: owns a **dedicated** ChromaDB
  collection base `equipment_manuals`, built with `build_embedding_lanes("equipment_manuals")`
  so it is isolated from `odysseus_rag` and reuses offline FastEmbed + hybrid retrieval.
  The per-page PDF extractor is **injectable** (default = real pypdf) so the store's
  logic is unit-testable without a PDF fixture.
- **`src/agent_tools/industrial_manuals.py`** — `IngestEquipmentManualTool` and
  `SearchEquipmentManualTool`: `execute(content, ctx) -> dict` handlers that validate
  model/attacker JSON, call the store, and **never raise**. Thin adapters; the store is
  injectable for tests.

**Admin-managed, shared KB.** Manuals are reference material an admin curates once; the
KB is not per-user (unlike personal docs). Both tools are admin-only.

## ManualStore

`equipment_manuals` collection. Chunk metadata:
`{manual_id, title, source (abs path), page (int|null), chunk_id, kind:"manual"}`.
`doc_id = "man_" + sha256(manual_id + "\x00" + str(page) + "\x00" + str(chunk_id))[:16]`
so re-ingesting the same manual is idempotent (existing ids are skipped, matching
`VectorRAG.add_document`).

- **`ingest_file(path, title=None, *, extract_pages=None) -> dict`** — resolve the file;
  pick an extractor by extension:
  - **PDF** → `extract_pages` (default: iterate `pypdf` `reader.pages`, yielding
    `(page_number, page_text)` 1-based). Each page's text is chunked with the existing
    sentence-aware splitter **within the page** (a chunk never straddles a page boundary,
    keeping citations exact); each chunk stored with its `page`.
  - **txt / md / json / docx / pptx / xlsx / epub** → the existing whole-file extractors
    (`read_text_file`, `extract_office_text`); `page=null`; chunked normally.
  - `manual_id = "man_" + sha256(abspath(source))[:12]` — **keyed on the absolute source
    path**, so re-ingesting the same file is idempotent (same `manual_id` → same per-page
    `doc_id`s → existing chunks skipped) and two different files never collide. `title` is
    a separate display label (defaults to the filename stem when not given). Returns
    `{manual_id, title, pages, chunks_indexed}`.
- **`list_manuals() -> list`** — distinct manuals in the KB with
  `{manual_id, title, source, chunk_count}` (derived from stored metadata).
- **`remove_manual(manual_id) -> dict`** — delete all chunks with that `manual_id` across
  lanes (Python-side id collection, like `VectorRAG.delete_by_source`). Returns
  `{removed_count}`.
- **`search(query, k=5, manual_id=None) -> list`** — hybrid retrieval via `query_lanes`
  + `dedupe_results`; optional `where={"manual_id": manual_id}` filter. Each hit:
  `{title, page, source, snippet, score}` (snippet = the chunk text, trimmed).

The store degrades safely: if no embedding lane is available (`build_embedding_lanes`
returns `[]`), `healthy` is False and every method returns an empty result / error dict —
never raises.

## Tool contract

Both handlers: `execute(content, ctx) -> dict`; `content` is JSON; `ctx` carries `owner`.
Never raise — every failure returns `{"error": …}`. Hostile/wrong-shape args are
shape-guarded (isinstance checks), not just value-checked (the lesson from
`run_workflow` / `diagnose_equipment` / `read_equipment`).

**`ingest_equipment_manual`** (admin, KB-mutating):
- `action` (`add` | `list` | `remove`, default `add`).
- `add`: `path` (required str, a local file), `title` (optional str). Guards: non-str /
  empty path → error; path not an existing file → error; unsupported extension → error;
  extraction yields no text → error. Returns
  `{"output": {manual_id, title, pages, chunks_indexed}}`.
- `list`: returns `{"output": {manuals: [...]}}`.
- `remove`: `manual_id` (required str) → `{"output": {removed_count}}`.

**`search_equipment_manual`** (admin, read-only):
- `query` (required str), `k` (optional int, default 5, clamped 1..20), `manual_id`
  (optional str filter). Returns a rendered citation block **and** structured results:
  ```
  {"output": {
      "citations": "**VFD Manual, p.42** — \"F0002 = overcurrent…\"\n**VFD Manual, p.7** — \"…\"",
      "results": [{"title","page","source","snippet","score"}, ...]
  }}
  ```
  A page-less hit renders as `**Title (excerpt N)**`. No results → a clear "no matching
  passage" output (not an error).

## Pairing with diagnostics (no hard coupling)

The tool descriptions and `agent_loop.TOOL_SECTIONS` entry instruct the agent: after
`diagnose_equipment` surfaces a fault code, or `read_equipment` returns an anomalous
value, call `search_equipment_manual` for that code/component and **cite** the passage in
its answer. The tools remain independent; the agent orchestrates. No change to
`diagnose_equipment` itself in v1.

## Security & gating

- **Admin-only** — both tools in `NON_ADMIN_BLOCKED_TOOLS`, matching the industrial
  siblings (`diagnose_equipment` / `read_equipment`). Only an admin curates the KB and
  reads it.
- **Plan-mode** — `search_equipment_manual` is read-only → also in
  `PLAN_MODE_READONLY_TOOLS`. `ingest_equipment_manual` mutates the KB → **not**
  plan-readonly (treated as a write; add to `_PLAN_MODE_KNOWN_MUTATORS` if that list is
  used, matching a write sibling).
- **No network.** Ingest reads a local file from an admin-supplied path; the path is
  shape-guarded and must be an existing file. (Reusing an admin-only file read matches
  the existing filesystem-tool trust model; no path-escape guard beyond "exists + is a
  file" is added in v1 since the caller is already an admin.)
- **Domain** — group both under the same `_DOMAIN_TOOL_MAP` domain as the other
  industrial tools.

## Error handling

The handlers never raise into the agent loop. All of: non-JSON / non-object `content`,
a missing/non-string required arg, an unknown `action`, an unsupported or unreadable
file, empty extraction, an unhealthy store, or the store raising → each returns
`{"error": …}`. `ManualStore` itself catches ChromaDB / extractor errors and returns
empty results or raises only into the tool's try/except (which converts to `{"error"}`).

## Testing

All headless — no PDF fixture, no hardware, no served model required for the unit tests.

- **`ManualStore` (unit, injected extractor):** inject a fake `extract_pages` returning
  `[(1, "text one"), (2, "text two")]` → assert chunks carry `page` 1 and 2 and the
  right `manual_id`; ingest a `.txt` fixture (real extractor) → assert `page=null`
  fallback; `search` returns cited hits with the seeded page; `remove_manual` deletes
  only that manual's chunks; re-ingest is idempotent (chunk count stable).
- **`ManualStore` (integration, one test):** a real `build_embedding_lanes` round-trip
  against a temp Chroma dir — ingest two short pages, search, assert the top hit's page.
- **Tool tests (injected fake store):** ingest add/list/remove happy paths; search
  citation-block format; the never-raises set (bad JSON, non-str path/query, unknown
  action, unsupported file, store raises); arg shape-guards; result shape.
- **Registration parity test:** both tools present at every surface (handlers/tags,
  `FUNCTION_TOOL_SCHEMAS`, `NON_ADMIN_BLOCKED_TOOLS`, search in `PLAN_MODE_READONLY_TOOLS`
  and ingest **not**, `BUILTIN_TOOL_DESCRIPTIONS`, `TOOL_SECTIONS`, `_DOMAIN_TOOL_MAP`).
- **Frozen boot-check (owed after rebuild):** `pypdf` + FastEmbed import in the bundle
  and both tools register — no new deps, so expected clean.

## Registration & bundling

- Register `ingest_equipment_manual` and `search_equipment_manual` at every builtin-tool
  surface (handler/tags, schema, security lists, `BUILTIN_TOOL_DESCRIPTIONS`,
  `TOOL_SECTIONS` + `_DOMAIN_TOOL_MAP`), with a parity test — matching the industrial
  siblings.
- **No new dependencies** and **no `Assist.spec` change** (pypdf + FastEmbed already
  bundled/collected). A frozen boot-check confirms both tools import in the bundle.

## Non-goals (this sub-project)

- A UI upload panel (v1 ingest is the admin agent tool).
- Auto-injection of manual context into every chat (v1 is explicit-tool retrieval so the
  agent decides when to cite).
- OCR of scanned / image-only PDFs (text-layer PDFs only; `pypdf` returns empty for
  image-only pages, which surfaces as an "empty extraction" error).
- Cross-manual synthesis / summarization tooling (the agent does that from cited hits).
- Per-user manual libraries (the KB is shared and admin-curated).
- Engineering-unit or diagram understanding (that lives with the vision tool).
