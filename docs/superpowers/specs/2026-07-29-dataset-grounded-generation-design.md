# AI Studio — Document-Grounded Synthetic Generation — Design

**Goal:** Extend the Dataset Builder's "Generate (AI)" card with an optional **grounding source** so
generated training rows are drawn from real source text — either a **document the user uploads** (a PDF or
text file, chunked by page) or the **ingested equipment-manual library** (passages retrieved by topic). Each
staged row shows the document + page it came from so the user can verify faithfulness; the **saved row stays
clean** (no citation baked in). Sub-project 3 of "AI Studio: dataset tools" (sub-project 1 = Builder+Validator,
sub-project 2 = free synthetic generation, both shipped on dev).

---

## The core idea

Sub-project 2 shipped a pure, injectable, never-raising generation core (`src/dataset_tools/generate.py`:
`build_generation_prompt`, `parse_generated_rows`, `generate_rows`). Grounding is **"feed real source text
into that generator as `context`"**: the prompt shifts from "invent rows about X" to "write rows answerable
ONLY from this passage," which makes the output factually faithful to the source. Two source adapters produce
a common list of grounding chunks `[{"source", "text"}]`; a thin orchestrator walks the chunks, calls the
shipped `generate_rows` per chunk with the chunk as `context`, tags each candidate row with its source, and
dedups across chunks by reusing `generate_rows`'s own `existing`-based dedup (passing the growing accepted
list). The row itself stays clean — the source label rides alongside it in the staging report only.

The building blocks already exist: `src/industrial/manual_store.py` provides `_pdf_pages(source)` (yields
`(1-based page_no, page_text)`), `_read_document_text(source, ext)`, and `ManualStore.search(query, k,
manual_id) -> [hit]` (hits carry `title`/`page`/`snippet`) over the dedicated `equipment_manuals` ChromaDB
collection. No new heavy dependency, no feasibility-gate.

## Scope decisions (baked in)

- **Both grounding sources:** an **uploaded document** (full multipart upload; PDF via `_pdf_pages`, text via
  `_read_document_text`; NOT ingested into the RAG — extracted just for grounding) AND the **ingested
  manual library** (topic query → `ManualStore.search`).
- **Clean saved rows:** grounding uses the passage; the staging preview shows each row's source
  (`"<title>, p.<n>"`); the saved training row carries no citation.
- Reuses sub-project 2's `generate_rows` unchanged in contract (adds one optional `context` param, backward
  compatible). Reuses the Generate card + staging UI; adds a source selector.

## Architecture / components

**Backend (main app, Python 3.14):**

- **`src/dataset_tools/generate.py`** (extend, backward-compatible):
  - `build_generation_prompt(fmt, count, brief, seed_rows=None, context=None)` — when `context` (a source
    passage, a str) is non-empty, prepend a grounding instruction to the system/user message: "Generate rows
    ANSWERABLE ONLY from the following source text. Do NOT invent facts not present in it. Source:\n<context>".
    `brief` (optional) still threads as extra guidance. No `context` → identical to today.
  - `generate_rows(fmt, count, brief, *, seed_rows=None, existing=None, model_call, batch_size=10,
    max_attempts=None, context=None)` — threads `context` to `build_generation_prompt`. Unchanged otherwise;
    still never raises.

- **`src/dataset_tools/ground.py`** (new — pure core, injectable deps, never-raises):
  - `chunk_document(pages, *, max_chars=2000) -> [{"source","text"}]` — `pages` is an iterable of
    `(page_no, page_text)`. Emits one chunk per page (`source="p.<n>"`), splitting a page's text into
    sub-chunks of at most `max_chars` (keeping the same page label). Skips blank pages. Never raises.
  - `async generate_grounded(chunks, fmt, count, *, model_call, existing=None, brief="", per_chunk=4,
    batch_size=4, max_chunks=200) -> dict` — walks `chunks` (capped at `max_chunks`); for each, calls the
    shipped `generate_rows(fmt, min(per_chunk, remaining), brief, existing=<accepted-so-far>,
    model_call=model_call, batch_size=batch_size, context=chunk["text"])`; tags every returned candidate with
    `"source": chunk["source"]`; accumulates valid non-duplicate rows (cross-chunk dedup comes free via the
    growing `existing`). Stops when `count` valid rows are collected or chunks are exhausted. Returns
    `{"rows":[{row,valid,error,duplicate,source}...], "valid","invalid","duplicates","requested","produced",
    "chunks_used", "error"?}`. Never raises (a `generate_rows` that reports an error is surfaced, not thrown).

- **Source adapters** (thin, in `ground.py` or the route module):
  - *Library:* `get_manual_store().search(query, k)` → for each hit build a chunk
    `{"source": "<title>, p.<page>", "text": hit["snippet"]}`.
  - *Upload:* `_pdf_pages(path)` (PDF) or `_read_document_text(path, ext)` (other) → `(page_no, text)` pairs →
    `chunk_document(...)`; label the source with the uploaded filename + page.

- **`routes/dataset_routes.py`** (extend the shipped admin router):
  - `POST /api/datasets/generate/grounded` (JSON, admin) — `{format, count, query, k?, brief?, existing?}` →
    library adapter → `generate_grounded` → staged report. Empty query / no hits → report with a clear
    `error`/message. Never 500.
  - `POST /api/datasets/generate/upload` (multipart, admin) — file + form fields (`format`, `count`,
    `brief?`, `existing?` as a JSON string) → save to a temp path → extract (`_pdf_pages`/`_read_document_text`)
    → `chunk_document` → `generate_grounded` → staged report; temp file cleaned up in `finally`. Unsupported
    file / extraction failure → report `error`, never 500. `count` clamped `[1, MAX_GENERATE=200]` (reuse the
    sub-project-2 constant). Both routes use the sub-project-2 default `model_call` (`_default_model_call`).

**Frontend (extend the Generate card):**
- A **Source** selector: *None (free)* [sub-project-2 behavior] / *Uploaded document* / *Manual library*.
  Selecting a source reveals its input: a file `<input type=file>` for upload, or a topic-query text input for
  the library. On Generate, the card POSTs to `/generate` (none), `/generate/upload` (multipart, for upload),
  or `/generate/grounded` (JSON, for library).
- **Staging shows the source:** each staged row renders its `source` label (e.g. "VFD Manual, p.42") when
  present; "Add valid rows" merges the clean rows (no source, no citation) into the working dataset, exactly as
  in sub-project 2.
- The Help "Datasets (AI Studio)" section gains one sentence about grounding (unique phrase:
  "grounded in a document or your manuals").

## Data flow

pick source → (upload: multipart → temp file → `_pdf_pages`/`_read_document_text` → `chunk_document`; library:
topic → `ManualStore.search` → chunks) → `generate_grounded` walks chunks → per-chunk grounded `generate_rows`
→ tag `source` + cross-chunk dedup → staged report (rows + source labels) → user reviews with sources → Add
valid rows (clean) → Save (sub-project 1).

## Error handling

- `chunk_document`, `generate_grounded` **never raise** — bad pages, a `model_call` that raises (surfaced via
  `generate_rows`), zero chunks → a report, never a throw.
- Routes **never 500**: both grounded routes return HTTP 200 with the report; failures (no query, no hits,
  unsupported/corrupt file, no model endpoint) ride the report's `error` field. `count` clamped. Upload temp
  file removed in `finally`.
- Admin-gated by the existing `setup_dataset_routes()` router dependency.
- Every generated row still validated by `normalize_row` before it can be staged valid — grounding cannot
  inject a malformed row.

## Testing

Headless:
- `build_generation_prompt(..., context=...)` — includes the grounding instruction + the passage; no `context`
  → byte-identical to the sub-project-2 prompt (regression).
- `chunk_document` — one chunk per non-blank page with `p.<n>` labels; a long page splits into `<=max_chars`
  sub-chunks keeping the label; empty/blank pages skipped; never raises on non-iterable/garbage input.
- `generate_grounded` with a **fake `model_call`** + fake chunks — rows tagged with the right `source`; stops
  at `count`; cross-chunk duplicates flagged (a row repeated in a later chunk is a duplicate); a `model_call`
  that raises → report `error`, no throw; empty chunks → `produced=0`, no throw; `max_chunks` bound honored.
- Library/upload adapters — with an injected fake store / fake extractor: hits → chunks with `"<title>,
  p.<page>"` labels; PDF pages → chunks; unsupported file → clean error.
- `routes/dataset_routes` — TestClient with injected `model_call` + store/extractor: `/generate/grounded`
  (admin, happy, empty-query→error, no-hits→message, never-500) and `/generate/upload` (admin, multipart happy,
  bad file→error, count clamp, never-500, temp cleaned).
- Frontend `dataset.js` — `node --check` + text guards (source selector ids, `/generate/upload` +
  `/generate/grounded` routes, source rendered in staging).
- Manual GUI verification owed (ingest/upload a real manual → Generate grounded → verify rows match the cited
  pages → Add → Save).

## Non-goals (this sub-project)

- Baking citations into saved rows (staging-only source display; deferred toggle).
- Scanned/image-only PDF OCR (text-layer PDFs only, via `_pdf_pages`).
- Per-manual filtering UI / `manual_id` scoping in the card (v1 retrieves across the library by topic).
- Chunk-overlap/sliding-window tuning, semantic chunking, quality/LLM-judge scoring of grounded rows,
  answerability verification (checking each generated row is actually entailed by its source).
- Adding the uploaded document to the RAG (it is extracted for grounding only, not ingested).
