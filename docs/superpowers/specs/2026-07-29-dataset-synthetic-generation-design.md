# AI Studio — Dataset Synthetic Generation — Design

**Goal:** An admin **"Generate (AI)"** panel inside the Dataset Builder modal. The user writes a short
brief, picks a target row count and format, and a locally-served chat model generates candidate training
rows. The rows are validated + deduped server-side and shown in a staging preview; the user merges the ones
they want into the dataset they are building. Sub-project 2 of "AI Studio: dataset tools" (sub-project 1 —
the Builder + Validator + Store — already shipped on dev).

---

## The core idea

Sub-project 1 gave us a validated, path-safe dataset builder whose rule for a valid training row lives in
one place (`normalize_row` / `validate_rows`, the three shapes: `{text}`; `{instruction, response}` +
optional `input`, with `output` as a `response` fallback; `{prompt, completion}`). Synthetic generation is
**"wrap a model call around that validator"**: ask a local LLM to emit rows in the target shape, parse its
(messy, untrusted) output, run every row through the *same* validator, drop duplicates, and stage the
survivors for the user to accept.

The app already exposes a clean, injectable, server-side chat call — the workflow engine's
`default_model_call(prompt, model, system, owner)` (`src/workflows/nodes.py:53`), which resolves the
configured default endpoint via `resolve_endpoint("default")` and POSTs to the OpenAI-compat API. Generation
reuses this exact pattern: an **injectable `model_call`** whose production default is a thin wrapper over
`resolve_endpoint("default")`, so the pure core never touches the network and tests inject a fake model.

There is no heavy dependency here (a model call + JSON parsing + the shipped validator), so no
feasibility-gate.

## Scope decisions (baked in)

- **Generation input = a topic/instructions brief** (natural language), optionally auto-seeded with a few of
  the dataset's existing rows as few-shot style examples. Document-grounding is a later sub-project.
- **Stage & review**, not auto-add: generated rows land in a preview with per-row valid/invalid/duplicate
  status; the user explicitly merges valid rows into the working dataset.
- **Active default model**: generation uses whatever chat model `resolve_endpoint("default")` returns. No
  model configured/served → a clear, actionable error (never a 500). No auto-serve, no per-run model picker.
- **Batching**: to reliably hit larger counts without truncation, the core requests rows in chunks
  (default batch 10) and accumulates validated, unique rows across batches until it reaches the requested
  count, a max-attempts safety bound, or a model error — then reports how many it produced.

## Architecture / components

**Backend (main app, Python 3.14):**

- **`src/dataset_tools/generate.py`** (pure, no I/O, injectable `model_call`, **never raises**):
  - `build_generation_prompt(fmt, count, brief, seed_rows) -> (system: str, user: str)` — constructs the
    generation prompt. The system message pins the role ("You generate high-quality fine-tuning data. Output
    ONLY JSONL — one JSON object per line — no prose, no markdown fences."), the **exact target row shape and
    its required keys** for `fmt`, and the requested batch count; the user message carries the brief and, if
    `seed_rows` is non-empty, a few of them rendered as example lines. Deterministic, string-only.
  - `parse_generated_rows(text) -> list[dict]` — robustly extract row objects from a raw model response:
    strip ```json / ``` fences, accept EITHER a top-level JSON array OR line-delimited JSON objects, skip
    blank/garbage/non-object lines, tolerate a truncated final line. Returns only `dict` rows. Never raises
    (any error → returns what parsed so far, possibly `[]`).
  - `async generate_rows(fmt, count, brief, *, seed_rows=None, existing=None, model_call, validate=validate_rows, batch_size=10, max_attempts=None) -> dict`
    — the batched generation loop. Maintains a `seen` set of canonical row signatures seeded from `existing`.
    Each attempt: `build_generation_prompt` for a chunk → `await model_call(user, system=..., owner=...)` →
    `parse_generated_rows` → for each parsed row, `normalize_row`-validate it and check the signature; accept
    valid, non-duplicate rows (adding their signature to `seen`), record invalid/duplicate ones for the
    report. Loop until `len(accepted) >= count`, attempts reach `max_attempts`
    (default `ceil(count / batch_size) * 3`, capped), or a `model_call` raises (caught → `error` field, stop).
    Returns:
    `{"rows": [{"row", "valid", "error", "duplicate"}...], "valid", "invalid", "duplicates", "requested",
    "produced", "attempts", "error"?}`. The `rows` list holds ALL candidates seen (accepted + rejected) so the
    UI can show ✓/✗/⚠; `produced` = count of accepted valid-unique rows. Never raises. Fixed knobs:
    `batch_size` defaults to 10 (not user-supplied — the route sets it); `max_attempts` defaults to
    `min(ceil(count / batch_size) * 3, 30)` so total model calls are bounded regardless of `count`.

- **`routes/dataset_routes.py`** (extend the shipped admin-gated router) — `POST /api/datasets/generate`:
  body `{fmt|format, count, brief, seed_rows?, existing?}` → clamps `count` to `[1, MAX_GENERATE]`
  (MAX_GENERATE = 200); `batch_size` is fixed server-side (10), not taken from the body → calls
  `generate_rows` with the **default `model_call`** (a module-level `async def _default_model_call` mirroring `workflows.nodes.default_model_call`:
  `resolve_endpoint("default")` + OpenAI-compat, raising `RuntimeError` if no endpoint). A `RuntimeError`
  from an unconfigured/unreachable endpoint is caught and returned as a 400/409-style `{"error"}` report
  (the route degrades to a clear message, **never 500**). Admin-gated by the existing router dependency.

**Frontend (extend the shipped Dataset modal):**

- **`static/js/dataset.js`** (extend) — a **Generate (AI)** card: a `brief` textarea, a `count` number input,
  reusing the modal's existing `#dataset-format` selector. "Generate" disables the button + shows a spinner,
  then POSTs `{format, count, brief, existing: rows, seed_rows: rows.slice(0, 3)}` (up to 3 existing rows as
  few-shot style examples) to `/api/datasets/generate`. The returned report renders a **staging list** (each row: ✓ valid / ✗ error /
  ⚠ duplicate, JSON preview via the existing `esc()`), a summary line ("produced M of N; A attempts"), and an
  **"Add valid rows"** button that concatenates the valid, non-duplicate staged rows into the working `rows[]`
  and calls `renderRows()`. All content HTML-escaped before `innerHTML` (XSS discipline from sub-project 1).
- **`static/index.html`** — the Generate card markup inside `#dataset-modal` (a new `admin-card`); the
  Help-manual "Datasets (AI Studio)" section gains one sentence about AI generation (unique phrase:
  "generate rows with a local model").

## Data flow

brief + count + format (+ working rows as `existing`/`seed_rows`) → `POST /api/datasets/generate` → default
`model_call` (batched, N/batch_size sequential calls) → raw LLM text per batch → **parse → validate
(normalize_row) → dedup** → staged report → user reviews ✓/✗/⚠ → **"Add valid rows"** merges accepted rows
into the Builder → Validate / Save as in sub-project 1.

## Error handling

- `parse_generated_rows` and `generate_rows` **never raise** — garbage/hostile/truncated model output is
  parsed leniently and bad rows are reported, not thrown. A `model_call` exception is caught and surfaced as
  the report's `error` field (generation stops, whatever was accepted so far is returned).
- **No default model configured/served** → the route returns a clear `{"error": "..."}` (surfaced in the
  panel as "Generation unavailable: serve or select a chat model first"), never a 500.
- **Count clamped** server-side to `[1, 200]`; `max_attempts` bounds total model calls so a model that keeps
  producing duplicates/garbage cannot loop forever (returns `produced < requested` with the attempt count).
- **Safety boundary:** every generated row is validated by sub-project 1's `normalize_row` before it can be
  staged as valid, and only user-accepted valid rows enter the dataset — generation can never inject a
  malformed row.

## Testing

Headless:
- `parse_generated_rows` — plain JSONL, ```json-fenced JSONL, a top-level JSON array, mixed prose + JSON,
  a truncated final line, and pure garbage → each returns the right `dict` rows (or `[]`), never raises.
- `build_generation_prompt` — includes the target shape's required keys, the count and brief, and renders
  seed rows when provided; string-only output.
- `generate_rows` with a **fake `model_call`** (canned per-batch outputs): happy path reaches `count`;
  batching loops across multiple calls and accumulates; duplicates (against `existing` and within the batch)
  are flagged not accepted; invalid rows reported; a `model_call` that raises → `error` field, no crash;
  `max_attempts` stops a duplicate-only model with `produced < requested`; hostile model output
  (non-str, arrays of non-objects) never raises.
- `routes/dataset_routes` — TestClient with an injected `model_call`: admin-gated; count clamp; a happy
  generate returns a staged report; model-not-configured → `{"error"}` with a 4xx (not 500).
- `dataset.js` — `node --check` gate + text-guards that the Generate card wires `#dataset-generate*` ids and
  the `/api/datasets/generate` route; `datasetCore.js`/`formToRow` unchanged.
- Manual GUI verification owed (serve a small model → Generate → review staging → Add valid rows → Save).

## Non-goals (this sub-project)

- **Document grounding** (generate from a manual/source doc with citations) — a later sub-project.
- **Streaming / background jobs** — generation is one synchronous request (batched internally) with a
  spinner; no progress stream, no async job queue.
- **Per-run model picker** and **auto-serve** — generation uses the active default endpoint only.
- Quality scoring / LLM-judge filtering of generated rows, augmentation, paraphrase-dedup (only exact-match
  dedup here), train/eval splits, image/caption generation.
