# AI Studio — Dataset Builder + Validator — Design

**Goal:** An admin **Dataset** panel that builds a training JSONL in-app — add rows via a structured
form, import/paste existing data, validate against the training format (per-line errors + stats), and
save a named dataset the Training feature can pick from a dropdown. Sub-project 1 of "AI Studio: dataset
tools" (the fine-tuning/LoRA half of AI Studio already shipped: engine, GUI, serve-adapter, export).

---

## The core idea + one small refactor

The training format is exactly `src/training/dataset.py`'s three JSONL row shapes: `{"text"}`,
`{"instruction","response"}` (optional `"input"`, `"output"` as a `response` fallback), and
`{"prompt","completion"}`. Today `load_jsonl` **raises on the first malformed row** — fine for training,
useless for a validator that must report *every* bad row.

So: extract the per-row rule into a **non-raising** shared function in `dataset.py` —
`normalize_row(row) -> (text: str|None, error: str|None)` — and have `load_jsonl` call it (still raising,
behavior-preserving) while the validator calls it to collect all errors. One source of truth for the rule.

There is no heavy dependency or feasibility risk here (pure Python + a standard frontend panel), so no
feasibility-gate.

## Architecture / components

**Backend (main app, Python 3.14):**
- **`src/training/dataset.py`** (refactor) — add `normalize_row(row) -> (str|None, str|None)` holding the
  existing shape rules (no raise); `load_jsonl` calls it and raises `ValueError(f"line {i}: {error}")` on a
  non-None error (byte-for-byte same behavior + messages as today).
- **`src/dataset_tools/validate.py`** (pure, no I/O) — `validate_rows(rows: list) -> dict` and
  `validate_jsonl_text(text: str) -> dict`, both returning a report:
  `{"total", "valid", "invalid", "errors": [{"line", "message"}], "stats": {"shapes": {"text","instruction","prompt"}, "char_len": {"min","max","avg"}, "approx_tokens"}}`.
  `validate_jsonl_text` also reports invalid-JSON and non-object lines (line-numbered). Reuses
  `normalize_row`. Never raises.
- **`src/dataset_tools/store.py`** (never-raises) — save/list/load/delete JSONL datasets under
  `<DATA_DIR>/training/datasets/`. `save(name, rows) -> {"ok","path"}|{"error"}` (sanitize `name` to a
  safe basename + `.jsonl`, reject `/`, `\`, `..`); `list() -> [{"name","path","rows","size"}]`;
  `load(name) -> {"rows"}|{"error"}`; `delete(name) -> {"ok"}|{"error"}`.
- **`routes/dataset_routes.py`** (admin-gated `APIRouter(prefix="/api/datasets", dependencies=[Depends(require_admin)])`) —
  `POST /validate` (body `{rows}` or `{text}` → report), `POST /` (body `{name, rows}` → save),
  `GET /` (list), `GET /{name}` (load rows), `DELETE /{name}`. Bad input → 400, never 500; wire into `app.py`.

**Frontend (a new admin Dataset modal):**
- **`static/js/datasetCore.js`** (pure) — `ROW_FORMATS` (the three shapes + their fields); `formToRow(format, fields) -> {row|null, error}`; a light `quickValidateRows(rows)` (client-side echo of the shapes) for instant feedback. Node-subprocess unit test.
- **`static/index.html` + `static/js/dataset.js`** — a `#dataset-modal` opened from **both** a sidebar
  Tools entry (`#tool-dataset-btn`) and an icon-rail button (`#rail-dataset`), admin-revealed (the
  both-surfaces pattern). Format selector → structured row-add form → live editable/deletable row list;
  **Import** (upload a `.jsonl` or paste text → parsed into rows, invalid lines flagged); **Validate** →
  render the report (per-line errors + stats); **Save** (name) → the saved-datasets list (load/delete).
  DOM module gets a `node --check` gate; the visual panel is manual-GUI-verification owed.
- **Training integration** — the Training modal's `.jsonl path` input gains a `<datalist>` populated from
  `GET /api/datasets` (saved dataset paths), so a built dataset flows straight into a run.

## Data flow

Build rows (form) or Import (upload/paste → rows) → **Validate** (report: errors + stats) → **Save**(name)
→ `<DATA_DIR>/training/datasets/<name>.jsonl` → appears in the Training dataset datalist → train.

## Error handling

- Validation **never raises** — it collects every bad row (invalid JSON, non-object, wrong shape) with a
  1-based line and a message, alongside stats.
- The store **never raises** — path-safe names (reject traversal), returns `{"error"}` on any failure.
- Routes are admin-gated; malformed request bodies → 400 (not 500). Save with an empty/duplicate/unsafe
  name → a clear 400.

## Testing

Headless:
- `normalize_row` — the three shapes + the failure cases; and `load_jsonl` still raises the same
  line-numbered messages (regression on the refactor).
- `validate_rows` / `validate_jsonl_text` — all-rows error collection (multiple bad rows all reported),
  invalid-JSON + non-object lines, the stats (shape counts, char-length min/max/avg, approx tokens),
  and never-raises on hostile input.
- `store` — save/list/load/delete round-trip; name sanitization + traversal rejection; hermetic under
  `tmp_path`.
- `routes/dataset_routes` — TestClient: admin-gated; validate/save/list/load/delete shapes; bad body → 400.
- `datasetCore.js` — `formToRow` per format + `ROW_FORMATS` via a node subprocess (utf-8); DOM module
  `node --check`.
- Manual GUI verification owed (build → validate → save → appears in Training).

## Scope decisions (baked in)

- **Separate Dataset modal** (not crammed into the busy Training modal); admin-gated (datasets feed the
  admin-only Training feature).
- Datasets are **JSONL files** under `<DATA_DIR>/training/datasets/`; the Training panel references them by
  path via a datalist.
- Structured form + import (no raw-JSONL editor in v1).

## Non-goals (this sub-project)

- **Synthetic data generation** (local LLM generates rows) — sub-project 2.
- Image captioning / labeling / validation (those feed image-LoRA training, which doesn't exist here).
- Dataset split / dedup / augmentation / dedup-by-similarity; train/eval splits; cloud sync.
- A raw-JSONL text editor (deferred; the structured form + import covers building without hand-writing JSON).
