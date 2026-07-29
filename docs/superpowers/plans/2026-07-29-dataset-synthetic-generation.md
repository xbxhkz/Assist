# Dataset Synthetic Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an admin "Generate (AI)" panel to the Dataset Builder that has a locally-served chat model produce candidate training rows, validated + deduped and staged for the user to merge into the dataset they are building.

**Architecture:** A pure, injectable, never-raising core (`src/dataset_tools/generate.py`) builds a generation prompt for the target row shape, parses the model's messy output into rows, validates each through sub-project 1's `normalize_row`, dedups across batches, and loops in chunks until it reaches the requested count or a bounded attempt limit. An admin-gated `POST /api/datasets/generate` route wires the core to a default `model_call` (a thin wrapper over `resolve_endpoint("default")` + the OpenAI-compat API, mirroring `src/workflows/nodes.py:default_model_call`). The frontend adds a Generate card to the existing Dataset modal that stages the results with ✓/✗/⚠ status and an "Add valid rows" button.

**Tech Stack:** Python 3.14 (stdlib `json`/`math` + FastAPI + httpx), ES-module browser JS, pytest (`--import-mode=importlib`), `node --check` for JS syntax.

## Global Constraints

- Main app is Python 3.14 and MUST NEVER import torch/peft/bitsandbytes/transformers/gguf/sentencepiece/trl/datasets/accelerate. `generate.py` is stdlib-only (`json`, `math`) + imports `normalize_row` from `src/training/dataset.py`.
- `parse_generated_rows` and `generate_rows` **NEVER raise** — any bad/hostile/truncated model output or a `model_call` that raises is caught and reported, never thrown.
- The route **NEVER returns 500**: it returns HTTP 200 with a report; a failure (no endpoint, model error) is surfaced in the report's `error` field, which the frontend displays. (This refines the spec's "4xx" wording toward a 200+error report, consistent with the existing `/validate` route.)
- The route is ADMIN-gated by the existing `setup_dataset_routes()` router dependency (`dependencies=[Depends(require_admin)]`) — do not add a second router; extend the shipped one.
- `count` is clamped server-side to `[1, MAX_GENERATE]` where `MAX_GENERATE = 200`. `batch_size` is fixed server-side at 10 (not read from the request body). `max_attempts` defaults to `min(ceil(count / batch_size) * 3, 30)`.
- Every generated row passes through `normalize_row` before it can be staged as valid; only user-accepted valid, non-duplicate rows enter the dataset.
- Frontend must HTML-escape every model/content string via the existing `esc()` before `innerHTML` (XSS discipline).
- Generation reuses the currently-selected `#dataset-format` selector and the module-level `rows[]` in `dataset.js` (as `existing` for dedup and first-3 as `seed_rows`). No new dependencies. No frozen rebuild needed to test.
- pytest `--import-mode=importlib`; commit to `dev`; commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; stage only the named files (NEVER `git add -A`; never stage `installer/Output/Assist-Setup.exe`).

## File Structure

- **Create `src/dataset_tools/generate.py`** — the generation core: `build_generation_prompt`, `parse_generated_rows` (Task 1), `generate_rows` (Task 2). Pure, injectable `model_call`, never raises.
- **Modify `routes/dataset_routes.py`** — add `_default_model_call` + `MAX_GENERATE` + the `POST /generate` endpoint on the existing router (Task 3).
- **Modify `static/index.html`** — the Generate card inside `#dataset-modal` (Task 4); the Datasets Help section sentence (Task 5).
- **Modify `static/js/dataset.js`** — `generate()` / `renderStaging()` / `addGenerated()` + a module-level `staged` array + button wiring (Task 4).
- **Tests:** `tests/test_dataset_generate.py` (Tasks 1+2), `tests/test_dataset_generate_routes.py` (Task 3), extend `tests/test_dataset_ui.py` (Tasks 4+5).

---

### Task 1: Generation prompt + output parser (pure)

**Files:**
- Create: `src/dataset_tools/generate.py`
- Test: `tests/test_dataset_generate.py`

**Interfaces:**
- Produces: `build_generation_prompt(fmt, count, brief, seed_rows=None) -> (system: str, user: str)`; `parse_generated_rows(text) -> list[dict]`. Both pure, never raise.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_generate.py`:

```python
from src.dataset_tools.generate import build_generation_prompt, parse_generated_rows


def test_prompt_includes_shape_count_and_brief():
    system, user = build_generation_prompt("instruction", 5, "about VFD faults", None)
    assert "instruction" in system and "response" in system
    assert "5" in user and "VFD faults" in user
    assert isinstance(system, str) and isinstance(user, str)


def test_prompt_renders_seed_examples_and_survives_bad_seeds():
    system, user = build_generation_prompt("text", 3, "", [{"text": "seed one"}, "not-a-dict", 42])
    assert "seed one" in user  # dict seed rendered; non-dict seeds skipped, no raise


def test_prompt_unknown_format_falls_back_to_text():
    system, user = build_generation_prompt("bogus", 2, "x", None)
    assert '"text"' in system


def test_parse_plain_jsonl():
    rows = parse_generated_rows('{"text": "a"}\n{"text": "b"}')
    assert rows == [{"text": "a"}, {"text": "b"}]


def test_parse_fenced_and_array_and_garbage():
    fenced = parse_generated_rows('```json\n{"text": "a"}\n{"text": "b"}\n```')
    assert fenced == [{"text": "a"}, {"text": "b"}]
    arr = parse_generated_rows('[{"prompt": "p", "completion": "c"}, {"x": 1}]')
    assert arr == [{"prompt": "p", "completion": "c"}, {"x": 1}]
    assert parse_generated_rows("total garbage, no json here") == []


def test_parse_tolerates_prose_and_truncated_tail():
    text = 'Here you go:\n{"text": "a"}\n{"text": "b"}\n{"text": "trунc'
    rows = parse_generated_rows(text)
    assert {"text": "a"} in rows and {"text": "b"} in rows  # truncated last line dropped


def test_parse_never_raises_on_non_str():
    for bad in (None, 42, ["x"], {"a": 1}):
        assert parse_generated_rows(bad) == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_generate.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.dataset_tools.generate'`).

- [ ] **Step 3: Implement**

Create `src/dataset_tools/generate.py`:

```python
"""Synthetic training-row generation: ask a model for rows in the target shape,
parse its (messy, untrusted) output, validate + dedup, stage for review.
Pure + injectable model_call. Never raises."""
import json
import math

from src.training.dataset import normalize_row

_SHAPE_DESC = {
    "text": 'exactly one key "text" (the training text)',
    "instruction": 'keys "instruction" and "response" (optionally also "input")',
    "prompt": 'keys "prompt" and "completion"',
}
_SHAPE_EXAMPLE = {
    "text": '{"text": "..."}',
    "instruction": '{"instruction": "...", "response": "..."}',
    "prompt": '{"prompt": "...", "completion": "..."}',
}


def build_generation_prompt(fmt, count, brief, seed_rows=None):
    """Build (system, user) messages instructing the model to emit JSONL rows of
    the target shape. Deterministic, string-only, never raises."""
    fmt = fmt if fmt in _SHAPE_DESC else "text"
    system = (
        "You generate high-quality fine-tuning data for a language model. "
        "Output ONLY JSONL: one JSON object per line, no prose, no markdown, no code fences. "
        f"Each object MUST have {_SHAPE_DESC[fmt]}. Example line: {_SHAPE_EXAMPLE[fmt]}"
    )
    try:
        n = int(count)
    except Exception:  # noqa: BLE001
        n = 1
    parts = [f"Generate {max(1, n)} diverse, high-quality training examples as JSONL."]
    b = brief if isinstance(brief, str) else ""
    if b.strip():
        parts.append("Topic / instructions: " + b.strip())
    examples = []
    for r in (seed_rows if isinstance(seed_rows, list) else []):
        if isinstance(r, dict):
            try:
                examples.append(json.dumps(r, ensure_ascii=False))
            except Exception:  # noqa: BLE001
                pass
        if len(examples) >= 3:
            break
    if examples:
        parts.append("Match the style and format of these examples:\n" + "\n".join(examples))
    parts.append("Output the JSONL now, one object per line.")
    return system, "\n\n".join(parts)


def parse_generated_rows(text):
    """Extract row dicts from a raw model response. Strips code fences, accepts a
    top-level JSON array OR line-delimited JSON objects, skips garbage/truncated
    lines. Returns only dicts. Never raises."""
    if not isinstance(text, str):
        return []
    s = text.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    # Whole-text JSON (array or single object) first.
    try:
        obj = json.loads(s)
    except Exception:  # noqa: BLE001
        obj = None
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        return [obj]
    # Fall back to line-delimited JSON objects.
    out = []
    for line in s.splitlines():
        line = line.strip().rstrip(",")
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(r, dict):
            out.append(r)
    return out
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_dataset_generate.py --import-mode=importlib -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dataset_tools/generate.py tests/test_dataset_generate.py
git commit -m "feat(dataset-tools): synthetic-gen prompt builder + output parser (pure)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Batched generation loop

**Files:**
- Modify: `src/dataset_tools/generate.py`
- Test: `tests/test_dataset_generate.py` (extend)

**Interfaces:**
- Consumes: `build_generation_prompt`, `parse_generated_rows` (Task 1); `normalize_row` (already imported).
- Produces: `async generate_rows(fmt, count, brief, *, seed_rows=None, existing=None, model_call, batch_size=10, max_attempts=None) -> dict`. `model_call` is `async (prompt, *, system=None) -> str`. Returns `{"rows":[{"row","valid","error","duplicate"}...], "valid", "invalid", "duplicates", "requested", "produced", "attempts", "error"?}`. Never raises.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dataset_generate.py`:

```python
import asyncio
from src.dataset_tools.generate import generate_rows


def _run(coro):
    return asyncio.run(coro)


def test_generate_happy_reaches_count():
    async def fake(prompt, system=None):
        return '{"text": "a"}\n{"text": "b"}'
    rep = _run(generate_rows("text", 2, "brief", model_call=fake, batch_size=10))
    assert rep["produced"] == 2 and rep["valid"] == 2 and rep["requested"] == 2
    assert [c["row"] for c in rep["rows"] if c["valid"] and not c["duplicate"]] == [{"text": "a"}, {"text": "b"}]


def test_generate_batches_across_calls():
    calls = {"n": 0}
    async def fake(prompt, system=None):
        calls["n"] += 1
        return '{"text": "row%d"}' % calls["n"]  # one new row per call
    rep = _run(generate_rows("text", 3, "b", model_call=fake, batch_size=1))
    assert rep["produced"] == 3 and calls["n"] == 3


def test_generate_flags_duplicates_against_existing_and_within_batch():
    async def fake(prompt, system=None):
        return '{"text": "dup"}\n{"text": "dup"}\n{"text": "new"}'
    rep = _run(generate_rows("text", 5, "b", existing=[{"text": "dup"}], model_call=fake, batch_size=10, max_attempts=1))
    assert rep["duplicates"] >= 2 and rep["produced"] == 1
    assert any(c["row"] == {"text": "new"} and c["valid"] and not c["duplicate"] for c in rep["rows"])


def test_generate_reports_invalid_rows():
    async def fake(prompt, system=None):
        return '{"nope": 1}\n{"text": "ok"}'
    rep = _run(generate_rows("text", 5, "b", model_call=fake, batch_size=10, max_attempts=1))
    assert rep["invalid"] >= 1 and rep["produced"] == 1


def test_generate_model_error_is_reported_not_raised():
    async def boom(prompt, system=None):
        raise RuntimeError("no default model endpoint configured")
    rep = _run(generate_rows("text", 3, "b", model_call=boom, batch_size=10))
    assert rep["produced"] == 0 and "error" in rep and "endpoint" in rep["error"]


def test_generate_max_attempts_bounds_a_duplicate_only_model():
    async def fake(prompt, system=None):
        return '{"text": "same"}'
    rep = _run(generate_rows("text", 10, "b", model_call=fake, batch_size=1, max_attempts=4))
    assert rep["produced"] == 1 and rep["attempts"] == 4  # never reaches 10


def test_generate_hostile_model_output_never_raises():
    async def fake(prompt, system=None):
        return 42  # not a string
    rep = _run(generate_rows("text", 3, "b", model_call=fake, batch_size=1, max_attempts=2))
    assert rep["produced"] == 0 and rep["attempts"] == 2
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_generate.py --import-mode=importlib -q`
Expected: FAIL (`cannot import name 'generate_rows'`).

- [ ] **Step 3: Implement**

Append to `src/dataset_tools/generate.py`:

```python
def _sig(row):
    try:
        return json.dumps(row, sort_keys=True, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return repr(row)


async def generate_rows(fmt, count, brief, *, seed_rows=None, existing=None,
                        model_call, batch_size=10, max_attempts=None):
    """Batched synthetic generation. Loops model_call in chunks, validating each
    row via normalize_row and deduping across batches, until it reaches `count`,
    exhausts `max_attempts`, or model_call raises. Never raises."""
    fmt = fmt if fmt in _SHAPE_DESC else "text"
    try:
        count = max(0, int(count))
    except Exception:  # noqa: BLE001
        count = 0
    try:
        batch_size = max(1, int(batch_size))
    except Exception:  # noqa: BLE001
        batch_size = 10
    if max_attempts is None:
        max_attempts = min(math.ceil(count / batch_size) * 3, 30) if count else 0
    seen = set()
    for r in (existing if isinstance(existing, list) else []):
        if isinstance(r, dict):
            seen.add(_sig(r))
    seed_rows = seed_rows if isinstance(seed_rows, list) else []
    candidates, accepted, attempts, err = [], 0, 0, None
    while accepted < count and attempts < max_attempts:
        attempts += 1
        _system, _user = build_generation_prompt(fmt, min(batch_size, count - accepted), brief, seed_rows)
        try:
            raw = await model_call(_user, system=_system)
        except Exception as e:  # noqa: BLE001
            err = f"model call failed: {e}"
            break
        for row in parse_generated_rows(raw if isinstance(raw, str) else ""):
            _text, verr = normalize_row(row)
            if verr:
                candidates.append({"row": row, "valid": False, "error": verr, "duplicate": False})
                continue
            sig = _sig(row)
            if sig in seen:
                candidates.append({"row": row, "valid": True, "error": None, "duplicate": True})
                continue
            seen.add(sig)
            candidates.append({"row": row, "valid": True, "error": None, "duplicate": False})
            accepted += 1
            if accepted >= count:
                break
    report = {
        "rows": candidates,
        "valid": sum(1 for c in candidates if c["valid"] and not c["duplicate"]),
        "invalid": sum(1 for c in candidates if not c["valid"]),
        "duplicates": sum(1 for c in candidates if c["duplicate"]),
        "requested": count,
        "produced": accepted,
        "attempts": attempts,
    }
    if err:
        report["error"] = err
    return report
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_dataset_generate.py --import-mode=importlib -q`
Expected: PASS (14 passed — 7 from Task 1 + 7 new).

- [ ] **Step 5: Commit**

```bash
git add src/dataset_tools/generate.py tests/test_dataset_generate.py
git commit -m "feat(dataset-tools): batched generate_rows loop (validate + dedup, never-raises)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Admin `POST /api/datasets/generate` route

**Files:**
- Modify: `routes/dataset_routes.py`
- Test: `tests/test_dataset_generate_routes.py`

**Interfaces:**
- Consumes: `generate_rows` (Task 2); `resolve_endpoint` (`src/endpoint_resolver.py`, signature `resolve_endpoint(prefix, fallback_url=None, fallback_model=None, fallback_headers=None, owner=None) -> (url, model, headers)`); the existing admin-gated router from `setup_dataset_routes()`.
- Produces: module-level `MAX_GENERATE = 200`, `async _default_model_call(prompt, *, system=None, owner=None) -> str`, and a `POST /api/datasets/generate` endpoint returning the `generate_rows` report (HTTP 200 always; errors in the report's `error` field).

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_generate_routes.py`:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.dataset_routes as dr


def _client(monkeypatch, fake_call):
    monkeypatch.setattr(dr, "require_admin", lambda: None)
    monkeypatch.setattr(dr, "_default_model_call", fake_call)
    app = FastAPI(); app.include_router(dr.setup_dataset_routes())
    return TestClient(app)


def test_generate_happy(monkeypatch):
    async def fake(prompt, system=None, owner=None):
        return '{"text": "a"}\n{"text": "b"}'
    c = _client(monkeypatch, fake)
    r = c.post("/api/datasets/generate", json={"format": "text", "count": 2, "brief": "b"})
    assert r.status_code == 200
    j = r.json()
    assert j["produced"] == 2 and j["requested"] == 2 and "error" not in j


def test_generate_count_clamped(monkeypatch):
    async def fake(prompt, system=None, owner=None):
        return '{"text": "x"}'
    c = _client(monkeypatch, fake)
    r = c.post("/api/datasets/generate", json={"format": "text", "count": 9999, "brief": "b"})
    assert r.status_code == 200 and r.json()["requested"] == 200  # MAX_GENERATE


def test_generate_no_endpoint_is_error_not_500(monkeypatch):
    async def boom(prompt, system=None, owner=None):
        raise RuntimeError("no default model endpoint configured")
    c = _client(monkeypatch, boom)
    r = c.post("/api/datasets/generate", json={"format": "text", "count": 3, "brief": "b"})
    assert r.status_code == 200 and "error" in r.json() and r.json()["produced"] == 0


def test_generate_bad_body_no_500(monkeypatch):
    async def fake(prompt, system=None, owner=None):
        return '{"text": "x"}'
    c = _client(monkeypatch, fake)
    r = c.post("/api/datasets/generate", json={})  # no count/brief/format
    assert r.status_code == 200 and "requested" in r.json()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_generate_routes.py --import-mode=importlib -q`
Expected: FAIL (route + `_default_model_call` missing).

- [ ] **Step 3: Implement**

Edit `routes/dataset_routes.py`. Add these imports at the top (after the existing imports):

```python
from src.dataset_tools.generate import generate_rows

MAX_GENERATE = 200


async def _default_model_call(prompt, *, system=None, owner=None):
    """Call the configured default chat model over the OpenAI-compat API.
    Mirrors src/workflows/nodes.py:default_model_call. Raises if no endpoint."""
    import httpx
    from src.endpoint_resolver import resolve_endpoint
    url, model, headers = resolve_endpoint("default", owner=owner)
    if not url:
        raise RuntimeError("no default model endpoint configured")
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    body = {"model": model, "messages": messages, "stream": False}
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(url, json=body, headers=headers or {})
        resp.raise_for_status()
        data = resp.json()
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
```

Then add this endpoint inside `setup_dataset_routes()`, immediately after the `@router.post("/validate")` block:

```python
    @router.post("/generate")
    async def generate(body: dict = Body(...)):
        fmt = body.get("format") or body.get("fmt") or "text"
        try:
            count = int(body.get("count", 10))
        except (TypeError, ValueError):
            count = 10
        count = max(1, min(count, MAX_GENERATE))
        return await generate_rows(
            fmt, count, body.get("brief", ""),
            seed_rows=body.get("seed_rows"), existing=body.get("existing"),
            model_call=_default_model_call, batch_size=10)
```

Note: `generate_rows` never raises (it catches model errors into the report's `error` field), so this route always returns HTTP 200 with a report — no `HTTPException`, no 500.

- [ ] **Step 4: Run the tests + import smoke**

Run: `python -m pytest tests/test_dataset_generate_routes.py --import-mode=importlib -q`
Expected: PASS (4 passed).
Run: `python -c "import app; print('OK')"`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add routes/dataset_routes.py tests/test_dataset_generate_routes.py
git commit -m "feat(dataset-tools): admin POST /api/datasets/generate route

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Frontend Generate card + controller

**Files:**
- Modify: `static/index.html`
- Modify: `static/js/dataset.js`
- Test: `tests/test_dataset_ui.py` (extend)

**Interfaces:**
- Consumes: `POST /api/datasets/generate` (Task 3); the existing `#dataset-format` selector, module-level `rows[]`, `$`/`esc`/`api`/`renderRows` in `dataset.js`.
- Produces: a Generate card (`#dataset-gen-brief`, `#dataset-gen-count`, `#dataset-generate`, `#dataset-gen-staging`) inside `#dataset-modal`; `generate()`/`renderStaging(rep)`/`addGenerated()` + module-level `staged` in `dataset.js`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dataset_ui.py`:

```python
def test_index_has_generate_card():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in ('id="dataset-gen-brief"', 'id="dataset-gen-count"',
               'id="dataset-generate"', 'id="dataset-gen-staging"'):
        assert el in html, f"{el} missing from index.html"


def test_dataset_js_wires_generate():
    src = (ROOT / "static" / "js" / "dataset.js").read_text(encoding="utf-8")
    for s in ('/api/datasets/generate', 'dataset-generate', 'renderStaging', 'addGenerated', 'staged'):
        assert s in src, f"{s} missing from dataset.js"
```

(The existing `test_dataset_js_syntax` in this file re-runs `node --check` on `dataset.js` and gates the new code's syntax.)

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_ui.py --import-mode=importlib -q`
Expected: FAIL (Generate card + wiring missing).

- [ ] **Step 3: Add the Generate card to `static/index.html`**

Insert this `admin-card` inside `#dataset-modal`, immediately AFTER the Import card (the `admin-card` that ends with the `#dataset-import-btn` button) and BEFORE the Rows card (the one with `#dataset-count`):

```html
      <div class="admin-card">
        <label>Generate with AI (uses the served chat model)<br>
          <textarea id="dataset-gen-brief" rows="2" style="width:100%" placeholder="e.g. concise Q&amp;A about troubleshooting VFD overvoltage faults"></textarea></label>
        <label>How many <input id="dataset-gen-count" type="number" min="1" max="200" value="10" style="width:80px"></label>
        <button id="dataset-generate" class="btn">Generate</button>
        <div id="dataset-gen-staging" style="max-height:200px;overflow:auto;font-size:12px;margin-top:6px"></div>
      </div>
```

- [ ] **Step 4: Add the controller logic to `static/js/dataset.js`**

Add a module-level `staged` array next to `let rows = [];` (line 7):

```javascript
let staged = [];
```

Add these three functions (place them after `importText()`):

```javascript
async function generate() {
  const brief = $('dataset-gen-brief') ? $('dataset-gen-brief').value.trim() : '';
  const count = $('dataset-gen-count') ? (parseInt($('dataset-gen-count').value, 10) || 10) : 10;
  const fmt = $('dataset-format') ? $('dataset-format').value : 'text';
  const btn = $('dataset-generate');
  const out = $('dataset-gen-staging');
  if (btn) { btn.disabled = true; btn.textContent = 'Generating…'; }
  if (out) out.innerHTML = '<div style="opacity:0.6">Generating… this can take a minute.</div>';
  try {
    const rep = await api('/api/datasets/generate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ format: fmt, count: count, brief: brief,
                             existing: rows, seed_rows: rows.slice(0, 3) }),
    });
    staged = rep.rows || [];
    renderStaging(rep);
  } catch (e) {
    if (out) out.textContent = 'Generate failed: ' + e.message;
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Generate'; }
  }
}

function renderStaging(rep) {
  const out = $('dataset-gen-staging'); if (!out) return;
  if (rep.error) { out.innerHTML = '<div style="color:#c00">' + esc(rep.error) + '</div>'; return; }
  const items = (staged || []).map(function (c) {
    const mark = c.valid ? (c.duplicate ? '⚠' : '✓') : '✗';
    const note = c.duplicate ? ' (duplicate)' : (c.error ? ' — ' + esc(c.error) : '');
    return '<div>' + mark + ' ' + esc(JSON.stringify(c.row)).slice(0, 140) + note + '</div>';
  }).join('');
  out.innerHTML = '<div>produced ' + rep.produced + ' of ' + rep.requested +
    ' · ' + rep.attempts + ' attempt(s)</div>' + items +
    '<button class="btn" id="dataset-gen-add">Add valid rows</button>';
  const add = $('dataset-gen-add');
  if (add) add.addEventListener('click', addGenerated);
}

function addGenerated() {
  const good = (staged || []).filter(function (c) { return c.valid && !c.duplicate; })
                             .map(function (c) { return c.row; });
  if (!good.length) { alert('No new valid rows to add.'); return; }
  rows = rows.concat(good); staged = []; renderRows();
  const out = $('dataset-gen-staging'); if (out) out.innerHTML = 'Added ' + good.length + ' row(s).';
}
```

Wire the Generate button in `init()` (add alongside the other button wiring, before the `Modals.register(...)` line):

```javascript
  const gen = $('dataset-generate'); if (gen) gen.addEventListener('click', generate);
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_dataset_ui.py --import-mode=importlib -q`
Expected: PASS (existing tests + the 2 new ones; `test_dataset_js_syntax`'s `node --check` confirms `dataset.js` still parses).

- [ ] **Step 6: Commit**

```bash
git add static/index.html static/js/dataset.js tests/test_dataset_ui.py
git commit -m "feat(dataset-tools): AI Generate card + staging controller

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Finalize — Help section + full suite

**Files:**
- Modify: `static/index.html` (Datasets Help section)
- Test: `tests/test_dataset_ui.py` (extend); run the whole dataset suite + `import app`

- [ ] **Step 1: Add the Help entry test**

Append to `tests/test_dataset_ui.py`:

```python
def test_help_mentions_ai_generation():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert "generate rows with a local model" in html
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_ui.py::test_help_mentions_ai_generation --import-mode=importlib -q`
Expected: FAIL (phrase not present yet).

- [ ] **Step 3: Add the sentence to the Datasets Help section**

In `static/index.html`, find the Help-manual `Datasets (AI Studio)` `<details>` block (its `<summary>` is "Datasets (AI Studio)"). Immediately after its existing `<p>...train on it directly.</p>` paragraph, add this second paragraph inside the same `<details>`:

```html
            <p><b>Generate with AI.</b> Instead of typing every row, describe what you want and let a served chat model <b>generate rows with a local model</b> — pick a count and format, click <b>Generate</b>, review the staged rows (valid / invalid / duplicate), and <b>Add valid rows</b> to the dataset. Serve or select a chat model first.</p>
```

- [ ] **Step 4: Run the full dataset suite + import smoke**

Run: `python -m pytest tests/test_dataset_generate.py tests/test_dataset_generate_routes.py tests/test_dataset_normalize.py tests/test_dataset_validate.py tests/test_dataset_store.py tests/test_dataset_routes.py tests/test_dataset_core_js.py tests/test_dataset_ui.py --import-mode=importlib -q`
Expected: PASS (all green).
Run: `python -c "import app; print('OK')"`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add static/index.html tests/test_dataset_ui.py
git commit -m "docs(dataset-tools): Help note for AI generation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Never-raises is load-bearing:** `parse_generated_rows` and `generate_rows` must return a value on ANY input (non-str model output, a `model_call` that raises, garbage/truncated JSON). Tests cover these — do not remove them.
- **Never 500:** the route returns HTTP 200 with a report; failures ride in the report's `error` field. There is no `HTTPException` in the generate route.
- **Admin-gated:** extend the existing `setup_dataset_routes()` router (already `Depends(require_admin)`); do not create a second router. The route tests monkeypatch `require_admin` to a no-op (sibling convention) — the gate is proven by the router construction, verify by reading.
- **No heavy-dep leak / no rebuild:** `generate.py` is stdlib + `normalize_row`; the route adds `httpx` (already bundled) + `resolve_endpoint`. Frontend + Python only — no frozen rebuild needed to test; the feature reaches users on the next installer rebuild.
- **Manual GUI verification owed** by the user: serve a small chat model → open Dataset → write a brief → Generate → review staging → Add valid rows → Save.
- **Scope:** brief-prompt generation + auto few-shot from existing rows only. NO document grounding, NO streaming/async jobs, NO per-run model picker, NO auto-serve (all deferred).
```
