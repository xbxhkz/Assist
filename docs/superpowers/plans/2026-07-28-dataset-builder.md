# AI Studio — Dataset Builder + Validator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An admin Dataset panel to build a training JSONL in-app — structured row form + import, validate against the training format (per-line errors + stats), and save a named dataset the Training feature picks from a datalist.

**Architecture:** Reuse the training format's row rule (a small non-raising refactor of `dataset.py` so validation and training share one source of truth); pure validator + a path-safe file store under the data dir; admin routes; a new Dataset modal mirroring the Training panel's admin-modal pattern.

**Tech Stack:** Python 3.14 (stdlib only: `json`, `os`, `re`); FastAPI; vanilla ES-module frontend.

## Global Constraints

- **No feasibility risk** (pure Python + a standard frontend panel) — no feasibility-gate task.
- **The `dataset.py` refactor is behavior-preserving:** `load_jsonl` must keep raising the exact same
  line-numbered `ValueError` messages (the existing `tests/test_training_dataset.py` must stay green).
- **Validator + store NEVER raise** — the validator collects every bad row into an `errors` list; the store
  returns `{"error": …}`. Routes are **admin-gated**; malformed input → 400, never 500.
- **Path-safety:** dataset names are sanitized to a safe basename (no `/`, `\`, `..`); datasets live under
  `<DATA_DIR>/training/datasets/`.
- **Admin GUI mirrors the Training panel:** the Dataset modal is revealed via `isAdmin()` from BOTH a
  sidebar Tools entry (`#tool-dataset-btn`) AND an icon-rail button (`#rail-dataset`), and registers with
  `Modals.register(id, {railBtnId, sidebarBtnId, closeFn})` (the both-surfaces pattern).
- **The report shape is fixed** (produced by the validator, consumed by the GUI):
  `{"total", "valid", "invalid", "errors":[{"line","message"}], "stats":{"shapes":{"text","instruction","prompt"},"char_len":{"min","max","avg"},"approx_tokens"}}`.
- pytest `--import-mode=importlib`. Node on PATH for JS tests. Commit directly to `dev`; messages end with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Stage only the files each task names — never
  `git add -A`; never stage `installer/Output/Assist-Setup.exe`, `assistappicon.png`, `assistlogo.png`, `build_assets/**`.
- ~220 unrelated pre-existing failures exist elsewhere — run only the test files each task names. (A
  pre-existing SQLAlchemy `MovedIn20Warning` at collection is known.)

### Verified anchors
- `src/training/dataset.py` — `_normalize(row)->str` (raises) at lines 8-28; `load_jsonl` at 31-53 (its
  json-decode + non-dict + `_normalize` blocks number errors as `f"line {i}: …"`).
- `app.py:768-769` — training router include pair (add the dataset router include right after).
- `routes/training_routes.py` — admin router pattern `APIRouter(prefix=…, dependencies=[Depends(require_admin)])`, `require_admin` from `core.middleware`.
- `static/index.html:487` — `<input id="training-dataset" …>` (gets the datalist); `#rail-training` (rail button, index.html ~783); `#tool-training-btn` (sidebar Tools entry); the Training modal `#training-modal` (insert the dataset modal near it). `static/js/training.js` — `openTraining()` (add the datalist fetch), the `isAdmin()`/`Modals.register` reveal pattern.
- `src/constants.py:12 DATA_DIR`.

---

### Task 1: Refactor `dataset.py` — extract non-raising `normalize_row`

**Files:**
- Modify: `src/training/dataset.py`
- Test: `tests/test_dataset_normalize.py` (new); `tests/test_training_dataset.py` (must still pass)

**Interfaces:**
- Produces: `normalize_row(row) -> (text: str|None, error: str|None)` — the single source of truth for the
  accepted row shapes; never raises. `load_jsonl` unchanged externally (still raises the same messages).

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_normalize.py`:

```python
from src.training.dataset import normalize_row


def test_text_row():
    assert normalize_row({"text": "hi"}) == ("hi", None)


def test_instruction_response():
    t, e = normalize_row({"instruction": "greet", "response": "hi"})
    assert e is None and t == "greet\nhi"


def test_instruction_with_input_and_output_fallback():
    t, e = normalize_row({"instruction": "sum", "input": "1+1", "output": "2"})
    assert e is None and t == "sum\n1+1\n2"


def test_prompt_completion():
    assert normalize_row({"prompt": "Q", "completion": "A"}) == ("Q\nA", None)


def test_missing_companion_is_error_not_raise():
    t, e = normalize_row({"instruction": "x"})
    assert t is None and "response" in e


def test_unknown_and_nondict():
    assert normalize_row({"nope": 1})[0] is None
    assert normalize_row("junk")[0] is None
    assert normalize_row(None)[1] is not None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_normalize.py --import-mode=importlib -q`
Expected: FAIL (`cannot import name 'normalize_row'`).

- [ ] **Step 3: Refactor**

Replace the `_normalize` function in `src/training/dataset.py` with a non-raising `normalize_row`:

```python
def normalize_row(row):
    """Normalize one dataset row to its training text. Returns (text, None) on
    success or (None, error_message) on a bad/unrecognized row. NEVER raises —
    the single source of truth for the accepted row shapes, shared by load_jsonl
    (which raises) and the dataset validator (which collects errors)."""
    if not isinstance(row, dict):
        return None, "each row must be a JSON object"
    text = row.get("text")
    if isinstance(text, str) and text.strip():
        return text, None
    instr = row.get("instruction")
    if isinstance(instr, str) and instr.strip():
        resp = row.get("response")
        if not (isinstance(resp, str) and resp.strip()):
            resp = row.get("output")
        if not (isinstance(resp, str) and resp.strip()):
            return None, "'instruction' row needs a non-empty string 'response' (or 'output')"
        inp = row.get("input")
        head = f"{instr}\n{inp}" if isinstance(inp, str) and inp.strip() else instr
        return f"{head}\n{resp}", None
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        comp = row.get("completion")
        if not (isinstance(comp, str) and comp.strip()):
            return None, "'prompt' row needs a non-empty string 'completion'"
        return f"{prompt}\n{comp}", None
    return None, "row needs 'text', or 'instruction'(+response), or 'prompt'(+completion)"
```

In `load_jsonl`, replace the non-dict check + `try: text = _normalize(obj) …` block with a `normalize_row`
call (behavior-preserving — same messages):

```python
            text, err = normalize_row(obj)
            if err:
                raise ValueError(f"line {i}: {err}")
            rows.append({"text": text})
```

(Remove the now-redundant explicit `if not isinstance(obj, dict): raise …` line — `normalize_row` returns
that exact message, so `f"line {i}: each row must be a JSON object"` is still produced.)

- [ ] **Step 4: Run the tests (new + regression)**

Run: `python -m pytest tests/test_dataset_normalize.py tests/test_training_dataset.py --import-mode=importlib -q`
Expected: PASS (new normalize tests + all existing dataset tests still green — proves the refactor preserved `load_jsonl`).

- [ ] **Step 5: Commit**

```bash
git add src/training/dataset.py tests/test_dataset_normalize.py
git commit -m "refactor(dataset): extract non-raising normalize_row (shared rule)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Dataset validator

**Files:**
- Create: `src/dataset_tools/__init__.py` (empty), `src/dataset_tools/validate.py`
- Test: `tests/test_dataset_validate.py`

**Interfaces:**
- Consumes: `normalize_row` (Task 1).
- Produces: `validate_rows(rows: list) -> dict` and `validate_jsonl_text(text: str) -> dict`, both returning
  the report shape from Global Constraints. Never raise.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_validate.py`:

```python
from src.dataset_tools.validate import validate_rows, validate_jsonl_text


def test_all_valid_stats():
    rep = validate_rows([{"text": "hello"}, {"instruction": "a", "response": "bb"}])
    assert rep["total"] == 2 and rep["valid"] == 2 and rep["invalid"] == 0
    assert rep["stats"]["shapes"] == {"text": 1, "instruction": 1, "prompt": 0}
    assert rep["stats"]["char_len"]["max"] >= 4 and rep["stats"]["approx_tokens"] >= 1


def test_collects_every_bad_row():
    rep = validate_rows([{"text": "ok"}, {"nope": 1}, {"instruction": "x"}])
    assert rep["valid"] == 1 and rep["invalid"] == 2
    assert [e["line"] for e in rep["errors"]] == [2, 3]


def test_text_reports_invalid_json_and_line_numbers():
    rep = validate_jsonl_text('{"text": "ok"}\n\n{not json\n{"prompt":"Q","completion":"A"}')
    assert rep["valid"] == 2 and rep["invalid"] == 1
    assert rep["errors"][0]["line"] == 3 and "JSON" in rep["errors"][0]["message"]


def test_never_raises_on_hostile():
    assert validate_rows("junk")["invalid"] >= 0
    assert validate_rows(None)["total"] == 0
    assert validate_jsonl_text(None)["total"] == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_validate.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.dataset_tools.validate'`).

- [ ] **Step 3: Implement**

Create `src/dataset_tools/__init__.py` (empty).

Create `src/dataset_tools/validate.py`:

```python
"""Validate a training dataset (all rows, non-raising) + compute stats. Pure."""
import json

from src.training.dataset import normalize_row


def _shape_of(row):
    if isinstance(row, dict):
        for key in ("text", "instruction", "prompt"):
            v = row.get(key)
            if isinstance(v, str) and v.strip():
                return key
    return "unknown"


def _len_stats(texts):
    if not texts:
        return {"char_len": {"min": 0, "max": 0, "avg": 0}, "approx_tokens": 0}
    lens = [len(t) for t in texts]
    return {"char_len": {"min": min(lens), "max": max(lens), "avg": round(sum(lens) / len(lens), 1)},
            "approx_tokens": round(sum(lens) / 4)}


def _report(total, texts, errors):
    shapes = {"text": 0, "instruction": 0, "prompt": 0}
    return {"total": total, "valid": len(texts), "invalid": len(errors),
            "errors": errors, "stats": {**_len_stats(texts), "shapes": shapes}}


def validate_rows(rows) -> dict:
    """Validate a list of already-parsed row dicts (numbered by position)."""
    if not isinstance(rows, list):
        return {"total": 0, "valid": 0, "invalid": 0,
                "errors": [{"line": 0, "message": "rows must be a list"}],
                "stats": _len_stats([]) | {"shapes": {"text": 0, "instruction": 0, "prompt": 0}}}
    errors, texts = [], []
    shapes = {"text": 0, "instruction": 0, "prompt": 0}
    for i, row in enumerate(rows, start=1):
        text, err = normalize_row(row)
        if err:
            errors.append({"line": i, "message": err})
            continue
        texts.append(text)
        s = _shape_of(row)
        if s in shapes:
            shapes[s] += 1
    return {"total": len(rows), "valid": len(texts), "invalid": len(errors),
            "errors": errors, "stats": {**_len_stats(texts), "shapes": shapes}}


def validate_jsonl_text(text) -> dict:
    """Validate raw JSONL text (numbered by source line; reports invalid JSON)."""
    errors, texts = [], []
    shapes = {"text": 0, "instruction": 0, "prompt": 0}
    total = 0
    for i, line in enumerate((text or "").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            obj = json.loads(line)
        except Exception as e:  # noqa: BLE001
            errors.append({"line": i, "message": f"invalid JSON ({e})"})
            continue
        t, err = normalize_row(obj)
        if err:
            errors.append({"line": i, "message": err})
            continue
        texts.append(t)
        s = _shape_of(obj)
        if s in shapes:
            shapes[s] += 1
    return {"total": total, "valid": len(texts), "invalid": len(errors),
            "errors": errors, "stats": {**_len_stats(texts), "shapes": shapes}}
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_dataset_validate.py --import-mode=importlib -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dataset_tools/__init__.py src/dataset_tools/validate.py tests/test_dataset_validate.py
git commit -m "feat(dataset-tools): all-rows validator + stats (never-raises)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Dataset store

**Files:**
- Create: `src/dataset_tools/store.py`
- Test: `tests/test_dataset_store.py`

**Interfaces:**
- Produces: `DatasetStore(base_dir=None)` with `save(name, rows) -> {"ok","path","name"}|{"error"}`,
  `list() -> [{"name","path","rows","size"}]`, `load(name) -> {"rows","name","path"}|{"error"}`,
  `delete(name) -> {"ok"}|{"error"}`; `get_dataset_store()` singleton. Never raises. Path-safe.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_store.py`:

```python
from src.dataset_tools.store import DatasetStore


def test_save_load_list_delete(tmp_path):
    s = DatasetStore(base_dir=str(tmp_path))
    out = s.save("my set", [{"text": "a"}, {"text": "b"}])
    assert out.get("ok") and out["name"] == "my-set" and out["path"].endswith("my-set.jsonl")
    lst = s.list()
    assert lst and lst[0]["name"] == "my-set" and lst[0]["rows"] == 2
    loaded = s.load("my-set")
    assert loaded["rows"] == [{"text": "a"}, {"text": "b"}]
    assert s.delete("my-set").get("ok") and s.list() == []


def test_name_sanitized_no_traversal(tmp_path):
    s = DatasetStore(base_dir=str(tmp_path))
    out = s.save("../../evil", [{"text": "x"}])
    # written inside base_dir under a sanitized basename, never escaping
    assert out.get("ok") and str(tmp_path) in out["path"] and ".." not in out["name"]


def test_empty_rows_and_bad_name_rejected(tmp_path):
    s = DatasetStore(base_dir=str(tmp_path))
    assert "error" in s.save("ok", [])
    assert "error" in s.save("", [{"text": "x"}])


def test_load_missing(tmp_path):
    assert "error" in DatasetStore(base_dir=str(tmp_path)).load("nope")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_store.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.dataset_tools.store'`).

- [ ] **Step 3: Implement**

Create `src/dataset_tools/store.py`:

```python
"""Save/list/load/delete training datasets as JSONL under <DATA_DIR>/training/
datasets/. Path-safe (sanitized names, no traversal). Never raises."""
import json
import os
import re


def _default_dir():
    from src.constants import DATA_DIR
    return os.path.join(DATA_DIR, "training", "datasets")


def _safe_name(name):
    base = os.path.basename(str(name or "").strip())
    base = re.sub(r"\.jsonl$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-.")
    return base


class DatasetStore:
    def __init__(self, base_dir=None):
        self._dir = base_dir or _default_dir()

    def _path(self, name):
        safe = _safe_name(name)
        return safe, (os.path.join(self._dir, safe + ".jsonl") if safe else None)

    def save(self, name, rows) -> dict:
        try:
            safe, path = self._path(name)
            if not safe:
                return {"error": "invalid dataset name"}
            if not isinstance(rows, list) or not rows:
                return {"error": "rows must be a non-empty list"}
            os.makedirs(self._dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            return {"ok": True, "path": path, "name": safe}
        except Exception as e:  # noqa: BLE001
            return {"error": f"save failed: {e}"}

    def list(self) -> list:
        out = []
        try:
            if not os.path.isdir(self._dir):
                return out
            for fn in sorted(os.listdir(self._dir)):
                if not fn.lower().endswith(".jsonl"):
                    continue
                p = os.path.join(self._dir, fn)
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        n = sum(1 for ln in f if ln.strip())
                    out.append({"name": fn[:-6], "path": p, "rows": n, "size": os.path.getsize(p)})
                except Exception:
                    pass
        except Exception:
            pass
        return out

    def load(self, name) -> dict:
        try:
            safe, path = self._path(name)
            if not path or not os.path.isfile(path):
                return {"error": "dataset not found"}
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        rows.append(json.loads(ln))
                    except Exception:
                        pass
            return {"rows": rows, "name": safe, "path": path}
        except Exception as e:  # noqa: BLE001
            return {"error": f"load failed: {e}"}

    def delete(self, name) -> dict:
        try:
            safe, path = self._path(name)
            if path and os.path.isfile(path):
                os.remove(path)
                return {"ok": True}
            return {"error": "dataset not found"}
        except Exception as e:  # noqa: BLE001
            return {"error": f"delete failed: {e}"}


_store = None


def get_dataset_store():
    global _store
    if _store is None:
        _store = DatasetStore()
    return _store
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_dataset_store.py --import-mode=importlib -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dataset_tools/store.py tests/test_dataset_store.py
git commit -m "feat(dataset-tools): path-safe dataset store (never-raises)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Admin routes + app wiring

**Files:**
- Create: `routes/dataset_routes.py`
- Modify: `app.py`
- Test: `tests/test_dataset_routes.py`

**Interfaces:**
- Consumes: `validate_rows`/`validate_jsonl_text` (Task 2), `get_dataset_store` (Task 3), `require_admin`.
- Produces: `setup_dataset_routes() -> APIRouter` at `/api/datasets`, admin-gated. Endpoints:
  `POST /validate`, `POST ` (save), `GET ` (list), `GET /{name}`, `DELETE /{name}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_routes.py`:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.dataset_routes as dr


class FakeStore:
    def __init__(self): self.saved = None
    def save(self, name, rows): self.saved = (name, rows); return {"ok": True, "path": "p/x.jsonl", "name": "x"}
    def list(self): return [{"name": "x", "path": "p/x.jsonl", "rows": 2, "size": 20}]
    def load(self, name): return {"rows": [{"text": "a"}], "name": name, "path": "p"}
    def delete(self, name): return {"ok": True}


def _client(monkeypatch, store=None):
    monkeypatch.setattr(dr, "require_admin", lambda: None)
    monkeypatch.setattr(dr, "get_dataset_store", lambda: store or FakeStore())
    app = FastAPI(); app.include_router(dr.setup_dataset_routes())
    return TestClient(app)


def test_validate_rows(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/api/datasets/validate", json={"rows": [{"text": "hi"}, {"nope": 1}]})
    assert r.status_code == 200 and r.json()["valid"] == 1 and r.json()["invalid"] == 1


def test_validate_text(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/api/datasets/validate", json={"text": '{"text":"hi"}\n{bad'})
    assert r.status_code == 200 and r.json()["invalid"] == 1


def test_save_list_load_delete(monkeypatch):
    store = FakeStore()
    c = _client(monkeypatch, store)
    assert c.post("/api/datasets", json={"name": "x", "rows": [{"text": "a"}]}).status_code == 200
    assert store.saved[0] == "x"
    assert c.get("/api/datasets").json()["datasets"][0]["name"] == "x"
    assert c.get("/api/datasets/x").json()["rows"] == [{"text": "a"}]
    assert c.request("DELETE", "/api/datasets/x").status_code == 200


def test_save_error_is_400(monkeypatch):
    class Bad(FakeStore):
        def save(self, name, rows): return {"error": "invalid dataset name"}
    r = _client(monkeypatch, Bad()).post("/api/datasets", json={"name": "", "rows": []})
    assert r.status_code == 400
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_routes.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'routes.dataset_routes'`).

- [ ] **Step 3: Implement**

Create `routes/dataset_routes.py`:

```python
"""Admin-gated Dataset builder/validator API (AI Studio)."""
from fastapi import APIRouter, Body, Depends, HTTPException

from core.middleware import require_admin
from src.dataset_tools.validate import validate_rows, validate_jsonl_text
from src.dataset_tools.store import get_dataset_store


def setup_dataset_routes() -> APIRouter:
    router = APIRouter(prefix="/api/datasets",
                       dependencies=[Depends(require_admin)])

    @router.post("/validate")
    async def validate(body: dict = Body(...)):
        if isinstance(body.get("text"), str):
            return validate_jsonl_text(body["text"])
        return validate_rows(body.get("rows", []))

    @router.post("")
    async def save(body: dict = Body(...)):
        out = get_dataset_store().save(body.get("name"), body.get("rows", []))
        if "error" in out:
            raise HTTPException(400, out["error"])
        return out

    @router.get("")
    async def list_datasets():
        return {"datasets": get_dataset_store().list()}

    @router.get("/{name}")
    async def load(name: str):
        out = get_dataset_store().load(name)
        if "error" in out:
            raise HTTPException(404, out["error"])
        return out

    @router.delete("/{name}")
    async def delete(name: str):
        out = get_dataset_store().delete(name)
        if "error" in out:
            raise HTTPException(404, out["error"])
        return out

    return router
```

Wire into `app.py` after the training router include (`app.py:768-769`):

```python
from routes.dataset_routes import setup_dataset_routes
app.include_router(setup_dataset_routes())
```

- [ ] **Step 4: Run the tests + import smoke**

Run: `python -m pytest tests/test_dataset_routes.py --import-mode=importlib -q`
Expected: PASS (4 passed).
Run: `python -c "import app"`
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add routes/dataset_routes.py app.py tests/test_dataset_routes.py
git commit -m "feat(dataset-tools): admin /api/datasets routes + app wiring

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `datasetCore.js` — pure form→row

**Files:**
- Create: `static/js/datasetCore.js`
- Test: `tests/test_dataset_core_js.py`

**Interfaces:**
- Produces (ESM): `ROW_FORMATS` (`{text:['text'], instruction:['instruction','input','response'], prompt:['prompt','completion']}`); `formToRow(format, fields) -> {row: object|null, error: string|null}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_core_js.py`:

```python
import json, subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "static" / "js" / "datasetCore.js"


def _node(expr):
    s = f"import * as m from {json.dumps(CORE.as_uri())};\n{expr}"
    p = subprocess.run(["node", "--input-type=module", "-e", s],
                       capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def test_formats():
    out = _node("console.log(JSON.stringify(Object.keys(m.ROW_FORMATS)));")
    assert set(json.loads(out)) == {"text", "instruction", "prompt"}


def test_form_to_row():
    out = _node("console.log(JSON.stringify(["
                "m.formToRow('text', {text:'hi'}),"
                "m.formToRow('instruction', {instruction:'a'}),"
                "m.formToRow('prompt', {prompt:'Q', completion:'A'})]));")
    a = json.loads(out)
    assert a[0]["row"] == {"text": "hi"} and a[0]["error"] is None
    assert a[1]["row"] is None and "response" in a[1]["error"]
    assert a[2]["row"] == {"prompt": "Q", "completion": "A"}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_core_js.py --import-mode=importlib -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

Create `static/js/datasetCore.js`:

```javascript
// Pure helpers for the Dataset builder — no DOM. Mirrors the training row shapes.
export const ROW_FORMATS = {
  text: ['text'],
  instruction: ['instruction', 'input', 'response'],
  prompt: ['prompt', 'completion'],
};

export function formToRow(format, fields) {
  const f = fields || {};
  const g = function (k) { return (f[k] || '').trim(); };
  if (format === 'text') {
    return g('text') ? { row: { text: g('text') }, error: null } : { row: null, error: 'text is required' };
  }
  if (format === 'instruction') {
    if (!g('instruction')) return { row: null, error: 'instruction is required' };
    if (!g('response')) return { row: null, error: 'response is required' };
    const row = { instruction: g('instruction'), response: g('response') };
    if (g('input')) row.input = g('input');
    return { row: row, error: null };
  }
  if (format === 'prompt') {
    if (!g('prompt')) return { row: null, error: 'prompt is required' };
    if (!g('completion')) return { row: null, error: 'completion is required' };
    return { row: { prompt: g('prompt'), completion: g('completion') }, error: null };
  }
  return { row: null, error: 'unknown format' };
}
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_dataset_core_js.py --import-mode=importlib -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add static/js/datasetCore.js tests/test_dataset_core_js.py
git commit -m "feat(dataset-tools): pure JS core (form->row)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Dataset modal + controller + Training datalist

**Files:**
- Modify: `static/index.html` (Dataset modal, `#rail-dataset`, `#tool-dataset-btn`, modulepreload + script, `#training-dataset` datalist)
- Create: `static/js/dataset.js`
- Modify: `static/js/training.js` (populate the dataset datalist on open)
- Test: `tests/test_dataset_ui.py`

**Interfaces:**
- Consumes: `datasetCore.js` (Task 5); `/api/datasets*` (Task 4); `modalManager.js`.
- Produces: a `#dataset-modal`, `#rail-dataset` + `#tool-dataset-btn` (admin-revealed), `static/js/dataset.js`; the Training dataset input gains `list="dataset-suggestions"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_ui.py`:

```python
import pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_index_has_dataset_modal_and_entries():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in ('id="dataset-modal"', 'id="rail-dataset"', 'id="tool-dataset-btn"',
               '/static/js/dataset.js', '/static/js/datasetCore.js',
               'id="dataset-suggestions"', 'list="dataset-suggestions"'):
        assert el in html, f"{el} missing from index.html"


def test_dataset_js_wires_admin_and_routes():
    src = (ROOT / "static" / "js" / "dataset.js").read_text(encoding="utf-8")
    for s in ('rail-dataset', 'tool-dataset-btn', '/api/datasets/validate', '/api/datasets',
              'isAdmin', 'Modals.register'):
        assert s in src, f"{s} missing from dataset.js"


def test_dataset_js_syntax(tmp_path):
    src = (ROOT / "static" / "js" / "dataset.js").read_text(encoding="utf-8")
    mjs = tmp_path / "dataset.mjs"; mjs.write_text(src, encoding="utf-8")
    p = subprocess.run(["node", "--check", str(mjs)], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr


def test_training_js_populates_dataset_datalist():
    src = (ROOT / "static" / "js" / "training.js").read_text(encoding="utf-8")
    assert "/api/datasets" in src and "dataset-suggestions" in src
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_ui.py --import-mode=importlib -q`
Expected: FAIL (elements + `dataset.js` missing).

- [ ] **Step 3: Create `static/js/dataset.js`**

Create `static/js/dataset.js`:

```javascript
// Dataset builder/validator panel (AI Studio). ES module. Admin-only: the entries
// stay hidden unless /api/auth/status reports is_admin. Mirrors training.js.
import * as Modals from './modalManager.js';
import { ROW_FORMATS, formToRow } from './datasetCore.js';

function $(id) { return document.getElementById(id); }
let rows = [];

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
  });
}

async function api(path, opts) {
  const res = await fetch(path, Object.assign({ credentials: 'same-origin' }, opts || {}));
  const data = await res.json().catch(() => ({}));
  if (!res.ok) { const d = data && data.detail; throw new Error(d || String(res.status)); }
  return data;
}

async function isAdmin() {
  try { const d = await (await fetch('/api/auth/status', { credentials: 'same-origin' })).json(); return !!d.is_admin; }
  catch (e) { return false; }
}

function openDataset() { $('dataset-modal').classList.remove('hidden'); renderFields(); renderRows(); refreshSaved(); }
function closeDataset() { $('dataset-modal').classList.add('hidden'); }

function renderFields() {
  const fmt = $('dataset-format') ? $('dataset-format').value : 'text';
  const host = $('dataset-fields'); if (!host) return;
  host.innerHTML = (ROW_FORMATS[fmt] || []).map(function (k) {
    return '<label>' + k + '<br><textarea data-field="' + k + '" rows="2" style="width:100%"></textarea></label>';
  }).join('');
}

function renderRows() {
  const host = $('dataset-rows'); if (!host) return;
  host.innerHTML = rows.length
    ? rows.map(function (r, i) {
        return '<div>' + (i + 1) + '. ' + esc(JSON.stringify(r)).slice(0, 160) +
               ' <button class="btn" data-del="' + i + '">✕</button></div>';
      }).join('')
    : '<div style="opacity:0.6">No rows yet.</div>';
  host.querySelectorAll('[data-del]').forEach(function (b) {
    b.addEventListener('click', function () { rows.splice(parseInt(b.getAttribute('data-del'), 10), 1); renderRows(); });
  });
  const c = $('dataset-count'); if (c) c.textContent = rows.length + ' row(s)';
}

function addRow() {
  const fmt = $('dataset-format').value;
  const fields = {};
  $('dataset-fields').querySelectorAll('[data-field]').forEach(function (el) { fields[el.getAttribute('data-field')] = el.value; });
  const out = formToRow(fmt, fields);
  if (out.error) { alert(out.error); return; }
  rows.push(out.row);
  $('dataset-fields').querySelectorAll('[data-field]').forEach(function (el) { el.value = ''; });
  renderRows();
}

function importText() {
  const ta = $('dataset-import'); if (!ta) return;
  const added = [];
  ta.value.split('\n').forEach(function (line) {
    line = line.trim(); if (!line) return;
    try { added.push(JSON.parse(line)); } catch (e) { /* skip bad line */ }
  });
  if (!added.length) { alert('No valid JSON lines found.'); return; }
  rows = rows.concat(added); ta.value = ''; renderRows();
}

async function validate() {
  const out = $('dataset-report');
  try {
    const rep = await api('/api/datasets/validate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ rows: rows }),
    });
    const errs = (rep.errors || []).slice(0, 20).map(function (e) { return 'line ' + e.line + ': ' + esc(e.message); }).join('<br>');
    if (out) out.innerHTML = 'valid ' + rep.valid + ' / ' + rep.total + ' · shapes ' +
      esc(JSON.stringify(rep.stats.shapes)) + ' · ~' + rep.stats.approx_tokens + ' tokens' +
      (errs ? '<br>' + errs : '');
  } catch (e) { if (out) out.textContent = 'Validate failed: ' + e.message; }
}

async function save() {
  const name = $('dataset-name') ? $('dataset-name').value.trim() : '';
  if (!name) { alert('Enter a dataset name.'); return; }
  try {
    const r = await api('/api/datasets', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: name, rows: rows }),
    });
    alert('Saved: ' + r.path); refreshSaved();
  } catch (e) { alert('Save failed: ' + e.message); }
}

async function refreshSaved() {
  const host = $('dataset-saved'); if (!host) return;
  try {
    const j = await api('/api/datasets');
    host.innerHTML = (j.datasets || []).map(function (d) {
      return '<div>' + esc(d.name) + ' (' + d.rows + ' rows) ' +
             '<button class="btn" data-load="' + esc(d.name) + '">Load</button>' +
             '<button class="btn" data-delds="' + esc(d.name) + '">Delete</button></div>';
    }).join('') || 'None yet.';
    host.querySelectorAll('[data-load]').forEach(function (b) {
      b.addEventListener('click', function () { loadSaved(b.getAttribute('data-load')); });
    });
    host.querySelectorAll('[data-delds]').forEach(function (b) {
      b.addEventListener('click', function () {
        api('/api/datasets/' + encodeURIComponent(b.getAttribute('data-delds')), { method: 'DELETE' })
          .then(refreshSaved).catch(function () {});
      });
    });
  } catch (e) {}
}

async function loadSaved(name) {
  try { const j = await api('/api/datasets/' + encodeURIComponent(name)); rows = j.rows || []; renderRows(); }
  catch (e) { alert('Load failed: ' + e.message); }
}

function init() {
  isAdmin().then(function (ok) {
    if (!ok) return;
    ['rail-dataset', 'tool-dataset-btn'].forEach(function (id) { const b = $(id); if (b) b.style.display = ''; });
  });
  ['rail-dataset', 'tool-dataset-btn'].forEach(function (id) { const b = $(id); if (b) b.addEventListener('click', openDataset); });
  const x = $('dataset-close'); if (x) x.addEventListener('click', closeDataset);
  const fmt = $('dataset-format'); if (fmt) fmt.addEventListener('change', renderFields);
  const add = $('dataset-add'); if (add) add.addEventListener('click', addRow);
  const imp = $('dataset-import-btn'); if (imp) imp.addEventListener('click', importText);
  const val = $('dataset-validate'); if (val) val.addEventListener('click', validate);
  const sv = $('dataset-save'); if (sv) sv.addEventListener('click', save);
  Modals.register('dataset-modal', { railBtnId: 'rail-dataset', sidebarBtnId: 'tool-dataset-btn', closeFn: closeDataset });
}

document.addEventListener('DOMContentLoaded', init);
```

- [ ] **Step 4: Add the modal + entries + datalist to `static/index.html`**

Insert the Dataset modal immediately BEFORE the `<div id="training-modal"` line:

```html
  <!-- AI Studio: Dataset builder/validator — admin -->
  <div id="dataset-modal" class="modal hidden">
    <div class="modal-content admin-modal-content" role="dialog" aria-label="Dataset builder">
      <div class="modal-header">
        <h4>Dataset builder</h4>
        <button class="close-btn" id="dataset-close" aria-label="Close">&#x2716;</button>
      </div>
      <div class="admin-card">
        <label>Format
          <select id="dataset-format">
            <option value="text">text</option>
            <option value="instruction">instruction + response</option>
            <option value="prompt">prompt + completion</option>
          </select>
        </label>
        <div id="dataset-fields"></div>
        <button id="dataset-add" class="btn">Add row</button>
      </div>
      <div class="admin-card">
        <label>Import (.jsonl — one JSON object per line)<br>
          <textarea id="dataset-import" rows="3" style="width:100%" placeholder='{"text": "..."}'></textarea></label>
        <button id="dataset-import-btn" class="btn">Import lines</button>
      </div>
      <div class="admin-card">
        <div>Rows: <b id="dataset-count">0 row(s)</b></div>
        <div id="dataset-rows" style="max-height:180px;overflow:auto;font-size:12px"></div>
        <button id="dataset-validate" class="btn">Validate</button>
        <div id="dataset-report" style="font-size:12px;margin-top:6px"></div>
      </div>
      <div class="admin-card">
        <label>Save as <input id="dataset-name" placeholder="my-dataset" style="width:60%"></label>
        <button id="dataset-save" class="btn">Save dataset</button>
        <div style="font-weight:600;margin-top:8px">Saved datasets</div>
        <div id="dataset-saved"></div>
      </div>
    </div>
  </div>
```

Insert the rail button immediately AFTER the `#rail-training` button line:

```html
    <button class="icon-rail-btn" id="rail-dataset" title="Dataset" style="display:none"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/></svg></button>
```

Insert the sidebar Tools entry immediately AFTER the `#tool-training-btn` `list-item` block (mirror its markup):

```html
        <div class="list-item" id="tool-dataset-btn" style="display:none">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;opacity:0.5;"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/></svg>
          <span class="grow">Dataset</span>
        </div>
```

Add the modulepreload after the `trainingCore.js` modulepreload line, and the script after the `training.js` script line:

```html
  <link rel="modulepreload" href="/static/js/datasetCore.js">
```
```html
<script type="module" src="/static/js/dataset.js"></script>
```

Give the Training dataset input a datalist — change the `#training-dataset` input line (index.html:487) to add `list="dataset-suggestions"` and add the datalist right after it:

```html
        <label>Dataset (.jsonl file path)<br><input id="training-dataset" list="dataset-suggestions" placeholder="C:\path\to\data.jsonl" style="width:100%"><datalist id="dataset-suggestions"></datalist></label>
```

- [ ] **Step 5: Populate the datalist in `static/js/training.js`**

In `training.js`, add a helper and call it from `openTraining` (which already calls `refreshEnv(); refreshAdapters(); …`). Add the function near `refreshAdapters`:

```javascript
async function refreshDatasetList() {
  try {
    const j = await api('/api/datasets');
    const dl = $('dataset-suggestions');
    if (dl) dl.innerHTML = (j.datasets || []).map(function (d) {
      return '<option value="' + esc(d.path) + '">' + esc(d.name) + '</option>';
    }).join('');
  } catch (e) {}
}
```

and add `refreshDatasetList();` to the `openTraining` function body (alongside the existing `refreshEnv(); refreshAdapters(); loadFreeVram(); resumeIfRunning();`).

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_dataset_ui.py tests/test_dataset_core_js.py tests/test_training_ui.py --import-mode=importlib -q`
Expected: PASS (the `node --check` gates confirm `dataset.js` and `training.js` parse).

- [ ] **Step 7: Commit**

```bash
git add static/index.html static/js/dataset.js static/js/training.js tests/test_dataset_ui.py
git commit -m "feat(dataset-tools): Dataset modal + controller; Training dataset datalist

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Finalize — Help section + full suite

**Files:**
- Modify: `static/index.html` (Help-manual "Datasets" section)
- Test: `tests/test_dataset_ui.py` (extend); run the whole dataset suite + `import app`

- [ ] **Step 1: Add the Help entry test**

Append to `tests/test_dataset_ui.py`:

```python
def test_help_manual_has_dataset_section():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert "build a training dataset without writing JSON" in html
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_dataset_ui.py::test_help_manual_has_dataset_section --import-mode=importlib -q`
Expected: FAIL (the Help phrase isn't present yet).

- [ ] **Step 3: Add the Help section**

In `static/index.html`, find the Help-manual `Training (LoRA / QLoRA)` `<details>` block and insert this sibling immediately after it:

```html
          <details>
            <summary style="cursor:pointer;font-weight:600;padding:6px 0;">Datasets (AI Studio)</summary>
            <p>Open <b>Dataset</b> from the sidebar (admin) to build a training dataset without writing JSON. Pick a <b>format</b> (plain text, instruction+response, or prompt+completion), fill the fields, and <b>Add row</b> — or <b>Import</b> lines from an existing <code>.jsonl</code>. <b>Validate</b> checks every row against the training format and shows per-line errors plus stats (row count, shapes, approximate tokens). <b>Save</b> writes it to your datasets folder, and it then appears in the Training panel's dataset dropdown so you can train on it directly.</p>
          </details>
```

- [ ] **Step 4: Run the full dataset suite + import smoke**

Run: `python -m pytest tests/test_dataset_normalize.py tests/test_dataset_validate.py tests/test_dataset_store.py tests/test_dataset_routes.py tests/test_dataset_core_js.py tests/test_dataset_ui.py tests/test_training_dataset.py tests/test_training_ui.py --import-mode=importlib -q`
Expected: PASS (all green).
Run: `python -c "import app; print('OK')"`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add static/index.html tests/test_dataset_ui.py
git commit -m "docs(dataset-tools): Help-manual Datasets section

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Behavior-preserving refactor:** Task 1 must keep `tests/test_training_dataset.py` green — `load_jsonl`'s
  raised messages are unchanged; only the internals now route through `normalize_row`.
- **Never-raises** on the validator, store, and routes is load-bearing (a bad dataset must report errors,
  never crash). Routes degrade malformed input to 400.
- **Both-surfaces admin reveal:** the Dataset tool needs BOTH a `#rail-dataset` icon button AND a
  `#tool-dataset-btn` sidebar Tools entry, both `display:none` and revealed by `isAdmin()` (the gap that
  hid Training from the sidebar earlier — don't repeat it).
- **No automated UI test** for the visual panel — `node --check` + the text-guards prove the module parses
  and wires the right ids/routes; manual GUI verification (build → validate → save → appears in Training)
  is owed by the user. This is frontend-only + Python; no rebuild is needed to test, but the shipped feature
  reaches users via the next installer rebuild.
- **Scope:** the builder + validator only. NO synthetic generation (sub-project 2), NO image captioning/
  labeling, NO raw-JSONL editor, NO split/dedup/augmentation.
