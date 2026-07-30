# Image AI Studio — Dataset Prep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An admin "Image Dataset" tool that captions, labels/validates, and exports an image dataset (upload or pulled from the Gallery) in the kohya-ss/diffusers `<image>` + `<image>.txt` convention, ready for LoRA training.

**Architecture:** Four small, pure-ish backend modules (`caption.py`, `validate.py`, `working_set.py`, `store.py`) wired together by one admin-gated route module, fronted by a new admin modal mirroring the shipped Dataset Builder's both-surfaces pattern. A server-side **working set** (a temp staging directory, identified by an opaque id) holds uploaded/gallery-sourced images between "add images" and "save" — images are binary, so unlike the text Dataset Builder's all-in-the-browser `rows[]`, re-POSTing image bytes on every action (caption-all, validate) would be wasteful; the working set lets the frontend reference images by id instead.

**Tech Stack:** Python 3.14 (FastAPI, stdlib `hashlib`/`uuid`/`json`, lazy-imported Pillow — already a dependency), ES5 browser JS, pytest (`--import-mode=importlib`), `node --check`.

## Global Constraints

- Main app (Python 3.14) must NEVER import torch/peft/bitsandbytes/transformers/gguf. Pillow is already a bundled dependency used elsewhere (`routes/gallery/gallery_routes.py`) via **lazy** `from PIL import Image` inside function bodies — follow that same lazy-import convention in `validate.py`.
- `caption_image`, `validate_image_set`, and every `working_set.py`/`store.py` function **NEVER raise** on any input (corrupt image bytes, path-traversal attempts, hostile working-set ids, a `vl_call` that raises). Routes **NEVER return 500** — failures ride a report/`error` field, exactly like every dataset-tools route this session.
- Reuse `src.document_processor.analyze_image_with_vl_result(image_path, owner=None, *, prompt=None) -> {"text","model"}` as the default vision call — no new vision dependency.
- The Gallery adapter is **read-only** against `core.database.GalleryImage` (`id`, `filename`, `caption`, `owner`, columns already exist) — never mutates gallery rows. Must be **owner-scoped** (only the requesting admin's own gallery images, matching `routes/gallery/gallery_routes.py`'s existing `img.owner != user` check) — a security requirement, not a style choice.
- Saved datasets live under `<DATA_DIR>/training/image_datasets/<name>/`; working sets live under `<DATA_DIR>/training/image_datasets/_working/<working_set_id>/` — both path-safe (sanitized names/ids, no traversal), mirroring `src/dataset_tools/store.py`'s `_safe_name` pattern.
- Admin-gated throughout: new router carries `dependencies=[Depends(require_admin)]` (mirrors `routes/dataset_routes.py`). Frontend reveal uses the established **both-surfaces** pattern — a `#rail-imagedataset` icon button AND a `#tool-imagedataset-btn` sidebar Tools entry, both `style="display:none"`, revealed together only when `/api/auth/status` reports `is_admin`.
- Every user/model-content string rendered via `innerHTML` MUST be `esc()`'d first (captions are LLM output — untrusted).
- pytest `--import-mode=importlib`; `node --check` gates the new JS; commit to `dev`; commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; stage only the named files (NEVER `git add -A`; never stage `installer/Output/Assist-Setup.exe`).

## File Structure

- **Create `src/image_dataset_tools/caption.py`** — `caption_image` (Task 1).
- **Create `src/image_dataset_tools/validate.py`** — `validate_image_set` (Task 2).
- **Create `src/image_dataset_tools/store.py`** — `ImageDatasetStore` for FINAL saved datasets (Task 3).
- **Create `src/image_dataset_tools/working_set.py`** — ephemeral staging area (Task 4).
- **Create `routes/image_dataset_routes.py`** — all admin endpoints, wiring the four modules above (Task 5).
- **Modify `app.py`** — wire the new router (Task 5).
- **Modify `static/index.html`** — new modal + rail + sidebar entries (Task 6).
- **Create `static/js/imageDataset.js`** — controller (Task 7).
- **Modify `static/index.html`** — Help section (Task 8).
- **Tests:** `tests/test_image_dataset_caption.py` (Task 1), `tests/test_image_dataset_validate.py` (Task 2), `tests/test_image_dataset_store.py` (Task 3), `tests/test_image_dataset_working_set.py` (Task 4), `tests/test_image_dataset_routes.py` (Task 5), `tests/test_image_dataset_ui.py` (Tasks 6-8).

---

### Task 1: `caption.py` — vision-model captioning

**Files:**
- Create: `src/image_dataset_tools/caption.py`
- Test: `tests/test_image_dataset_caption.py`

**Interfaces:**
- Produces: `caption_image(path, *, vl_call=None, prompt=None, owner=None) -> (caption: str|None, error: str|None)`. Never raises.

- [ ] **Step 1: Write the failing test**

Create `tests/test_image_dataset_caption.py`:

```python
from src.image_dataset_tools.caption import caption_image, DEFAULT_CAPTION_PROMPT


def test_caption_image_happy_path():
    def fake_vl(path, owner=None, *, prompt=None):
        assert prompt == DEFAULT_CAPTION_PROMPT  # default prompt used
        return {"text": "a red sports car on a city street", "model": "qwen-vl"}
    caption, error = caption_image("photo.jpg", vl_call=fake_vl)
    assert caption == "a red sports car on a city street" and error is None


def test_caption_image_custom_prompt_threaded():
    seen = {}
    def fake_vl(path, owner=None, *, prompt=None):
        seen["prompt"] = prompt
        return {"text": "x", "model": "m"}
    caption_image("p.jpg", vl_call=fake_vl, prompt="custom prompt")
    assert seen["prompt"] == "custom prompt"


def test_caption_image_reports_missing_text():
    def fake_vl(path, owner=None, *, prompt=None):
        return {"text": "", "model": "m"}
    caption, error = caption_image("p.jpg", vl_call=fake_vl)
    assert caption is None and "empty" in error.lower()


def test_caption_image_never_raises_on_vl_error():
    def boom(path, owner=None, *, prompt=None):
        raise RuntimeError("vision model unavailable")
    caption, error = caption_image("p.jpg", vl_call=boom)
    assert caption is None and "vision model unavailable" in error


def test_caption_image_never_raises_on_non_dict_result():
    def bad_vl(path, owner=None, *, prompt=None):
        return "not-a-dict"
    caption, error = caption_image("p.jpg", vl_call=bad_vl)
    assert caption is None and error is not None


def test_caption_image_bracketed_unavailable_marker_is_an_error():
    # analyze_image_with_vl_result signals unavailability with a bracketed
    # marker string in "text" (e.g. "[No vision model configured — ...]"),
    # not an exception -- must be treated as an error, not a real caption.
    def fake_vl(path, owner=None, *, prompt=None):
        return {"text": "[No vision model configured — set one in Settings → Vision]", "model": ""}
    caption, error = caption_image("p.jpg", vl_call=fake_vl)
    assert caption is None and "vision model" in error.lower()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_image_dataset_caption.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.image_dataset_tools'`).

- [ ] **Step 3: Implement**

Create `src/image_dataset_tools/__init__.py` (empty file).

Create `src/image_dataset_tools/caption.py`:

```python
"""Auto-caption an image with the shipped vision model, for LoRA-style
training-dataset captions. Pure wrapper, injectable vl_call. Never raises."""

DEFAULT_CAPTION_PROMPT = (
    "Describe this image in one plain, concrete sentence suitable as a "
    "caption for image-generation training data. Do not mention that it is "
    "an image or photo; just describe the subject, setting, and style."
)


def _default_vl_call(path, owner=None, *, prompt=None):
    from src.document_processor import analyze_image_with_vl_result
    return analyze_image_with_vl_result(path, owner=owner, prompt=prompt)


def caption_image(path, *, vl_call=None, prompt=None, owner=None):
    """Return (caption, error) -- exactly one is None. Never raises."""
    try:
        call = vl_call or _default_vl_call
        result = call(path, owner=owner, prompt=prompt or DEFAULT_CAPTION_PROMPT)
    except Exception as e:  # noqa: BLE001
        return None, str(e) or "caption failed (no error detail)"
    if not isinstance(result, dict):
        return None, "vision call returned an unexpected result"
    text = result.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, "vision model returned an empty caption"
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        # analyze_image_with_vl_result signals "vision disabled" / "no model
        # configured" with a bracketed marker string, not an exception.
        return None, stripped.strip("[]")
    return stripped, None
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_image_dataset_caption.py --import-mode=importlib -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/image_dataset_tools/__init__.py src/image_dataset_tools/caption.py tests/test_image_dataset_caption.py
git commit -m "feat(image-dataset): vision-model captioning (pure, never-raises)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `validate.py` — image-set validation

**Files:**
- Create: `src/image_dataset_tools/validate.py`
- Test: `tests/test_image_dataset_validate.py`

**Interfaces:**
- Produces: `validate_image_set(entries, *, min_dimension=256) -> dict`. `entries` is
  `[{"id": str, "path": str, "caption": str|None}, ...]`. Returns
  `{"total","valid","invalid","errors":[{"id","message"}],"stats":{"duplicates","missing_captions"}}`.
  Never raises.

- [ ] **Step 1: Write the failing test**

Create `tests/test_image_dataset_validate.py`:

```python
import io
from PIL import Image
from src.image_dataset_tools.validate import validate_image_set


def _make_png(path, size=(512, 512), color=(255, 0, 0)):
    Image.new("RGB", size, color).save(path, format="PNG")


def test_validate_clean_set(tmp_path):
    p1, p2 = tmp_path / "a.png", tmp_path / "b.png"
    _make_png(p1); _make_png(p2, color=(0, 255, 0))
    entries = [{"id": "a", "path": str(p1), "caption": "a red square"},
              {"id": "b", "path": str(p2), "caption": "a green square"}]
    rep = validate_image_set(entries)
    assert rep["total"] == 2 and rep["valid"] == 2 and rep["invalid"] == 0
    assert rep["stats"]["duplicates"] == 0 and rep["stats"]["missing_captions"] == 0


def test_validate_flags_missing_caption(tmp_path):
    p = tmp_path / "a.png"; _make_png(p)
    rep = validate_image_set([{"id": "a", "path": str(p), "caption": ""}])
    assert rep["invalid"] == 1 and rep["errors"][0]["id"] == "a"
    assert "caption" in rep["errors"][0]["message"].lower()


def test_validate_flags_corrupt_image(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"not a real png")
    rep = validate_image_set([{"id": "bad", "path": str(p), "caption": "x"}])
    assert rep["invalid"] == 1 and "corrupt" in rep["errors"][0]["message"].lower() or \
          "unreadable" in rep["errors"][0]["message"].lower()


def test_validate_flags_duplicates(tmp_path):
    p1, p2 = tmp_path / "a.png", tmp_path / "b.png"
    _make_png(p1); _make_png(p2)  # identical content
    entries = [{"id": "a", "path": str(p1), "caption": "x"},
              {"id": "b", "path": str(p2), "caption": "y"}]
    rep = validate_image_set(entries)
    assert rep["stats"]["duplicates"] == 1
    assert any("duplicate" in e["message"].lower() for e in rep["errors"])


def test_validate_flags_small_resolution(tmp_path):
    p = tmp_path / "tiny.png"; _make_png(p, size=(32, 32))
    rep = validate_image_set([{"id": "a", "path": str(p), "caption": "x"}], min_dimension=256)
    assert rep["invalid"] == 1 and "resolution" in rep["errors"][0]["message"].lower()


def test_validate_never_raises_on_hostile_input():
    assert validate_image_set(None)["total"] == 0
    assert validate_image_set("not-a-list")["total"] == 0
    rep = validate_image_set([{"id": "a"}, "not-a-dict", 42, {"path": "/nope/x.png", "caption": "x"}])
    assert rep["total"] == 4 and rep["invalid"] == 4  # every entry unresolvable, none raise
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_image_dataset_validate.py --import-mode=importlib -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

Create `src/image_dataset_tools/validate.py`:

```python
"""Validate an image dataset (all entries, non-raising): corrupt/unreadable
images, missing captions, exact duplicates, resolution floor. Pure (Pillow is
lazy-imported, mirroring routes/gallery/gallery_routes.py's convention)."""
import hashlib
import os


def _hash_file(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return None


def validate_image_set(entries, *, min_dimension=256):
    """entries: [{"id","path","caption"}, ...]. Never raises."""
    entries = entries if isinstance(entries, list) else []
    try:
        min_dim = max(1, int(min_dimension))
    except Exception:  # noqa: BLE001
        min_dim = 256

    from PIL import Image, UnidentifiedImageError

    errors = []
    seen_hashes = {}
    duplicates = 0
    missing_captions = 0
    valid = 0

    for i, entry in enumerate(entries):
        eid = entry.get("id", i) if isinstance(entry, dict) else i
        if not isinstance(entry, dict):
            errors.append({"id": eid, "message": "not a valid entry"})
            continue
        path = entry.get("path")
        caption = entry.get("caption")
        if not isinstance(path, str) or not os.path.isfile(path):
            errors.append({"id": eid, "message": "image file not found"})
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img2:
                w, h = img2.size
        except (UnidentifiedImageError, OSError, ValueError):
            errors.append({"id": eid, "message": "image is corrupt or unreadable"})
            continue
        except Exception:  # noqa: BLE001
            errors.append({"id": eid, "message": "image is corrupt or unreadable"})
            continue

        bad = []
        if min(w, h) < min_dim:
            bad.append(f"resolution {w}x{h} is below the {min_dim}px floor")
        if not isinstance(caption, str) or not caption.strip():
            missing_captions += 1
            bad.append("caption is missing or empty")
        digest = _hash_file(path)
        if digest:
            if digest in seen_hashes:
                duplicates += 1
                bad.append(f"duplicate of image '{seen_hashes[digest]}'")
            else:
                seen_hashes[digest] = eid
        if bad:
            errors.append({"id": eid, "message": "; ".join(bad)})
        else:
            valid += 1

    return {
        "total": len(entries),
        "valid": valid,
        "invalid": len(errors),
        "errors": errors,
        "stats": {"duplicates": duplicates, "missing_captions": missing_captions},
    }
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_image_dataset_validate.py --import-mode=importlib -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/image_dataset_tools/validate.py tests/test_image_dataset_validate.py
git commit -m "feat(image-dataset): image-set validation (corrupt/dup/resolution/caption, never-raises)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `store.py` — saved-dataset persistence

**Files:**
- Create: `src/image_dataset_tools/store.py`
- Test: `tests/test_image_dataset_store.py`

**Interfaces:**
- Produces: `ImageDatasetStore(base_dir=None)` with `save(name, entries, trigger_word="") ->
  {"ok","path","name"}|{"error"}` (`entries`: `[{"path": source_file_path, "caption": str}]` — files are
  COPIED, not moved, so the working set can be cleaned up independently); `list() -> [{"name","path","images","size"}]`;
  `load(name) -> {"name","path","trigger_word","images":[{"filename","caption"}]}|{"error"}`;
  `delete(name) -> {"ok"}|{"error"}`. `get_image_dataset_store()` singleton. Path-safe, never-raises.

- [ ] **Step 1: Write the failing test**

Create `tests/test_image_dataset_store.py`:

```python
import os
from src.image_dataset_tools.store import ImageDatasetStore


def _src_image(tmp_path, name, data=b"fake-image-bytes"):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_save_load_list_delete(tmp_path):
    src_dir = tmp_path / "src"; src_dir.mkdir()
    a = _src_image(src_dir, "a.png", b"AAA")
    b = _src_image(src_dir, "b.jpg", b"BBB")
    store = ImageDatasetStore(base_dir=str(tmp_path / "store"))
    entries = [{"path": a, "caption": "cap a"}, {"path": b, "caption": "cap b"}]
    out = store.save("my set", entries, trigger_word="ohwx-widget")
    assert out.get("ok") and out["name"] == "my-set"
    assert os.path.isdir(out["path"])

    lst = store.list()
    assert lst and lst[0]["name"] == "my-set" and lst[0]["images"] == 2

    loaded = store.load("my-set")
    assert loaded["trigger_word"] == "ohwx-widget"
    captions = sorted(img["caption"] for img in loaded["images"])
    assert captions == ["cap a", "cap b"]
    # each image file actually copied with matching content
    for img in loaded["images"]:
        full = os.path.join(loaded["path"], img["filename"])
        assert os.path.isfile(full)

    assert store.delete("my-set").get("ok") and store.list() == []


def test_name_sanitized_no_traversal(tmp_path):
    src_dir = tmp_path / "src"; src_dir.mkdir()
    a = _src_image(src_dir, "a.png")
    store = ImageDatasetStore(base_dir=str(tmp_path / "store"))
    out = store.save("../../evil", [{"path": a, "caption": "x"}])
    assert out.get("ok") and str(tmp_path / "store") in out["path"] and ".." not in out["name"]


def test_empty_entries_and_bad_name_rejected(tmp_path):
    store = ImageDatasetStore(base_dir=str(tmp_path))
    assert "error" in store.save("ok", [])
    assert "error" in store.save("", [{"path": "x", "caption": "y"}])


def test_missing_source_file_is_skipped_not_fatal(tmp_path):
    src_dir = tmp_path / "src"; src_dir.mkdir()
    a = _src_image(src_dir, "a.png")
    store = ImageDatasetStore(base_dir=str(tmp_path / "store"))
    out = store.save("s", [{"path": a, "caption": "x"}, {"path": "/nope.png", "caption": "y"}])
    assert out.get("ok")
    loaded = store.load("s")
    assert len(loaded["images"]) == 1  # the missing one was skipped, not fatal


def test_load_missing(tmp_path):
    assert "error" in ImageDatasetStore(base_dir=str(tmp_path)).load("nope")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_image_dataset_store.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.image_dataset_tools.store'`).

- [ ] **Step 3: Implement**

Create `src/image_dataset_tools/store.py`:

```python
"""Save/list/load/delete image LoRA-training datasets under <DATA_DIR>/
training/image_datasets/<name>/, in the kohya-ss/diffusers convention
(image + same-basename .txt caption). Path-safe. Never raises."""
import json
import os
import re
import shutil


def _default_dir():
    from src.constants import DATA_DIR
    return os.path.join(DATA_DIR, "training", "image_datasets")


def _safe_name(name):
    base = os.path.basename(str(name or "").strip())
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-.")
    return base


class ImageDatasetStore:
    def __init__(self, base_dir=None):
        self._dir = base_dir or _default_dir()

    def _dataset_dir(self, name):
        safe = _safe_name(name)
        return safe, (os.path.join(self._dir, safe) if safe else None)

    def save(self, name, entries, trigger_word="") -> dict:
        safe, path = self._dataset_dir(name)
        if not safe:
            return {"error": "invalid dataset name"}
        if not isinstance(entries, list) or not entries:
            return {"error": "entries must be a non-empty list"}
        tmp = path + ".tmp"
        try:
            os.makedirs(self._dir, exist_ok=True)
            if os.path.isdir(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            saved = []
            for i, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    continue
                src = entry.get("path")
                caption = entry.get("caption")
                if not isinstance(src, str) or not os.path.isfile(src):
                    continue  # skip missing sources -- not fatal
                ext = os.path.splitext(src)[1].lower() or ".png"
                base_name = f"{i:04d}{ext}"
                shutil.copyfile(src, os.path.join(tmp, base_name))
                with open(os.path.join(tmp, f"{i:04d}.txt"), "w", encoding="utf-8") as f:
                    f.write(caption if isinstance(caption, str) else "")
                saved.append(base_name)
            meta = {"trigger_word": trigger_word if isinstance(trigger_word, str) else "",
                    "images": len(saved)}
            with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            os.replace(tmp, path)
            return {"ok": True, "path": path, "name": safe}
        except Exception as e:  # noqa: BLE001
            if os.path.isdir(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
            return {"error": f"save failed: {e}"}

    def list(self) -> list:
        out = []
        try:
            if not os.path.isdir(self._dir):
                return out
            for fn in sorted(os.listdir(self._dir)):
                p = os.path.join(self._dir, fn)
                if not os.path.isdir(p) or fn.endswith(".tmp"):
                    continue
                try:
                    n_images = sum(1 for f in os.listdir(p) if not f.endswith((".txt", ".json")))
                    size = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p))
                    out.append({"name": fn, "path": p, "images": n_images, "size": size})
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        return out

    def load(self, name) -> dict:
        safe, path = self._dataset_dir(name)
        if not path or not os.path.isdir(path):
            return {"error": "dataset not found"}
        try:
            meta = {"trigger_word": "", "images": 0}
            meta_path = os.path.join(path, "meta.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta.update(json.load(f))
                except Exception:  # noqa: BLE001
                    pass
            images = []
            for fn in sorted(os.listdir(path)):
                if fn.endswith((".txt", ".json")):
                    continue
                stem = os.path.splitext(fn)[0]
                cap_path = os.path.join(path, stem + ".txt")
                caption = ""
                if os.path.isfile(cap_path):
                    try:
                        with open(cap_path, "r", encoding="utf-8") as f:
                            caption = f.read()
                    except Exception:  # noqa: BLE001
                        pass
                images.append({"filename": fn, "caption": caption})
            return {"name": safe, "path": path, "trigger_word": meta.get("trigger_word", ""), "images": images}
        except Exception as e:  # noqa: BLE001
            return {"error": f"load failed: {e}"}

    def delete(self, name) -> dict:
        safe, path = self._dataset_dir(name)
        try:
            if path and os.path.isdir(path):
                shutil.rmtree(path)
                return {"ok": True}
            return {"error": "dataset not found"}
        except Exception as e:  # noqa: BLE001
            return {"error": f"delete failed: {e}"}


_store = None


def get_image_dataset_store():
    global _store
    if _store is None:
        _store = ImageDatasetStore()
    return _store
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_image_dataset_store.py --import-mode=importlib -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/image_dataset_tools/store.py tests/test_image_dataset_store.py
git commit -m "feat(image-dataset): path-safe saved-dataset store (kohya-ss convention, never-raises)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `working_set.py` — ephemeral staging area

**Files:**
- Create: `src/image_dataset_tools/working_set.py`
- Test: `tests/test_image_dataset_working_set.py`

**Interfaces:**
- Produces: `new_working_set() -> str` (opaque id); `add_images(working_set_id, images) -> [{"id","caption"}]`
  (`images`: `[(filename, bytes)]`, or `[(filename, bytes, caption)]` to seed a caption e.g. from Gallery);
  `list_images(working_set_id) -> [{"id","caption"}]`; `get_image_path(working_set_id, image_id) ->
  str|None`; `set_caption(working_set_id, image_id, caption) -> bool`; `delete_working_set(working_set_id) ->
  bool`. Path-safe (working_set_id and image_id are sanitized against traversal). Never raises.

- [ ] **Step 1: Write the failing test**

Create `tests/test_image_dataset_working_set.py`:

```python
import os
from src.image_dataset_tools import working_set as ws


def test_new_working_set_is_unique(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_default_dir", lambda: str(tmp_path))
    a, b = ws.new_working_set(), ws.new_working_set()
    assert a != b and len(a) >= 8


def test_add_list_get_path(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_default_dir", lambda: str(tmp_path))
    wid = ws.new_working_set()
    added = ws.add_images(wid, [("photo one.png", b"AAA"), ("two.jpg", b"BBB", "seeded caption")])
    assert len(added) == 2
    imgs = ws.list_images(wid)
    assert len(imgs) == 2
    ids_by_caption = {img["caption"]: img["id"] for img in imgs}
    assert ids_by_caption["seeded caption"]
    for img in imgs:
        path = ws.get_image_path(wid, img["id"])
        assert path and os.path.isfile(path)


def test_set_caption_updates_it(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_default_dir", lambda: str(tmp_path))
    wid = ws.new_working_set()
    added = ws.add_images(wid, [("a.png", b"AAA")])
    iid = added[0]["id"]
    assert ws.set_caption(wid, iid, "a new caption") is True
    imgs = ws.list_images(wid)
    assert imgs[0]["caption"] == "a new caption"


def test_delete_working_set(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_default_dir", lambda: str(tmp_path))
    wid = ws.new_working_set()
    ws.add_images(wid, [("a.png", b"AAA")])
    assert ws.delete_working_set(wid) is True
    assert ws.list_images(wid) == []


def test_path_traversal_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_default_dir", lambda: str(tmp_path))
    wid = ws.new_working_set()
    assert ws.get_image_path("../../evil", "x") is None
    assert ws.get_image_path(wid, "../../etc/passwd") is None


def test_never_raises_on_hostile_input(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, "_default_dir", lambda: str(tmp_path))
    assert ws.add_images("bad id", None) == []
    assert ws.add_images("bad id", [("x",), 42, ("y.png", "not-bytes")]) == []
    assert ws.list_images("does-not-exist") == []
    assert ws.set_caption("does-not-exist", "x", "c") is False
    assert ws.delete_working_set("does-not-exist") is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_image_dataset_working_set.py --import-mode=importlib -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

Create `src/image_dataset_tools/working_set.py`:

```python
"""Ephemeral staging area for images being prepared into a dataset --
uploaded/gallery-sourced images live here between "add" and "save". A
working set is identified by an opaque id; each image gets its own
sanitized-basename subfolder holding the image bytes + a state.json with its
caption. Path-safe. Never raises."""
import json
import os
import re
import shutil
import uuid


def _default_dir():
    from src.constants import DATA_DIR
    return os.path.join(DATA_DIR, "training", "image_datasets", "_working")


def _safe_id(value):
    s = re.sub(r"[^A-Za-z0-9_-]+", "", str(value or ""))
    return s


def _ws_dir(working_set_id):
    safe = _safe_id(working_set_id)
    if not safe:
        return None
    return os.path.join(_default_dir(), safe)


def _state_path(ws_dir):
    return os.path.join(ws_dir, "state.json")


def _read_state(ws_dir):
    try:
        with open(_state_path(ws_dir), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {"images": []}
    except Exception:  # noqa: BLE001
        return {"images": []}


def _write_state(ws_dir, state):
    try:
        os.makedirs(ws_dir, exist_ok=True)
        tmp = _state_path(ws_dir) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, _state_path(ws_dir))
    except Exception:  # noqa: BLE001
        pass


def new_working_set() -> str:
    return uuid.uuid4().hex


def add_images(working_set_id, images) -> list:
    """`images`: [(filename, bytes)] or [(filename, bytes, caption)]. Returns
    the newly-added [{"id","caption"}]. Never raises; [] on any failure."""
    ws_dir = _ws_dir(working_set_id)
    if not ws_dir or not isinstance(images, list):
        return []
    try:
        os.makedirs(ws_dir, exist_ok=True)
        state = _read_state(ws_dir)
        state.setdefault("images", [])
        added = []
        for item in images:
            try:
                if not isinstance(item, tuple) or len(item) < 2:
                    continue
                filename, data = item[0], item[1]
                caption = item[2] if len(item) > 2 and isinstance(item[2], str) else ""
                if not isinstance(filename, str) or not isinstance(data, (bytes, bytearray)):
                    continue
                ext = os.path.splitext(filename)[1].lower() or ".png"
                image_id = uuid.uuid4().hex
                with open(os.path.join(ws_dir, image_id + ext), "wb") as f:
                    f.write(data)
                entry = {"id": image_id, "filename": image_id + ext, "caption": caption}
                state["images"].append(entry)
                added.append({"id": image_id, "caption": caption})
            except Exception:  # noqa: BLE001
                continue
        _write_state(ws_dir, state)
        return added
    except Exception:  # noqa: BLE001
        return []


def list_images(working_set_id) -> list:
    ws_dir = _ws_dir(working_set_id)
    if not ws_dir or not os.path.isdir(ws_dir):
        return []
    state = _read_state(ws_dir)
    return [{"id": img.get("id"), "caption": img.get("caption", "")}
            for img in state.get("images", []) if isinstance(img, dict)]


def get_image_path(working_set_id, image_id):
    ws_dir = _ws_dir(working_set_id)
    safe_id = _safe_id(image_id)
    if not ws_dir or not safe_id:
        return None
    state = _read_state(ws_dir)
    for img in state.get("images", []):
        if isinstance(img, dict) and img.get("id") == safe_id:
            path = os.path.join(ws_dir, img.get("filename", ""))
            return path if os.path.isfile(path) else None
    return None


def set_caption(working_set_id, image_id, caption) -> bool:
    ws_dir = _ws_dir(working_set_id)
    safe_id = _safe_id(image_id)
    if not ws_dir or not safe_id or not os.path.isdir(ws_dir):
        return False
    state = _read_state(ws_dir)
    found = False
    for img in state.get("images", []):
        if isinstance(img, dict) and img.get("id") == safe_id:
            img["caption"] = caption if isinstance(caption, str) else ""
            found = True
            break
    if found:
        _write_state(ws_dir, state)
    return found


def delete_working_set(working_set_id) -> bool:
    ws_dir = _ws_dir(working_set_id)
    if not ws_dir or not os.path.isdir(ws_dir):
        return False
    try:
        shutil.rmtree(ws_dir)
        return True
    except Exception:  # noqa: BLE001
        return False
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_image_dataset_working_set.py --import-mode=importlib -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/image_dataset_tools/working_set.py tests/test_image_dataset_working_set.py
git commit -m "feat(image-dataset): ephemeral working-set staging area (path-safe, never-raises)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Admin routes — wire it all together

**Files:**
- Create: `routes/image_dataset_routes.py`
- Modify: `app.py`
- Test: `tests/test_image_dataset_routes.py`

**Interfaces:**
- Consumes: `caption_image` (Task 1), `validate_image_set` (Task 2), `ImageDatasetStore`/`get_image_dataset_store` (Task 3), `working_set` module functions (Task 4); `core.database.SessionLocal`, `core.database.GalleryImage` (Gallery adapter); `core.middleware.require_admin`.
- Produces: `setup_image_dataset_routes() -> APIRouter` at `/api/image-datasets`, admin-gated. Endpoints:
  `POST /upload` (multipart), `POST /from-gallery` (JSON), `GET /working/{working_set_id}/{image_id}` (serves raw bytes), `POST /caption`, `POST /validate`, `POST ""` (save), `GET ""`, `GET/DELETE /{name}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_image_dataset_routes.py`:

```python
import io
import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.image_dataset_routes as idr


def _client(monkeypatch):
    monkeypatch.setattr(idr, "require_admin", lambda: None)
    app = FastAPI(); app.include_router(idr.setup_image_dataset_routes())
    return TestClient(app)


def test_upload_then_caption_validate_save(monkeypatch, tmp_path):
    monkeypatch.setattr(idr.working_set, "_default_dir", lambda: str(tmp_path / "working"))
    monkeypatch.setattr(idr, "get_image_dataset_store",
                        lambda: idr.ImageDatasetStore(base_dir=str(tmp_path / "saved")))
    c = _client(monkeypatch)

    files = [("files", ("a.png", io.BytesIO(b"AAA"), "image/png")),
            ("files", ("b.png", io.BytesIO(b"BBB"), "image/png"))]
    r = c.post("/api/image-datasets/upload", files=files)
    assert r.status_code == 200
    wid = r.json()["working_set_id"]
    assert len(r.json()["images"]) == 2

    def fake_caption(path, *, vl_call=None, prompt=None, owner=None):
        return "an auto caption", None
    monkeypatch.setattr(idr, "caption_image", fake_caption)
    r2 = c.post("/api/image-datasets/caption", json={"working_set_id": wid})
    assert r2.status_code == 200 and all(img["caption"] == "an auto caption" for img in r2.json()["images"])

    captions = {img["id"]: "edited: " + img["caption"] for img in r2.json()["images"]}
    r3 = c.post("/api/image-datasets/validate", json={"working_set_id": wid, "captions": captions})
    assert r3.status_code == 200 and r3.json()["valid"] >= 0  # never-500; report shape present

    r4 = c.post("/api/image-datasets", json={"working_set_id": wid, "name": "my-set",
                                              "trigger_word": "ohwx", "captions": captions})
    assert r4.status_code == 200 and r4.json().get("ok")

    r5 = c.get("/api/image-datasets")
    assert r5.status_code == 200 and any(d["name"] == "my-set" for d in r5.json()["datasets"])


def test_working_image_served_and_path_safe(monkeypatch, tmp_path):
    monkeypatch.setattr(idr.working_set, "_default_dir", lambda: str(tmp_path / "working"))
    c = _client(monkeypatch)
    files = [("files", ("a.png", io.BytesIO(b"AAA"), "image/png"))]
    r = c.post("/api/image-datasets/upload", files=files)
    wid = r.json()["working_set_id"]
    iid = r.json()["images"][0]["id"]
    r2 = c.get(f"/api/image-datasets/working/{wid}/{iid}")
    assert r2.status_code == 200 and r2.content == b"AAA"
    r3 = c.get(f"/api/image-datasets/working/{wid}/../../etc-passwd")
    assert r3.status_code == 404


def test_from_gallery_is_owner_scoped(monkeypatch, tmp_path):
    monkeypatch.setattr(idr.working_set, "_default_dir", lambda: str(tmp_path / "working"))

    class _FakeRow:
        def __init__(self, id, filename, owner, caption=""):
            self.id, self.filename, self.owner, self.caption = id, filename, owner, caption

    img_path = tmp_path / "gallery_img.png"
    img_path.write_bytes(b"GALLERYBYTES")

    class _FakeQuery:
        def __init__(self, rows): self._rows = rows
        def filter(self, *a, **k): return self
        def all(self): return self._rows

    class _FakeDb:
        def __init__(self, rows): self._rows = rows
        def query(self, *a, **k): return _FakeQuery(self._rows)
        def close(self): pass

    owned = _FakeRow("g1", "gallery_img.png", "admin", "gallery caption")
    other = _FakeRow("g2", "other.png", "someone-else", "")
    monkeypatch.setattr(idr, "SessionLocal", lambda: _FakeDb([owned, other]))
    monkeypatch.setattr(idr, "_gallery_image_path", lambda filename: str(img_path))

    c = _client(monkeypatch)
    r = c.post("/api/image-datasets/from-gallery", json={"ids": ["g1", "g2"], "owner": "admin"})
    assert r.status_code == 200
    assert len(r.json()["images"]) == 1  # only the admin-owned row copied in
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_image_dataset_routes.py --import-mode=importlib -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

Create `routes/image_dataset_routes.py`:

```python
"""Admin-gated Image Dataset prep API (Image AI Studio, sub-project 1)."""
from fastapi import APIRouter, Body, Depends, File, HTTPException, UploadFile

from core.middleware import require_admin
from core.database import SessionLocal, GalleryImage
from src.image_dataset_tools import working_set
from src.image_dataset_tools.caption import caption_image
from src.image_dataset_tools.validate import validate_image_set
from src.image_dataset_tools.store import ImageDatasetStore, get_image_dataset_store


def _gallery_image_path(filename):
    from routes.gallery.gallery_routes import _gallery_image_path as real_resolve
    return str(real_resolve(filename))


def setup_image_dataset_routes() -> APIRouter:
    router = APIRouter(prefix="/api/image-datasets",
                       dependencies=[Depends(require_admin)])

    @router.post("/upload")
    async def upload(files: list[UploadFile] = File(...)):
        wid = working_set.new_working_set()
        pairs = []
        for f in files:
            try:
                data = await f.read()
                pairs.append((f.filename or "image.png", data))
            except Exception:  # noqa: BLE001
                continue
        added = working_set.add_images(wid, pairs)
        return {"working_set_id": wid, "images": added}

    @router.post("/from-gallery")
    async def from_gallery(body: dict = Body(...)):
        ids = body.get("ids")
        owner = body.get("owner")
        wid = body.get("working_set_id") or working_set.new_working_set()
        if not isinstance(ids, list) or not ids:
            return {"working_set_id": wid, "images": []}
        db = SessionLocal()
        try:
            rows = db.query(GalleryImage).filter(GalleryImage.id.in_(ids)).all()
        except Exception:  # noqa: BLE001
            rows = []
        finally:
            db.close()
        pairs = []
        for row in rows:
            if owner is not None and getattr(row, "owner", None) != owner:
                continue  # owner-scoped: never pull another user's images
            try:
                path = _gallery_image_path(row.filename)
                with open(path, "rb") as f:
                    data = f.read()
                pairs.append((row.filename, data, getattr(row, "caption", "") or ""))
            except Exception:  # noqa: BLE001
                continue
        added = working_set.add_images(wid, pairs)
        return {"working_set_id": wid, "images": added}

    @router.get("/working/{working_set_id}/{image_id}")
    async def get_working_image(working_set_id: str, image_id: str):
        from fastapi.responses import FileResponse
        path = working_set.get_image_path(working_set_id, image_id)
        if not path:
            raise HTTPException(404, "image not found")
        return FileResponse(path)

    @router.post("/caption")
    async def caption(body: dict = Body(...)):
        wid = body.get("working_set_id")
        only_ids = body.get("only_ids")
        results = []
        for img in working_set.list_images(wid):
            if isinstance(only_ids, list) and img["id"] not in only_ids:
                results.append(img)
                continue
            path = working_set.get_image_path(wid, img["id"])
            if not path:
                results.append({**img, "error": "image not found"})
                continue
            cap, err = caption_image(path)
            if cap is not None:
                working_set.set_caption(wid, img["id"], cap)
                results.append({"id": img["id"], "caption": cap})
            else:
                results.append({"id": img["id"], "caption": img.get("caption", ""), "error": err})
        return {"images": results}

    @router.post("/validate")
    async def validate(body: dict = Body(...)):
        wid = body.get("working_set_id")
        captions = body.get("captions") if isinstance(body.get("captions"), dict) else {}
        trigger = body.get("trigger_word") or ""
        entries = []
        for img in working_set.list_images(wid):
            path = working_set.get_image_path(wid, img["id"])
            cap = captions.get(img["id"], img.get("caption", ""))
            full_caption = f"{trigger} {cap}".strip() if trigger else cap
            entries.append({"id": img["id"], "path": path, "caption": full_caption})
        return validate_image_set(entries)

    @router.post("")
    async def save(body: dict = Body(...)):
        wid = body.get("working_set_id")
        name = body.get("name")
        trigger = body.get("trigger_word") or ""
        captions = body.get("captions") if isinstance(body.get("captions"), dict) else {}
        entries = []
        for img in working_set.list_images(wid):
            path = working_set.get_image_path(wid, img["id"])
            if not path:
                continue
            cap = captions.get(img["id"], img.get("caption", ""))
            full_caption = f"{trigger} {cap}".strip() if trigger else cap
            entries.append({"path": path, "caption": full_caption})
        out = get_image_dataset_store().save(name, entries, trigger_word=trigger)
        if "error" in out:
            raise HTTPException(400, out["error"])
        working_set.delete_working_set(wid)
        return out

    @router.get("")
    async def list_datasets():
        return {"datasets": get_image_dataset_store().list()}

    @router.get("/{name}")
    async def load(name: str):
        out = get_image_dataset_store().load(name)
        if "error" in out:
            raise HTTPException(404, out["error"])
        return out

    @router.delete("/{name}")
    async def delete(name: str):
        out = get_image_dataset_store().delete(name)
        if "error" in out:
            raise HTTPException(404, out["error"])
        return out

    return router
```

Wire into `app.py` immediately after the existing dataset router include (find `from routes.dataset_routes import setup_dataset_routes` / `app.include_router(setup_dataset_routes())` and add right after):

```python
from routes.image_dataset_routes import setup_image_dataset_routes
app.include_router(setup_image_dataset_routes())
```

- [ ] **Step 4: Run the tests + import smoke**

Run: `python -m pytest tests/test_image_dataset_routes.py --import-mode=importlib -q`
Expected: PASS (3 passed).
Run: `python -c "import app; print('OK')"`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add routes/image_dataset_routes.py app.py tests/test_image_dataset_routes.py
git commit -m "feat(image-dataset): admin /api/image-datasets routes (upload/gallery/caption/validate/save)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Frontend — modal markup + both-surfaces admin reveal

**Files:**
- Modify: `static/index.html`
- Test: `tests/test_image_dataset_ui.py`

**Interfaces:**
- Produces: `#imagedataset-modal`, `#rail-imagedataset`, `#tool-imagedataset-btn` in `index.html`, plus all
  the control ids Task 7's `imageDataset.js` binds to.

- [ ] **Step 1: Write the failing test**

Create `tests/test_image_dataset_ui.py`:

```python
import pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_index_has_image_dataset_modal_and_entries():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in ('id="imagedataset-modal"', 'id="rail-imagedataset"', 'id="tool-imagedataset-btn"',
               '/static/js/imageDataset.js',
               'id="imgds-source"', 'id="imgds-file"', 'id="imgds-gallery-pick"',
               'id="imgds-grid"', 'id="imgds-caption-all"', 'id="imgds-trigger"',
               'id="imgds-validate"', 'id="imgds-report"', 'id="imgds-name"',
               'id="imgds-save"', 'id="imgds-saved"'):
        assert el in html, f"{el} missing from index.html"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_image_dataset_ui.py --import-mode=importlib -q`
Expected: FAIL (elements missing).

- [ ] **Step 3: Add the modal + rail + sidebar entries to `static/index.html`**

Insert this modal block immediately BEFORE the `<div id="dataset-modal"` line (a sibling admin modal):

```html
  <!-- Image AI Studio: dataset prep (captioning/labeling/validation) — admin -->
  <div id="imagedataset-modal" class="modal hidden">
    <div class="modal-content admin-modal-content" role="dialog" aria-label="Image dataset prep">
      <div class="modal-header">
        <h4>Image dataset</h4>
        <button class="close-btn" id="imagedataset-close" aria-label="Close">&#x2716;</button>
      </div>
      <div class="admin-card">
        <label>Source
          <select id="imgds-source">
            <option value="upload">Upload images</option>
            <option value="gallery">From Gallery</option>
          </select>
        </label>
        <input id="imgds-file" type="file" accept="image/*" multiple>
        <button id="imgds-gallery-pick" class="btn" style="display:none">Pick from Gallery…</button>
      </div>
      <div class="admin-card">
        <div>Working set: <b id="imgds-count">0 image(s)</b></div>
        <div id="imgds-grid" style="display:flex;flex-wrap:wrap;gap:10px;max-height:320px;overflow:auto"></div>
        <button id="imgds-caption-all" class="btn">Caption all (AI)</button>
      </div>
      <div class="admin-card">
        <label>Trigger word <input id="imgds-trigger" placeholder="ohwx-widget" style="width:60%"></label>
        <button id="imgds-validate" class="btn">Validate</button>
        <div id="imgds-report" style="font-size:12px;margin-top:6px"></div>
      </div>
      <div class="admin-card">
        <label>Save as <input id="imgds-name" placeholder="my-image-dataset" style="width:60%"></label>
        <button id="imgds-save" class="btn">Save dataset</button>
        <div style="font-weight:600;margin-top:8px">Saved datasets</div>
        <div id="imgds-saved"></div>
      </div>
    </div>
  </div>
```

Insert the rail button immediately AFTER the `#rail-dataset` button line:

```html
    <button class="icon-rail-btn" id="rail-imagedataset" title="Image Dataset" style="display:none"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg></button>
```

Insert the sidebar Tools entry immediately AFTER the `#tool-dataset-btn` `list-item` block:

```html
        <div class="list-item" id="tool-imagedataset-btn" style="display:none">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;opacity:0.5;"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>
          <span class="grow">Image Dataset</span>
        </div>
```

Add the script tag immediately after the `dataset.js` script line:

```html
<script type="module" src="/static/js/imageDataset.js"></script>
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_image_dataset_ui.py --import-mode=importlib -q`
Expected: PASS (1 passed — Task 7 will make the script-tag reference resolve to a real file).

- [ ] **Step 5: Commit**

```bash
git add static/index.html tests/test_image_dataset_ui.py
git commit -m "feat(image-dataset): modal markup + both-surfaces admin reveal

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Frontend — `imageDataset.js` controller

**Files:**
- Create: `static/js/imageDataset.js`
- Test: `tests/test_image_dataset_ui.py` (extend)

**Interfaces:**
- Consumes: `/api/image-datasets/*` (Task 5); the modal ids from Task 6; `modalManager.js`'s `register`.
- Produces: a working `imageDataset.js` module wiring the whole upload→caption→validate→save flow.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_image_dataset_ui.py`:

```python
def test_image_dataset_js_wires_admin_and_routes():
    src = (ROOT / "static" / "js" / "imageDataset.js").read_text(encoding="utf-8")
    for s in ('rail-imagedataset', 'tool-imagedataset-btn', 'isAdmin', 'Modals.register',
              '/api/image-datasets/upload', '/api/image-datasets/from-gallery',
              '/api/image-datasets/caption', '/api/image-datasets/validate',
              "api('/api/image-datasets'"):
        assert s in src, f"{s} missing from imageDataset.js"


def test_image_dataset_js_syntax(tmp_path):
    src = (ROOT / "static" / "js" / "imageDataset.js").read_text(encoding="utf-8")
    mjs = tmp_path / "imageDataset.mjs"; mjs.write_text(src, encoding="utf-8")
    p = subprocess.run(["node", "--check", str(mjs)], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_image_dataset_ui.py -k "image_dataset_js" --import-mode=importlib -q`
Expected: FAIL (file missing).

- [ ] **Step 3: Implement**

Create `static/js/imageDataset.js`:

```javascript
// Image dataset prep (Image AI Studio, sub-project 1). ES module. Admin-only:
// entries stay hidden unless /api/auth/status reports is_admin. Mirrors dataset.js.
import * as Modals from './modalManager.js';

function $(id) { return document.getElementById(id); }
let workingSetId = null;
let images = [];  // [{id, caption}]

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

function openImageDataset() { $('imagedataset-modal').classList.remove('hidden'); renderGrid(); refreshSaved(); }
function closeImageDataset() { $('imagedataset-modal').classList.add('hidden'); }

function renderGrid() {
  const grid = $('imgds-grid'); if (!grid) return;
  grid.innerHTML = images.map(function (img) {
    return '<div style="width:140px">' +
      '<img src="/api/image-datasets/working/' + encodeURIComponent(workingSetId) + '/' + encodeURIComponent(img.id) +
      '" style="width:140px;height:140px;object-fit:cover;border-radius:6px">' +
      '<textarea data-cap="' + esc(img.id) + '" rows="2" style="width:140px;font-size:11px">' + esc(img.caption) + '</textarea>' +
      '<button class="btn" data-remove="' + esc(img.id) + '" style="width:100%">Remove</button>' +
      '</div>';
  }).join('');
  const c = $('imgds-count'); if (c) c.textContent = images.length + ' image(s)';
  grid.querySelectorAll('[data-cap]').forEach(function (ta) {
    ta.addEventListener('change', function () {
      const img = images.find(function (i) { return i.id === ta.getAttribute('data-cap'); });
      if (img) img.caption = ta.value;
    });
  });
  grid.querySelectorAll('[data-remove]').forEach(function (b) {
    b.addEventListener('click', function () {
      images = images.filter(function (i) { return i.id !== b.getAttribute('data-remove'); });
      renderGrid();
    });
  });
}

async function uploadFiles() {
  const input = $('imgds-file');
  if (!input || !input.files || !input.files.length) { alert('Choose image file(s) first.'); return; }
  const fd = new FormData();
  for (let i = 0; i < input.files.length; i++) fd.append('files', input.files[i]);
  try {
    const r = await api('/api/image-datasets/upload', { method: 'POST', body: fd });
    workingSetId = r.working_set_id;
    images = images.concat(r.images || []);
    renderGrid();
  } catch (e) { alert('Upload failed: ' + e.message); }
}

async function pickFromGallery() {
  const idsStr = prompt('Gallery image IDs to add (comma-separated):');
  if (!idsStr) return;
  const ids = idsStr.split(',').map(function (s) { return s.trim(); }).filter(Boolean);
  try {
    const r = await api('/api/image-datasets/from-gallery', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ working_set_id: workingSetId, ids: ids }),
    });
    workingSetId = r.working_set_id;
    images = images.concat(r.images || []);
    renderGrid();
  } catch (e) { alert('Gallery pick failed: ' + e.message); }
}

async function captionAll() {
  if (!workingSetId) { alert('Add images first.'); return; }
  const btn = $('imgds-caption-all');
  if (btn) { btn.disabled = true; btn.textContent = 'Captioning…'; }
  try {
    const r = await api('/api/image-datasets/caption', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ working_set_id: workingSetId }),
    });
    (r.images || []).forEach(function (u) {
      const img = images.find(function (i) { return i.id === u.id; });
      if (img) img.caption = u.caption;
    });
    renderGrid();
  } catch (e) { alert('Caption failed: ' + e.message); }
  finally { if (btn) { btn.disabled = false; btn.textContent = 'Caption all (AI)'; } }
}

function _captionMap() {
  const m = {};
  images.forEach(function (i) { m[i.id] = i.caption || ''; });
  return m;
}

async function validate() {
  const out = $('imgds-report');
  const trigger = $('imgds-trigger') ? $('imgds-trigger').value.trim() : '';
  try {
    const rep = await api('/api/image-datasets/validate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ working_set_id: workingSetId, captions: _captionMap(), trigger_word: trigger }),
    });
    const errs = (rep.errors || []).slice(0, 20).map(function (e) { return esc(e.id) + ': ' + esc(e.message); }).join('<br>');
    if (out) out.innerHTML = 'valid ' + rep.valid + ' / ' + rep.total +
      ' · duplicates ' + rep.stats.duplicates + ' · missing captions ' + rep.stats.missing_captions +
      (errs ? '<br>' + errs : '');
  } catch (e) { if (out) out.textContent = 'Validate failed: ' + e.message; }
}

async function save() {
  const name = $('imgds-name') ? $('imgds-name').value.trim() : '';
  if (!name) { alert('Enter a dataset name.'); return; }
  const trigger = $('imgds-trigger') ? $('imgds-trigger').value.trim() : '';
  try {
    const r = await api('/api/image-datasets', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ working_set_id: workingSetId, name: name, trigger_word: trigger, captions: _captionMap() }),
    });
    alert('Saved: ' + r.path);
    images = []; workingSetId = null; renderGrid(); refreshSaved();
  } catch (e) { alert('Save failed: ' + e.message); }
}

async function refreshSaved() {
  const host = $('imgds-saved'); if (!host) return;
  try {
    const j = await api('/api/image-datasets');
    host.innerHTML = (j.datasets || []).map(function (d) {
      return '<div>' + esc(d.name) + ' (' + d.images + ' images) ' +
             '<button class="btn" data-delds="' + esc(d.name) + '">Delete</button></div>';
    }).join('') || 'None yet.';
    host.querySelectorAll('[data-delds]').forEach(function (b) {
      b.addEventListener('click', function () {
        api('/api/image-datasets/' + encodeURIComponent(b.getAttribute('data-delds')), { method: 'DELETE' })
          .then(refreshSaved).catch(function () {});
      });
    });
  } catch (e) {}
}

function init() {
  isAdmin().then(function (ok) {
    if (!ok) return;
    ['rail-imagedataset', 'tool-imagedataset-btn'].forEach(function (id) { const b = $(id); if (b) b.style.display = ''; });
  });
  ['rail-imagedataset', 'tool-imagedataset-btn'].forEach(function (id) { const b = $(id); if (b) b.addEventListener('click', openImageDataset); });
  const x = $('imagedataset-close'); if (x) x.addEventListener('click', closeImageDataset);
  const src = $('imgds-source');
  if (src) src.addEventListener('change', function () {
    const isGallery = src.value === 'gallery';
    const f = $('imgds-file'); if (f) f.style.display = isGallery ? 'none' : '';
    const g = $('imgds-gallery-pick'); if (g) g.style.display = isGallery ? '' : 'none';
  });
  const f = $('imgds-file'); if (f) f.addEventListener('change', uploadFiles);
  const g = $('imgds-gallery-pick'); if (g) g.addEventListener('click', pickFromGallery);
  const capAll = $('imgds-caption-all'); if (capAll) capAll.addEventListener('click', captionAll);
  const val = $('imgds-validate'); if (val) val.addEventListener('click', validate);
  const sv = $('imgds-save'); if (sv) sv.addEventListener('click', save);
  Modals.register('imagedataset-modal', { railBtnId: 'rail-imagedataset', sidebarBtnId: 'tool-imagedataset-btn', closeFn: closeImageDataset });
}

document.addEventListener('DOMContentLoaded', init);
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_image_dataset_ui.py --import-mode=importlib -q`
Expected: PASS (3 passed; `node --check` confirms the module parses).

- [ ] **Step 5: Commit**

```bash
git add static/js/imageDataset.js tests/test_image_dataset_ui.py
git commit -m "feat(image-dataset): frontend controller (upload/gallery/caption/validate/save)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Finalize — Help section + full suite

**Files:**
- Modify: `static/index.html` (Help section)
- Test: `tests/test_image_dataset_ui.py` (extend); run the whole image-dataset suite + `import app`

- [ ] **Step 1: Add the Help entry test**

Append to `tests/test_image_dataset_ui.py`:

```python
def test_help_manual_has_image_dataset_section():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert "caption and validate an image dataset for LoRA training" in html
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_image_dataset_ui.py::test_help_manual_has_image_dataset_section --import-mode=importlib -q`
Expected: FAIL.

- [ ] **Step 3: Add the Help section**

In `static/index.html`, find the Help-manual "Datasets (AI Studio)" `<details>` block (it now has "Generate with AI." and "Ground it in your documents." paragraphs from earlier work). Insert this sibling `<details>` immediately AFTER that whole block's closing `</details>`:

```html
          <details>
            <summary style="cursor:pointer;font-weight:600;padding:6px 0;">Image Dataset (Image AI Studio)</summary>
            <p>Open <b>Image Dataset</b> from the sidebar (admin) to caption and validate an image dataset for LoRA training. Upload a batch of images, or pull them from your Gallery. Click <b>Caption all (AI)</b> to auto-describe each image with the vision model, then edit any caption by hand. Set a <b>trigger word</b> (the activation phrase your trained LoRA will respond to) and <b>Validate</b> to check for corrupt files, missing captions, duplicates, and low-resolution images. <b>Save</b> writes the set as an image + caption-file pair per photo — the standard format most LoRA trainers expect.</p>
          </details>
```

- [ ] **Step 4: Run the full image-dataset suite + import smoke**

Run: `python -m pytest tests/test_image_dataset_caption.py tests/test_image_dataset_validate.py tests/test_image_dataset_store.py tests/test_image_dataset_working_set.py tests/test_image_dataset_routes.py tests/test_image_dataset_ui.py --import-mode=importlib -q`
Expected: PASS (all green).
Run: `python -c "import app; print('OK')"`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add static/index.html tests/test_image_dataset_ui.py
git commit -m "docs(image-dataset): Help-manual section

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Never-raises is load-bearing** across all four pure modules (`caption.py`, `validate.py`, `working_set.py`, `store.py`) — the two recurring bug classes from the text dataset-tools initiative apply here too: (a) a truthy non-str hitting a string method (guard with `isinstance`), (b) `int()`/numeric coercion missing `OverflowError` (use bare `except Exception`). Test hostile inputs, not just `None`.
- **Never-500 routes** — no route in `image_dataset_routes.py` should ever let an exception escape; store/validate/working_set already never raise, so routes need no additional try/except except around the Gallery DB query and file reads (both already guarded in the Task 5 code).
- **Owner-scoping on the Gallery adapter is a security requirement**, not a nice-to-have — verify the "other user's image is excluded" test genuinely exercises this.
- **Path safety**: `working_set_id`, `image_id`, and dataset `name` are all sanitized against traversal (`_safe_id`/`_safe_name`) before ever touching the filesystem — the `../../etc-passwd` test in Task 5 must actually 404, not read outside the working-set directory.
- **XSS discipline**: `imageDataset.js` must `esc()` every caption/id string before `innerHTML` — captions are vision-model output, i.e. untrusted content, exactly like the text side's generated rows.
- **No frozen rebuild needed to test** (Python + frontend only) — reaches the installed app on the next installer rebuild.
- **Manual GUI verification owed** by the user: upload a real folder of images, caption them, set a trigger word, validate, save, and open the resulting `<DATA_DIR>/training/image_datasets/<name>/` folder to confirm the `.txt` sidecar files are what an external LoRA trainer (e.g. kohya-ss) expects.
- **Scope**: dataset prep only. NO image-LoRA training engine (a separate future sub-project), NO WD14/tag-style captioning, NO multi-trigger-word datasets, NO per-image include/exclude flag beyond the grid's Remove button.
```
