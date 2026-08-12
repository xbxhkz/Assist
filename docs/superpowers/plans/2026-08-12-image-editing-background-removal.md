# AI Image Editing Sub-project 1: Background Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user remove an image's background two ways — a new chat/agent tool operating on an uploaded attachment, and the Gallery editor's existing "Bg Remove" button, which is currently broken on Windows — both backed by a bundled U2Net ONNX model with no `rembg`/`transformers` dependency.

**Architecture:** A new module, `src/bg_removal.py`, wraps a bundled ONNX model via `onnxruntime` (already a dependency). The Gallery's existing `/api/image/remove-bg` route gets its broken `rembg`/`transformers` backend swapped for this module — no new UI, no new route. A new builtin agent tool, `remove_background`, is the first tool in this codebase to resolve a chat attachment directly (no existing tool does this today — confirmed by research, not assumed).

**Tech Stack:** `onnxruntime` (already bundled via FastEmbed), Pillow + numpy (already bundled), no new runtime dependencies.

## Global Constraints

- No new runtime dependencies — `onnxruntime`, Pillow, and numpy are all already bundled.
- The bundled model is U2Net (Apache-2.0), not BRIA RMBG (non-commercial-licensed) — Assist is publicly distributed via GitHub releases, so license matters.
- `src/bg_removal.py`'s public function is `remove_background(image_bytes: bytes, *, session=None) -> bytes` — bytes in, PNG bytes out, matching the shape both the route (via a small PIL↔bytes conversion at its call site) and the tool need.
- Tests must never require the real `u2net.onnx` file to be present — it only exists after `scripts/fetch_bg_removal_model.py` runs at build time. Every test injects a fake `session`, mirroring `src/vision/yolo.py`'s established `model=None` injectable-singleton pattern exactly.
- **Spec correction, found during this plan's own research, not guessed**: the approved spec assumed a *new* Gallery route reusing an established "attachment resolution" pattern. Neither assumption holds — `/api/image/remove-bg` already exists (broken on Windows: only knows `rembg`/`transformers`, neither usable in the frozen build) and gets its backend swapped instead of a new route being added (Task 2); and no existing builtin tool calls `upload_handler.resolve_upload()` today — `remove_background` (Task 3) is the first. Both corrections were approved by the user via AskUserQuestion before this plan was written.
- The existing chat-facing `edit_image` tool (`src/tools/image.py`, action enum including `"rembg"`) is confirmed broken for all 4 of its actions (it POSTs to `/api/gallery/{action}` routes that don't exist anywhere) — explicitly out of scope, left untouched per the user's decision.

---

### Task 1: `src/bg_removal.py` core module + model bundling

**Files:**
- Create: `src/bg_removal.py`
- Create: `scripts/fetch_bg_removal_model.py`
- Modify: `Assist.spec`
- Test: `tests/test_bg_removal.py`

**Interfaces:**
- Produces: `remove_background(image_bytes: bytes, *, session=None) -> bytes` — RGBA PNG bytes with the background made transparent. `session` is an injectable onnxruntime-session-like object (must expose `.get_inputs()[0].name` and `.run(output_names, feed_dict)`), matching `src/vision/yolo.py`'s `model=None` pattern exactly. When omitted, lazily constructs a real `onnxruntime.InferenceSession` from the bundled model, cached in a module-level singleton.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bg_removal.py`:

```python
"""bg_removal.remove_background runs U2Net inference via an injectable
onnxruntime session (mirroring src/vision/yolo.py's model=None pattern) so
tests never need the real u2net.onnx file, which only exists after
scripts/fetch_bg_removal_model.py runs at build time. See
docs/superpowers/specs/2026-08-12-image-editing-background-removal-design.md.
"""
import io

import numpy as np
from PIL import Image

from src import bg_removal


def _make_jpeg(size=(64, 64), color=(200, 50, 50)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class _FakeInput:
    name = "input"


class _FakeSession:
    def get_inputs(self):
        return [_FakeInput()]

    def run(self, output_names, feed):
        arr = np.ones((1, 1, bg_removal._MODEL_INPUT_SIZE, bg_removal._MODEL_INPUT_SIZE), dtype=np.float32)
        return [arr]


def test_remove_background_returns_rgba_png_with_injected_session():
    result = bg_removal.remove_background(_make_jpeg(), session=_FakeSession())

    out = Image.open(io.BytesIO(result))
    assert out.format == "PNG"
    assert out.mode == "RGBA"
    assert out.size == (64, 64)


def test_remove_background_preserves_original_dimensions():
    result = bg_removal.remove_background(_make_jpeg(size=(100, 50)), session=_FakeSession())

    out = Image.open(io.BytesIO(result))
    assert out.size == (100, 50)


def test_remove_background_uses_injected_session_not_the_real_one(monkeypatch):
    calls = []

    class _TrackedSession(_FakeSession):
        def run(self, output_names, feed):
            calls.append("ran")
            return super().run(output_names, feed)

    def _fail_if_called():
        raise AssertionError("should not construct a real onnxruntime session")

    monkeypatch.setattr(bg_removal, "_get_session", _fail_if_called)

    bg_removal.remove_background(_make_jpeg(), session=_TrackedSession())

    assert calls == ["ran"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bg_removal.py -v --import-mode=importlib`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.bg_removal'`.

- [ ] **Step 3: Create `src/bg_removal.py`**

```python
"""Background removal via a bundled U2Net ONNX model.

No rembg/transformers dependency: rembg is Windows-unsupported in this app's
own Cookbook UI (static/js/cookbook.js's _winUnsupported set), and the
frozen build has no pip available at runtime to install either. Mirrors
src/vision/yolo.py's lazy-singleton + injectable-session pattern so tests
never need the real model file, which only exists after
scripts/fetch_bg_removal_model.py runs at build time. See
docs/superpowers/specs/2026-08-12-image-editing-background-removal-design.md.
"""
import io
import os
import sys

import numpy as np
from PIL import Image

_MODEL_INPUT_SIZE = 320  # U2Net's standard input resolution
_session = None


def _model_path() -> str:
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        return os.path.join(base, "bg_removal", "u2net.onnx")
    return os.path.join(os.path.dirname(__file__), "..", "build_assets", "bg_removal", "u2net.onnx")


def _get_session():
    global _session
    if _session is None:
        import onnxruntime
        _session = onnxruntime.InferenceSession(_model_path())
    return _session


def _preprocess(img: Image.Image) -> np.ndarray:
    resized = img.convert("RGB").resize((_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE), Image.LANCZOS)
    arr = np.array(resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(arr, axis=0).astype(np.float32)


def remove_background(image_bytes: bytes, *, session=None) -> bytes:
    """Return RGBA PNG bytes with the background made transparent."""
    img = Image.open(io.BytesIO(image_bytes))
    original_size = img.size

    sess = session or _get_session()
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: _preprocess(img)})[0]

    mask = output[0][0]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(original_size, Image.LANCZOS)

    rgba = img.convert("RGBA")
    rgba.putalpha(mask_img)

    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return buf.getvalue()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bg_removal.py -v --import-mode=importlib`
Expected: PASS (all 3 tests).

- [ ] **Step 5: Create `scripts/fetch_bg_removal_model.py`**

This mirrors `scripts/fetch_sd_server.py`'s raw-`urllib.request`-download pattern (a single file here, no zip extraction needed) plus `scripts/fetch_llama_server.py`'s idempotency check (skip if already present). Note the docstring's explicit call-out: the exact URL is from training-knowledge memory of where the `rembg` project (danielgatis/rembg) hosts its pre-exported ONNX models as GitHub Release assets, not freshly verified — confirm it's live with `curl -sIL <url>` before this is relied on in a real release build; no existing fetch script in this codebase does checksum verification, so none is added here either, matching established convention.

```python
"""Fetch the U2Net ONNX background-removal model into build_assets/ so it
can be bundled into the frozen build (Assist.spec collects the whole
build_assets/bg_removal/ directory, mirroring build_assets/yolo/). Mirrors
scripts/fetch_sd_server.py's raw-download pattern -- a single file here, no
zip extraction needed.

NOTE: MODEL_URL below is from training-knowledge memory of where the rembg
project (danielgatis/rembg) hosts its pre-exported ONNX models as GitHub
Release assets -- NOT freshly verified. Confirm with `curl -sIL <url>`
before relying on this in a real release build. No checksum verification
exists here, matching every sibling fetch_*.py script in this repo (none of
them verify a hash either -- only existence-after-write is checked).
"""
import os
import sys
import urllib.request

ASSET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "build_assets", "bg_removal")
)
MODEL_URL = os.getenv(
    "BG_REMOVAL_MODEL_URL",
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
)
MODEL_PATH = os.path.join(ASSET_DIR, "u2net.onnx")


def main() -> int:
    if os.path.isfile(MODEL_PATH):
        print(f"Background-removal model already present at {MODEL_PATH}, skipping.")
        return 0

    os.makedirs(ASSET_DIR, exist_ok=True)
    print(f"Downloading background-removal model from {MODEL_URL} ...")
    try:
        with urllib.request.urlopen(MODEL_URL) as resp:  # noqa: S310
            data = resp.read()
    except Exception as e:
        print(f"ERROR: failed to download model: {e}", file=sys.stderr)
        return 1

    with open(MODEL_PATH, "wb") as f:
        f.write(data)

    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH} after download", file=sys.stderr)
        return 1

    print(f"Background-removal model saved to {MODEL_PATH} ({len(data)} bytes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Bundle the model directory in `Assist.spec`**

Find this line:

```python
    ('build_assets/yolo', 'yolo'),
```

Immediately after it, insert:

```python
    ('build_assets/bg_removal', 'bg_removal'),
```

- [ ] **Step 7: Commit**

```bash
git add src/bg_removal.py scripts/fetch_bg_removal_model.py Assist.spec tests/test_bg_removal.py
git commit -m "feat(bg-removal): U2Net ONNX background-removal module + model bundling

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Fix the existing `/api/image/remove-bg` route

**Files:**
- Modify: `routes/gallery/gallery_routes.py`
- Test: `tests/test_gallery_remove_bg_route.py`

**Interfaces:**
- Consumes: `remove_background(image_bytes, *, session=None)` (Task 1), imported under an alias to avoid colliding with this route handler's own name (the route function is ALSO named `remove_background` — see Step 3).

**Context**: this route already exists, is already wired end-to-end to a real "Bg Remove" button in the Gallery editor, and already handles a `hint_mask`-based crop/compose/alpha-multiply flow correctly — only its model-inference step is broken (only knows `rembg`/`transformers`, neither usable in the frozen Windows build). This task replaces ONLY that step; the surrounding hint/crop/compose/alpha logic is untouched.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gallery_remove_bg_route.py`:

```python
"""POST /api/image/remove-bg's model-inference step now calls
src.bg_removal.remove_background instead of the broken rembg/transformers
imports (rembg is Windows-unsupported; transformers/RMBG-1.4 is not
bundled). The surrounding hint-mask crop/compose/alpha-multiply logic is
unchanged. See
docs/superpowers/specs/2026-08-12-image-editing-background-removal-design.md.
"""
import asyncio
import base64
import io
from types import SimpleNamespace

from PIL import Image

import routes.gallery.gallery_routes as gallery_routes_module


def _route(router, path, method):
    for r in router.routes:
        if r.path == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(path)


def _png_bytes(size=(40, 40), color=(10, 20, 30, 255)):
    img = Image.new("RGBA", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def test_remove_bg_route_calls_bg_removal_module(monkeypatch):
    monkeypatch.setattr(gallery_routes_module, "require_privilege", lambda request, priv: "alice")

    captured = {}

    def fake_run_bg_removal_model(image_bytes, **kwargs):
        captured["called_with_bytes"] = image_bytes
        # Return a same-size RGBA PNG so the route's compose step succeeds.
        out = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()

    monkeypatch.setattr(gallery_routes_module, "run_bg_removal_model", fake_run_bg_removal_model)

    router = gallery_routes_module.setup_gallery_routes()
    handler = _route(router, "/api/image/remove-bg", "POST")

    image_b64 = base64.b64encode(_png_bytes()).decode()
    out = asyncio.run(handler(_FakeRequest({"image": image_b64})))

    assert "called_with_bytes" in captured
    assert "image" in out
    result_img = Image.open(io.BytesIO(base64.b64decode(out["image"])))
    assert result_img.mode == "RGBA"


def test_remove_bg_route_returns_error_on_model_failure(monkeypatch):
    monkeypatch.setattr(gallery_routes_module, "require_privilege", lambda request, priv: "alice")

    def failing_run_bg_removal_model(image_bytes, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(gallery_routes_module, "run_bg_removal_model", failing_run_bg_removal_model)

    router = gallery_routes_module.setup_gallery_routes()
    handler = _route(router, "/api/image/remove-bg", "POST")

    image_b64 = base64.b64encode(_png_bytes()).decode()
    out = asyncio.run(handler(_FakeRequest({"image": image_b64})))

    assert "error" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gallery_remove_bg_route.py -v --import-mode=importlib`
Expected: FAIL — `AttributeError: module 'routes.gallery.gallery_routes' has no attribute 'run_bg_removal_model'`.

- [ ] **Step 3: Add the import**

Find this line:

```python
from src.auth_helpers import get_current_user, owner_filter, require_privilege
```

Immediately after it, insert:

```python
from src.bg_removal import remove_background as run_bg_removal_model
```

(Aliased on import — the route handler function a few hundred lines below is ALSO named `remove_background`; importing the module function under its own name at module scope would be shadowed by that nested function definition, since Python resolves the name from the enclosing scope where the route function itself is bound. The alias avoids this collision entirely.)

- [ ] **Step 4: Replace the broken model-inference block**

Find this exact block:

```python
        try:
            from rembg import remove
            cut = remove(crop)
        except ImportError:
            try:
                from transformers import pipeline
                pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
                mask_img = pipe(crop, return_mask=True).convert("L")
                tmp = crop.copy()
                tmp.putalpha(mask_img)
                cut = tmp
            except Exception:
                return {"error": "No background removal model available. Install rembg: pip install rembg"}
```

Replace it with:

```python
        crop_buf = io.BytesIO()
        crop.convert("RGBA").save(crop_buf, format="PNG")
        try:
            cut_bytes = run_bg_removal_model(crop_buf.getvalue())
        except Exception as e:
            logger.warning("Background removal failed", exc_info=True)
            return {"error": f"Background removal failed: {e}"}
        cut = Image.open(io.BytesIO(cut_bytes)).convert("RGBA")
```

Everything else in the function (the `hint_mask`/`bbox` cropping before this block, and the compose/alpha-multiply logic after it) stays exactly as it is — do not modify it.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_gallery_remove_bg_route.py -v --import-mode=importlib`
Expected: PASS (both tests).

Run: `pytest tests/test_gallery_routes.py -v --import-mode=importlib` (or whatever the actual existing Gallery route test filename is — search `tests/` first) to confirm no regressions to the rest of the Gallery routes file.

- [ ] **Step 6: Commit**

```bash
git add routes/gallery/gallery_routes.py tests/test_gallery_remove_bg_route.py
git commit -m "fix(gallery): /api/image/remove-bg now uses the bundled ONNX model, not broken rembg/transformers

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `remove_background` builtin agent tool

**Files:**
- Create: `src/agent_tools/image_tools.py`
- Modify: `src/agent_tools/__init__.py` (2 registration points: import + `TOOL_HANDLERS`, plus `TOOL_TAGS`)
- Modify: `src/tool_schemas.py` (1 registration point: `FUNCTION_TOOL_SCHEMAS`)
- Modify: `src/agent_loop.py` (2 registration points: `TOOL_SECTIONS`, `_DOMAIN_TOOL_MAP["desktop"]`)
- Modify: `src/tool_index.py` (1 registration point: `BUILTIN_TOOL_DESCRIPTIONS`)
- Test: `tests/test_remove_background_tool.py`
- Test: `tests/test_remove_background_registration.py`

**Interfaces:**
- Consumes: `src.bg_removal.remove_background(image_bytes, *, session=None)` (Task 1), `src.upload_handler.UploadHandler.resolve_upload(upload_id, owner=None, ...)` (existing).
- Produces: `remove_background_tool(content, ctx, *, remover=None, upload_resolver=None)` in `src/agent_tools/image_tools.py`, and `RemoveBackgroundTool` (a thin `execute(self, content, ctx)` wrapper class), registered as the builtin tool name `"remove_background"`.

**This is the first builtin tool to call `resolve_upload()` directly** — no existing tool does this (chat-uploaded images are analyzed in message preprocessing today, before any tool runs). There is also no existing singleton accessor for the app's `UploadHandler` instance reachable from a `Tool.execute()` — `ctx` only ever contains `progress_cb`/`subproc_env`/`session_id`/`owner`. This task's tool constructs its own throwaway `UploadHandler` instance instead, mirroring an existing precedent (`routes/document_helpers.py:178-183` already does exactly this independently of `app.state.upload_handler`, since `resolve_upload`'s read path needs no cross-request state).

**Another self-review correction**: the approved spec's Data Flow section says the result is "saved as a new Gallery image AND returned inline" for the chat path specifically (the Gallery-editor path, per Task 2's correction, already has its own existing manual "Save" step in the editor UI, so no auto-save is needed there — that's unchanged, pre-existing behavior, not a gap). Without Gallery-persistence, a chat-tool-returned image would exist only in that one chat message with no way to find it again — exactly the downside the user's own "save to Gallery" choice was meant to avoid. This tool therefore also writes a new `GalleryImage` row, using the exact field set `POST /api/gallery/upload` itself uses (`routes/gallery/gallery_routes.py:230-248`) — confirmed via direct read, not guessed: `id`, `filename`, `prompt`, `model`, `owner`, `file_hash`, `file_size` (only `id`/`filename`/`owner`/`file_hash`/`file_size` are populated meaningfully here; EXIF-derived fields like `width`/`height` don't apply to a synthetically-generated PNG and are left at their column defaults). **Verify `GENERATED_IMAGES_DIR`'s exact import source** (very likely `src.constants`, matching every other persisted-path constant's established home, e.g. `DATA_DIR`/`UPLOAD_DIR` — confirm with a quick grep before writing the import) before finalizing this step.

**On admin-gating**: unlike `webcam_look`/`diagnose_equipment` (admin-only, desktop/privacy-sensitive), the most directly analogous EXISTING tool for this feature is `generate_image` — the Gallery's own `/api/image/remove-bg` route gates by the `can_generate_images` *privilege*, not a blanket admin check. **Before writing `NON_ADMIN_BLOCKED_TOOLS`**, grep `src/tool_security.py` for `"generate_image"` and confirm whether it's a member of `NON_ADMIN_BLOCKED_TOOLS`. Mirror whatever you find exactly — add `"remove_background"` to that set ONLY if `"generate_image"` is also there; leave it out otherwise. Do not default to admin-only without checking this.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_remove_background_tool.py`:

```python
"""remove_background_tool resolves a chat attachment (the first builtin
tool to call upload_handler.resolve_upload() directly -- no existing tool
does this today), runs it through src.bg_removal, and returns an inline
data: URI via the established image_url tool-result convention (the same
mechanism generate_image/webcam_look already use). Never raises into the
agent loop, matching diagnose_equipment's established pattern. See
docs/superpowers/specs/2026-08-12-image-editing-background-removal-design.md.
"""
import asyncio
import base64
import json

import pytest

from src.agent_tools.image_tools import RemoveBackgroundTool, remove_background_tool


def _fake_upload_resolver(found=True, path="/tmp/fake.png"):
    def resolver(upload_id, owner=None):
        if not found:
            return None
        return {"id": upload_id, "path": path, "name": "fake.png", "mime": "image/png"}
    return resolver


def _fake_remover(output=b"fake-png-bytes"):
    def remover(image_bytes, **kwargs):
        return output
    return remover


def test_missing_attachment_id_returns_error():
    result = asyncio.run(remove_background_tool("{}", {"owner": "alice"}))
    assert "error" in result


def test_invalid_json_returns_error():
    result = asyncio.run(remove_background_tool("not json", {"owner": "alice"}))
    assert "error" in result


def test_unresolvable_attachment_returns_error():
    content = json.dumps({"attachment_id": "missing-id"})
    result = asyncio.run(remove_background_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=False),
        remover=_fake_remover(),
    ))
    assert "error" in result


def _fake_gallery_saver(image_id="gallery-img-1"):
    calls = []

    def saver(image_bytes, owner):
        calls.append((image_bytes, owner))
        return image_id

    saver.calls = calls
    return saver


def test_successful_removal_returns_image_url_and_saves_to_gallery(tmp_path):
    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})
    gallery_saver = _fake_gallery_saver(image_id="gallery-img-1")

    result = asyncio.run(remove_background_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        remover=_fake_remover(output=b"removed-bg-bytes"),
        gallery_saver=gallery_saver,
    ))

    assert "image_url" in result
    assert result["image_url"].startswith("data:image/png;base64,")
    decoded = base64.b64decode(result["image_url"].split(",", 1)[1])
    assert decoded == b"removed-bg-bytes"
    assert result["gallery_image_id"] == "gallery-img-1"
    assert gallery_saver.calls == [(b"removed-bg-bytes", "alice")]


def test_model_failure_returns_error_not_raise(tmp_path):
    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})

    def failing_remover(image_bytes, **kwargs):
        raise RuntimeError("boom")

    result = asyncio.run(remove_background_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        remover=failing_remover,
        gallery_saver=_fake_gallery_saver(),
    ))

    assert "error" in result


def test_gallery_save_failure_still_returns_image_url(tmp_path):
    # Saving to Gallery is a best-effort convenience, not the primary
    # deliverable -- if it fails, the user should still get the image
    # inline rather than losing the whole result.
    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})

    def failing_saver(image_bytes, owner):
        raise RuntimeError("db unavailable")

    result = asyncio.run(remove_background_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        remover=_fake_remover(output=b"removed-bg-bytes"),
        gallery_saver=failing_saver,
    ))

    assert "image_url" in result
    assert "gallery_image_id" not in result


def test_tool_class_delegates_to_module_function():
    tool = RemoveBackgroundTool()
    result = asyncio.run(tool.execute("{}", {"owner": "alice"}))
    assert "error" in result
```

Create `tests/test_remove_background_registration.py` (mirrors `tests/test_desktop_registration.py`'s structure but as a standalone, tool-specific parity test — not added to that file's own hardcoded `DESKTOP` list, since this tool's `NON_ADMIN_BLOCKED_TOOLS` membership is determined by mirroring `generate_image`'s, not assumed to match the admin-only desktop tools):

```python
"""remove_background must be registered everywhere a builtin tool needs to
be -- this codebase has a documented gotcha (found building webcam_look)
that a new tool needs the full registration set, not just a handler entry.
See docs/superpowers/specs/2026-08-12-image-editing-background-removal-design.md."""


def test_remove_background_registered_everywhere():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    from src.agent_loop import TOOL_SECTIONS, _DOMAIN_TOOL_MAP
    from src.tool_index import BUILTIN_TOOL_DESCRIPTIONS

    assert "remove_background" in TOOL_HANDLERS
    assert "remove_background" in TOOL_TAGS
    names = [(s.get("function") or {}).get("name") for s in FUNCTION_TOOL_SCHEMAS]
    assert "remove_background" in names
    assert "remove_background" in TOOL_SECTIONS
    assert "remove_background" in _DOMAIN_TOOL_MAP["desktop"]
    assert "remove_background" in BUILTIN_TOOL_DESCRIPTIONS


def test_remove_background_matches_generate_image_admin_gating():
    from src.tool_security import NON_ADMIN_BLOCKED_TOOLS
    generate_image_blocked = "generate_image" in NON_ADMIN_BLOCKED_TOOLS
    remove_background_blocked = "remove_background" in NON_ADMIN_BLOCKED_TOOLS
    assert remove_background_blocked == generate_image_blocked, (
        "remove_background's NON_ADMIN_BLOCKED_TOOLS membership must mirror "
        "generate_image's, not default to admin-only"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_remove_background_tool.py tests/test_remove_background_registration.py -v --import-mode=importlib`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agent_tools.image_tools'`.

- [ ] **Step 3: Create `src/agent_tools/image_tools.py`**

```python
"""The `remove_background` builtin tool: strip the background from an
already-uploaded chat image attachment, returning a transparent PNG inline
in the chat response via the established image_url convention. Runs the
bundled U2Net ONNX model (src/bg_removal.py) -- no rembg/transformers
dependency. This is the first builtin tool to call
upload_handler.resolve_upload() directly; no existing accessor for the
app's UploadHandler singleton is reachable from a Tool's ctx, so this
constructs its own throwaway instance, mirroring
routes/document_helpers.py's existing precedent for the same reason (the
read path needs no cross-request state). NEVER raises into the agent --
every failure returns {"error": ...}, matching diagnose_equipment's
established pattern. See
docs/superpowers/specs/2026-08-12-image-editing-background-removal-design.md.
"""
import base64
import json


def _default_gallery_saver(image_bytes, owner):
    """Persist a new Gallery image, mirroring POST /api/gallery/upload's own
    GalleryImage field set exactly (routes/gallery/gallery_routes.py:230-248),
    minus the EXIF-derived fields that don't apply to a synthetically
    generated PNG. Returns the new image's id."""
    import hashlib
    import uuid
    from pathlib import Path

    from core.database import GalleryImage, SessionLocal
    from src.constants import GENERATED_IMAGES_DIR

    db = SessionLocal()
    try:
        img_dir = Path(GENERATED_IMAGES_DIR)
        img_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex[:12]}.png"
        (img_dir / filename).write_bytes(image_bytes)

        img_id = str(uuid.uuid4())
        db.add(GalleryImage(
            id=img_id,
            filename=filename,
            prompt="Background removed",
            model="remove_background",
            owner=owner,
            file_hash=hashlib.sha256(image_bytes).hexdigest(),
            file_size=len(image_bytes),
        ))
        db.commit()
        return img_id
    finally:
        db.close()


async def remove_background_tool(content, ctx, *, remover=None, upload_resolver=None, gallery_saver=None):
    ctx = ctx or {}
    owner = ctx.get("owner")

    try:
        args = json.loads(content) if content and content.strip() else {}
        if not isinstance(args, dict):
            return {"error": "remove_background: arguments must be a JSON object"}
    except (ValueError, TypeError):
        return {"error": "remove_background: arguments must be valid JSON"}

    attachment_id = args.get("attachment_id")
    if not isinstance(attachment_id, str) or not attachment_id.strip():
        return {"error": "remove_background: an 'attachment_id' is required"}

    if upload_resolver is None:
        from src.constants import DATA_DIR, UPLOAD_DIR
        from src.upload_handler import UploadHandler
        upload_resolver = UploadHandler(DATA_DIR, UPLOAD_DIR).resolve_upload

    try:
        info = upload_resolver(attachment_id, owner=owner)
    except Exception as e:
        return {"error": f"remove_background: could not resolve attachment: {e}"}

    if not info or not info.get("path"):
        return {"error": f"remove_background: attachment '{attachment_id}' not found"}

    try:
        with open(info["path"], "rb") as f:
            image_bytes = f.read()
    except OSError as e:
        return {"error": f"remove_background: could not read attachment: {e}"}

    if remover is None:
        from src.bg_removal import remove_background as remover

    try:
        result_bytes = remover(image_bytes)
    except Exception as e:
        return {"error": f"remove_background: model failed: {e}"}

    image_url = "data:image/png;base64," + base64.b64encode(result_bytes).decode("ascii")
    result = {"output": "Background removed.", "image_url": image_url}

    # Best-effort: saving to Gallery makes the result findable later, but
    # isn't the primary deliverable -- a save failure must not lose the
    # image the model already successfully produced.
    saver = gallery_saver or _default_gallery_saver
    try:
        result["gallery_image_id"] = saver(result_bytes, owner)
    except Exception:
        pass

    return result


class RemoveBackgroundTool:
    async def execute(self, content, ctx):
        return await remove_background_tool(content, ctx)
```

- [ ] **Step 4: Register in `src/agent_tools/__init__.py`**

Find this line:

```python
from .industrial_tools import DiagnoseEquipmentTool
```

Immediately after it, insert:

```python
from .image_tools import RemoveBackgroundTool
```

Find this line:

```python
    "diagnose_equipment": DiagnoseEquipmentTool().execute,
```

Immediately after it, insert:

```python
    "remove_background": RemoveBackgroundTool().execute,
```

Find the `TOOL_TAGS` set's `"diagnose_equipment",` entry and add `"remove_background",` immediately after it, matching the set's existing formatting.

- [ ] **Step 5: Add the schema in `src/tool_schemas.py`**

Find the `diagnose_equipment` entry in `FUNCTION_TOOL_SCHEMAS` (around line 738-754) and insert this new entry immediately after its closing `},`:

```python
    {
        "type": "function",
        "function": {
            "name": "remove_background",
            "description": "Remove the background from an image the user uploaded in this chat, returning a transparent PNG.",
            "parameters": {
                "type": "object",
                "properties": {
                    "attachment_id": {"type": "string", "description": "The id of the uploaded image attachment"},
                },
                "required": ["attachment_id"],
            },
        },
    },
```

- [ ] **Step 6: Add to `TOOL_SECTIONS` and `_DOMAIN_TOOL_MAP` in `src/agent_loop.py`**

Find the `TOOL_SECTIONS` dict's `"webcam_look"` entry and insert this new entry immediately after it:

```python
    "remove_background": """```remove_background
{"attachment_id": "<id from an uploaded image>"}
```
Remove the background from an image the user uploaded in chat, returning a transparent PNG shown inline in your response.""",
```

Find this block:

```python
    "desktop": {"launch_app", "find_files", "list_windows", "control_window", "capture_screen",
                "webcam_look", "diagnose_equipment",
                "ingest_equipment_manual", "search_equipment_manual",
                "list_ui_elements", "click_element", "set_element_text", "mouse", "keyboard"},
```

Replace it with:

```python
    "desktop": {"launch_app", "find_files", "list_windows", "control_window", "capture_screen",
                "webcam_look", "diagnose_equipment", "remove_background",
                "ingest_equipment_manual", "search_equipment_manual",
                "list_ui_elements", "click_element", "set_element_text", "mouse", "keyboard"},
```

- [ ] **Step 7: Add to `BUILTIN_TOOL_DESCRIPTIONS` in `src/tool_index.py`**

Find the `"webcam_look"` entry in `BUILTIN_TOOL_DESCRIPTIONS` and insert this new entry immediately after it:

```python
    "remove_background": "Remove the background from an uploaded chat image, returning a transparent PNG.",
```

- [ ] **Step 8: Resolve the `NON_ADMIN_BLOCKED_TOOLS` question**

Read `src/tool_security.py` and search for `"generate_image"`. If it appears in `NON_ADMIN_BLOCKED_TOOLS`, add `"remove_background"` alongside it in that same set. If it does NOT appear there, do not add `"remove_background"` either — leave `NON_ADMIN_BLOCKED_TOOLS` untouched. This is verified, not guessed — `tests/test_remove_background_registration.py`'s `test_remove_background_matches_generate_image_admin_gating` (Step 1) enforces whichever answer is correct.

- [ ] **Step 9: Run tests to verify they pass**

Run: `pytest tests/test_remove_background_tool.py tests/test_remove_background_registration.py -v --import-mode=importlib`
Expected: PASS (7 + 2 = 9 tests).

Run: `pytest tests/test_desktop_registration.py -v --import-mode=importlib` to confirm the pre-existing desktop-tool parity test still passes unaffected (this task doesn't touch that test file's own `DESKTOP` list).

Run: `python -c "import app"` to confirm the app still imports cleanly with the new tool wired in.

- [ ] **Step 10: Commit**

```bash
git add src/agent_tools/image_tools.py src/agent_tools/__init__.py src/tool_schemas.py src/agent_loop.py src/tool_index.py src/tool_security.py tests/test_remove_background_tool.py tests/test_remove_background_registration.py
git commit -m "feat(bg-removal): remove_background builtin agent tool

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

(Note: `src/tool_security.py` is only actually modified if Step 8 determined `remove_background` belongs in `NON_ADMIN_BLOCKED_TOOLS` — if not, omit it from this `git add`.)
