# AI Image Editing Sub-project 3: Face-Swap Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user supply a source-face image and a target image and get back the target with the source face swapped in — via chat or the Gallery editor — with the underlying model obtained only through explicit, informed license acceptance, and every output carrying provenance metadata identifying it as AI-face-edited.

**Architecture:** A new module, `src/face_swap.py`, runs InsightFace's face-detection/alignment step followed by the InSwapper ONNX model. Neither model is bundled in Assist's installer — both are licensed non-commercial/research-only by InsightFace, confirmed by direct research to cover the detection pack (`buffalo_l`) exactly as it covers the swap model, and InsightFace's own library auto-downloads them with no consent hook. The module gates on an explicit `face_swap_license_accepted` setting before ever letting that downloader run. Two entry points — a new builtin chat tool and a Gallery editor addition — both call the same core function, matching sub-projects 1 and 2's established pattern.

**Tech Stack:** `insightface` (new bundled dependency — this app's first ML package requiring gated, non-bundled model weights, unlike every prior sub-project's auto-fetched models), `onnxruntime` (already bundled), Pillow + numpy (already bundled).

**Spec:** `docs/superpowers/specs/2026-08-13-image-editing-face-swap-design.md`

## Global Constraints

- No visible watermark — provenance marking is metadata-only, per the approved design.
- Neither InsightFace model pack (`buffalo_l` detection, `inswapper_128.onnx` swap) is ever fetched without `face_swap_license_accepted` being `True` first. This check happens inside `src/face_swap.py` itself, before any InsightFace API call that could trigger its own implicit downloader — never at a call site that could be bypassed.
- `insightface` ships as a bundled Python dependency (added to `requirements.txt` + `Assist.spec`'s `collect_all` loop), per the user's explicit choice — NOT an optional/Cookbook-style manual-install feature.
- Still images only. No tunable parameters exposed to the model or user. No bundled face catalog. No auto-publish/sharing integration. No identity/consent verification of depicted people. (All per the approved spec's Out of Scope section.)
- The new builtin tool's registration applies every lesson already learned building `remove_background`/`edit_image_prompt`, from day one: the 6 standard points, the `src/tool_execution.py` dispatcher owner-threading branch, the `can_generate_images` privilege gate in `routes/chat_routes.py`, and the `_PLAN_MODE_KNOWN_MUTATORS` entry in `src/tool_security.py`. `NON_ADMIN_BLOCKED_TOOLS` gets no new entry (mirrors `generate_image`, confirmed absent by direct research against the current file).
- Tests must never require the real InsightFace model weights (which don't exist in the dev/test environment) or a real license acceptance — every test injects fake `analyzer`/`swapper` objects, mirroring `src/bg_removal.py`'s established `session=None` injectable pattern.

---

### Task 1: Bundle `insightface` as a build dependency

**Files:**
- Modify: `requirements.txt`
- Modify: `Assist.spec`
- Test: `tests/test_insightface_packaging.py`

**Interfaces:**
- Produces: `insightface` becomes importable in this dev environment and is collected by PyInstaller in the frozen build. No application code changes yet — this task only makes the package available for Task 2 to build on.

**Context**: `insightface` is not installed in this environment or referenced anywhere as a real dependency today (its only prior mention, `routes/shell_routes.py`'s Cookbook pip-install allowlist, is confirmed a no-op in the frozen build — pip isn't bundled there). This task adds it the same way `onnxruntime`/`ultralytics` are already bundled: declared in `requirements.txt`, installed into this dev environment, and collected via `Assist.spec`'s existing `collect_all` loop.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_insightface_packaging.py`:

```python
"""insightface must be a real, importable dependency (not the Cookbook
pip-install allowlist, which is a no-op in the frozen build) and must be
collected by PyInstaller in Assist.spec, mirroring onnxruntime/ultralytics.
See docs/superpowers/specs/2026-08-13-image-editing-face-swap-design.md."""
from pathlib import Path

_SPEC_FILE = Path(__file__).resolve().parent.parent / "Assist.spec"
_REQUIREMENTS_FILE = Path(__file__).resolve().parent.parent / "requirements.txt"


def test_insightface_importable():
    import insightface  # noqa: F401


def test_face_analysis_class_importable():
    from insightface.app import FaceAnalysis
    assert callable(FaceAnalysis)


def test_model_zoo_get_model_importable():
    from insightface.model_zoo import get_model
    assert callable(get_model)


def test_requirements_declares_insightface():
    text = _REQUIREMENTS_FILE.read_text(encoding="utf-8")
    assert "insightface" in text


def test_assist_spec_collects_insightface():
    text = _SPEC_FILE.read_text(encoding="utf-8")
    assert '"insightface"' in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_insightface_packaging.py -v --import-mode=importlib`
Expected: FAIL — `ModuleNotFoundError: No module named 'insightface'` on the first three tests; the two source-text tests also fail (string not present yet).

- [ ] **Step 3: Add `insightface` to `requirements.txt`**

Find this line:

```
ultralytics
```

Immediately after it (before the blank line/`pytest` section that follows), insert:

```
# Face-swap tooling (the face_swap tool): InsightFace provides face
# detection/alignment + the pretrained swap model. The Python package ships
# by default so the pipeline is available once the (separately gated, never
# auto-bundled) model weights are downloaded -- see src/face_swap.py. The
# frozen build bundles the package itself via collect_all in Assist.spec;
# the model weights are not part of that bundle.
insightface
```

- [ ] **Step 4: Install into this dev environment**

Run: `pip install insightface`
Expected: installs successfully (pulls in `onnx`, `scikit-image`, and other transitive dependencies automatically).

If installation fails or produces a version conflict with an already-pinned dependency in this environment, STOP and report — do not force an install that breaks other packages; escalate for guidance rather than guessing at a resolution.

- [ ] **Step 5: Bundle it in `Assist.spec`**

Find this line:

```python
             "asyncua"):
```

Replace it with:

```python
             "asyncua",
             # Face-swap tooling (face_swap tool): InsightFace's face
             # detection/alignment + swap-model runtime. Model WEIGHTS
             # (buffalo_l, inswapper_128.onnx) are gated behind explicit
             # license acceptance (src/face_swap.py) and are never bundled
             # here -- only the Python package/runtime is.
             "insightface"):
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_insightface_packaging.py -v --import-mode=importlib`
Expected: PASS (all 5 tests).

- [ ] **Step 7: Commit**

```bash
git add requirements.txt Assist.spec tests/test_insightface_packaging.py
git commit -m "build(face-swap): bundle insightface as a build dependency

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `src/face_swap.py` core module

**Files:**
- Create: `src/face_swap.py`
- Test: `tests/test_face_swap.py`

**Interfaces:**
- Consumes: `insightface.app.FaceAnalysis`, `insightface.model_zoo.get_model` (Task 1); `src.settings.get_setting` (existing).
- Produces:
  - `swap_face(source_face_bytes: bytes, target_image_bytes: bytes, *, analyzer=None, swapper=None) -> bytes` — PNG bytes in (both images), PNG bytes out (with embedded provenance metadata). Raises `LicenseNotAcceptedError`, `NoFaceDetectedError`, or lets underlying inference errors propagate — callers apply the never-raises discipline at their own boundary, matching `src/bg_removal.py`'s convention.
  - `LicenseNotAcceptedError` and `NoFaceDetectedError` — both raised by this module, both importable from `src.face_swap`.

**Before writing the implementation, verify the real API against the actually-installed package** (available after Task 1) rather than trusting the shapes below blindly — they're drawn from InsightFace's own published usage examples, not from running the code in this repo. Specifically confirm: (a) `FaceAnalysis.__init__`'s exact kwarg for redirecting where model files are read from/downloaded to (the plan below assumes `root=`; verify this is correct, e.g. via `help(insightface.app.FaceAnalysis.__init__)` or reading the installed package's source under site-packages), (b) `insightface.model_zoo.get_model`'s exact signature and whether it accepts a direct file path or a bare model name, (c) the `Face` object's attribute names returned by `FaceAnalysis.get()` (the plan below only relies on truthiness of the returned list and passes `Face` objects through opaquely to `swapper.get()`, so this is lower-risk, but confirm `analyzer.get(img)` returns a plain Python list, not something else falsy-but-non-empty). If anything differs from what's below, adjust the implementation accordingly and note the deviation in your report — don't silently guess past a mismatch.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_face_swap.py`:

```python
"""face_swap.swap_face runs InsightFace's detection + InSwapper via
injectable analyzer/swapper objects, mirroring src/bg_removal.py's
session=None pattern -- tests never need the real (gated, non-bundled)
model weights or a real license acceptance. See
docs/superpowers/specs/2026-08-13-image-editing-face-swap-design.md.
"""
import io

import numpy as np
import pytest
from PIL import Image

from src import face_swap


def _make_png(size=(64, 64), color=(200, 50, 50)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeFace:
    pass


class _FakeAnalyzer:
    def __init__(self, faces_by_call):
        self._faces_by_call = list(faces_by_call)

    def get(self, img):
        return self._faces_by_call.pop(0)


class _FakeSwapper:
    def get(self, target_img, target_face, source_face, paste_back=True):
        # Return a same-shape array so PIL can re-encode it as a valid PNG.
        return target_img


def test_swap_face_returns_png_bytes_with_provenance_metadata(monkeypatch):
    monkeypatch.setattr(face_swap, "get_setting", lambda key, default=None: True)
    analyzer = _FakeAnalyzer([[_FakeFace()], [_FakeFace()]])
    swapper = _FakeSwapper()

    result = face_swap.swap_face(
        _make_png(color=(10, 20, 30)), _make_png(color=(200, 100, 50)),
        analyzer=analyzer, swapper=swapper,
    )

    out = Image.open(io.BytesIO(result))
    assert out.format == "PNG"
    assert out.text.get("assist:ai-edited") == "face-swap"


def test_swap_face_raises_when_license_not_accepted(monkeypatch):
    monkeypatch.setattr(face_swap, "get_setting", lambda key, default=None: False)

    with pytest.raises(face_swap.LicenseNotAcceptedError):
        face_swap.swap_face(
            _make_png(), _make_png(),
            analyzer=_FakeAnalyzer([[_FakeFace()], [_FakeFace()]]),
            swapper=_FakeSwapper(),
        )


def test_swap_face_raises_when_no_face_in_source(monkeypatch):
    monkeypatch.setattr(face_swap, "get_setting", lambda key, default=None: True)
    analyzer = _FakeAnalyzer([[]])  # source: no faces found

    with pytest.raises(face_swap.NoFaceDetectedError):
        face_swap.swap_face(
            _make_png(), _make_png(),
            analyzer=analyzer, swapper=_FakeSwapper(),
        )


def test_swap_face_raises_when_no_face_in_target(monkeypatch):
    monkeypatch.setattr(face_swap, "get_setting", lambda key, default=None: True)
    analyzer = _FakeAnalyzer([[_FakeFace()], []])  # source ok, target: none

    with pytest.raises(face_swap.NoFaceDetectedError):
        face_swap.swap_face(
            _make_png(), _make_png(),
            analyzer=analyzer, swapper=_FakeSwapper(),
        )


def test_swap_face_does_not_construct_real_analyzer_when_injected(monkeypatch):
    """Injection must genuinely bypass the real (license-gated, model-file-
    requiring) singleton construction, not just the swap call."""
    monkeypatch.setattr(face_swap, "get_setting", lambda key, default=None: True)

    def _fail_if_called():
        raise AssertionError("should not construct a real FaceAnalysis")

    monkeypatch.setattr(face_swap, "_get_analyzer", _fail_if_called)
    monkeypatch.setattr(face_swap, "_get_swapper", _fail_if_called)

    face_swap.swap_face(
        _make_png(), _make_png(),
        analyzer=_FakeAnalyzer([[_FakeFace()], [_FakeFace()]]),
        swapper=_FakeSwapper(),
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_face_swap.py -v --import-mode=importlib`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.face_swap'`.

- [ ] **Step 3: Create `src/face_swap.py`**

```python
"""Face-swap via a locally-run InsightFace pipeline (face detection/
alignment + the pretrained InSwapper model). Both the detection model pack
(buffalo_l) and the swap model (inswapper_128.onnx) are licensed for
non-commercial research use only by InsightFace -- neither is bundled in
Assist's installer or fetched automatically. This module never lets
InsightFace's own implicit downloader run without the user's explicit,
recorded acceptance (the face_swap_license_accepted setting) -- see
_ensure_models_available(). Output PNGs carry embedded provenance metadata
identifying them as AI-face-swapped (metadata only, no visible watermark,
per the approved design). See
docs/superpowers/specs/2026-08-13-image-editing-face-swap-design.md.
"""
import io
import os

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from src.settings import get_setting

_MODEL_PACK_NAME = "buffalo_l"
_SWAP_MODEL_FILENAME = "inswapper_128.onnx"

_analyzer = None
_swapper = None


class LicenseNotAcceptedError(Exception):
    """Raised when a swap is attempted before the InsightFace model
    license (covering both buffalo_l and inswapper_128.onnx) has been
    explicitly accepted via the face_swap_license_accepted setting."""


class NoFaceDetectedError(Exception):
    """Raised when face detection finds no usable face in an input image."""


def _model_root() -> str:
    from src.constants import DATA_DIR
    return os.path.join(DATA_DIR, "face_swap_models")


def _ensure_models_available():
    """Gate: the InsightFace license must be explicitly accepted before
    InsightFace's own downloader is allowed to run. Called from both
    _get_analyzer() and _get_swapper() so neither singleton can be
    constructed (and neither model implicitly downloaded) without passing
    this check first."""
    if not get_setting("face_swap_license_accepted", False):
        raise LicenseNotAcceptedError(
            "Face-swap requires accepting InsightFace's model license first "
            "-- go to Settings -> AI Features -> Face Swap to review and "
            "accept it."
        )


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _ensure_models_available()
        from insightface.app import FaceAnalysis
        _analyzer = FaceAnalysis(name=_MODEL_PACK_NAME, root=_model_root())
        _analyzer.prepare(ctx_id=0, det_size=(640, 640))
    return _analyzer


def _get_swapper():
    global _swapper
    if _swapper is None:
        _ensure_models_available()
        from insightface.model_zoo import get_model
        _swapper = get_model(
            os.path.join(_model_root(), "models", _SWAP_MODEL_FILENAME),
            download=True,
        )
    return _swapper


def _provenance_metadata() -> PngInfo:
    info = PngInfo()
    info.add_text("assist:ai-edited", "face-swap")
    return info


def swap_face(source_face_bytes: bytes, target_image_bytes: bytes, *,
              analyzer=None, swapper=None) -> bytes:
    """Swap the face from source_face_bytes into target_image_bytes.
    Returns PNG bytes with embedded provenance metadata. Raises
    LicenseNotAcceptedError, NoFaceDetectedError, or lets underlying
    inference errors propagate -- callers apply the never-raises discipline
    at their own boundary, matching src/bg_removal.py's convention."""
    analyzer = analyzer if analyzer is not None else _get_analyzer()
    swapper = swapper if swapper is not None else _get_swapper()

    source_img = np.array(Image.open(io.BytesIO(source_face_bytes)).convert("RGB"))[:, :, ::-1]
    target_img = np.array(Image.open(io.BytesIO(target_image_bytes)).convert("RGB"))[:, :, ::-1]

    source_faces = analyzer.get(source_img)
    if not source_faces:
        raise NoFaceDetectedError("No face detected in the source image")
    target_faces = analyzer.get(target_img)
    if not target_faces:
        raise NoFaceDetectedError("No face detected in the target image")

    result = swapper.get(target_img, target_faces[0], source_faces[0], paste_back=True)

    out = Image.fromarray(result[:, :, ::-1])
    buf = io.BytesIO()
    out.save(buf, format="PNG", pnginfo=_provenance_metadata())
    return buf.getvalue()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_face_swap.py -v --import-mode=importlib`
Expected: PASS (all 5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/face_swap.py tests/test_face_swap.py
git commit -m "feat(face-swap): core InsightFace swap module with gated model licensing

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `face_swap` builtin agent tool + full registration

**Files:**
- Modify: `src/agent_tools/image_tools.py` (add `face_swap_tool` + `FaceSwapTool`, reusing existing shared helpers)
- Modify: `src/agent_tools/__init__.py` (import + `TOOL_HANDLERS` + `TOOL_TAGS`)
- Modify: `src/tool_schemas.py` (`FUNCTION_TOOL_SCHEMAS`)
- Modify: `src/agent_loop.py` (`TOOL_SECTIONS` + `_DOMAIN_TOOL_MAP["desktop"]`)
- Modify: `src/tool_index.py` (`BUILTIN_TOOL_DESCRIPTIONS`)
- Modify: `src/tool_execution.py` (owner-threading dispatch branch)
- Modify: `routes/chat_routes.py` (`can_generate_images` privilege gate)
- Modify: `src/tool_security.py` (`_PLAN_MODE_KNOWN_MUTATORS`)
- Test: `tests/test_face_swap_tool.py`
- Test: `tests/test_face_swap_registration.py`

**Interfaces:**
- Consumes: `src.face_swap.swap_face(source_face_bytes, target_image_bytes, *, analyzer=None, swapper=None)` (Task 2); `src.face_swap.LicenseNotAcceptedError`, `NoFaceDetectedError`; `src.upload_handler.UploadHandler.resolve_upload` (existing); `_resolve_attachment_bytes(tool_name, attachment_id, owner, upload_resolver)` and `_image_result(output_message, result_bytes, saved)` (existing shared helpers in `src/agent_tools/image_tools.py`, extracted during sub-project 2's whole-branch-review fix wave — this is the third tool through them, exactly the case they were generalized for).
- Produces: `face_swap_tool(content, ctx, *, swapper=None, upload_resolver=None, gallery_saver=None)` in `src/agent_tools/image_tools.py`, and `FaceSwapTool` (a thin `execute(self, content, ctx)` wrapper class), registered as builtin tool `"face_swap"`.

**On the tool's `swapper` injection parameter naming**: to avoid confusion with `src.face_swap.swap_face`'s own `swapper=` kwarg (which is InsightFace's internal swap-model object), this tool's injectable parameter is named `swapper` too but is a plain `(source_face_bytes, target_image_bytes) -> bytes` callable (defaulting to `src.face_swap.swap_face` itself) — mirroring exactly how `remove_background_tool`'s `remover=` and `edit_image_prompt_tool`'s `editor=` each wrap their respective core module's top-level function as a single injectable callable, not the lower-level model objects inside it.

**On the never-raises boundary**: this tool must catch `LicenseNotAcceptedError` and `NoFaceDetectedError` specifically (both propagate a clear, user-facing message already, so `str(e)` is the right error text) alongside the generic `Exception` catch every other tool already uses for its model-call step — Python's `except Exception` already covers both (they're both plain `Exception` subclasses), so no special-casing is needed in the `try/except` structure itself, only in verifying the resulting error text reads clearly (which it does, since both exception messages are already written as complete, actionable sentences).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_face_swap_tool.py`:

```python
"""face_swap_tool resolves TWO chat attachments (source face, target
image), runs them through src.face_swap, and returns a short served URL
via the established image_url convention. Never raises into the agent
loop, matching remove_background_tool/edit_image_prompt_tool's established
pattern. See
docs/superpowers/specs/2026-08-13-image-editing-face-swap-design.md.
"""
import asyncio
import json

from src.agent_tools.image_tools import FaceSwapTool, face_swap_tool


def _fake_upload_resolver(paths_by_id):
    def resolver(upload_id, owner=None):
        path = paths_by_id.get(upload_id)
        if path is None:
            return None
        return {"id": upload_id, "path": path, "name": "fake.png", "mime": "image/png"}
    return resolver


def _fake_swapper(output=b"fake-swapped-png"):
    def swapper(source_face_bytes, target_image_bytes):
        return output
    return swapper


def _fake_gallery_saver(image_id="gallery-img-1", filename="abc123def456.png"):
    calls = []

    def saver(image_bytes, owner):
        calls.append((image_bytes, owner))
        return {"id": image_id, "filename": filename}

    saver.calls = calls
    return saver


def test_missing_source_face_id_returns_error():
    content = json.dumps({"target_image_id": "up-2"})
    result = asyncio.run(face_swap_tool(content, {"owner": "alice"}))
    assert "error" in result


def test_missing_target_image_id_returns_error():
    content = json.dumps({"source_face_id": "up-1"})
    result = asyncio.run(face_swap_tool(content, {"owner": "alice"}))
    assert "error" in result


def test_invalid_json_returns_error():
    result = asyncio.run(face_swap_tool("not json", {"owner": "alice"}))
    assert "error" in result


def test_unresolvable_source_returns_error():
    content = json.dumps({"source_face_id": "missing", "target_image_id": "up-2"})
    result = asyncio.run(face_swap_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver({"up-2": "/tmp/target.png"}),
        swapper=_fake_swapper(),
    ))
    assert "error" in result


def test_unresolvable_target_returns_error():
    content = json.dumps({"source_face_id": "up-1", "target_image_id": "missing"})
    result = asyncio.run(face_swap_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver({"up-1": "/tmp/source.png"}),
        swapper=_fake_swapper(),
    ))
    assert "error" in result


def test_successful_swap_returns_short_url_and_saves_to_gallery(tmp_path):
    source_file = tmp_path / "source.png"
    source_file.write_bytes(b"source-bytes")
    target_file = tmp_path / "target.png"
    target_file.write_bytes(b"target-bytes")
    content = json.dumps({"source_face_id": "up-1", "target_image_id": "up-2"})
    gallery_saver = _fake_gallery_saver(image_id="gallery-img-1", filename="abc123def456.png")

    result = asyncio.run(face_swap_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver({"up-1": str(source_file), "up-2": str(target_file)}),
        swapper=_fake_swapper(output=b"swapped-bytes"),
        gallery_saver=gallery_saver,
    ))

    assert result["image_url"] == "/api/generated-image/abc123def456.png"
    assert result["gallery_image_id"] == "gallery-img-1"
    assert gallery_saver.calls == [(b"swapped-bytes", "alice")]


def test_license_not_accepted_returns_clear_error(tmp_path, monkeypatch):
    import src.agent_tools.image_tools as image_tools
    from src.face_swap import LicenseNotAcceptedError

    def failing_swapper(source_face_bytes, target_image_bytes):
        raise LicenseNotAcceptedError("Face-swap requires accepting InsightFace's model license first")

    source_file = tmp_path / "source.png"
    source_file.write_bytes(b"source-bytes")
    target_file = tmp_path / "target.png"
    target_file.write_bytes(b"target-bytes")
    content = json.dumps({"source_face_id": "up-1", "target_image_id": "up-2"})

    result = asyncio.run(face_swap_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver({"up-1": str(source_file), "up-2": str(target_file)}),
        swapper=failing_swapper,
        gallery_saver=_fake_gallery_saver(),
    ))

    assert "error" in result
    assert "license" in result["error"].lower()


def test_no_face_detected_returns_clear_error(tmp_path):
    from src.face_swap import NoFaceDetectedError

    def failing_swapper(source_face_bytes, target_image_bytes):
        raise NoFaceDetectedError("No face detected in the source image")

    source_file = tmp_path / "source.png"
    source_file.write_bytes(b"source-bytes")
    target_file = tmp_path / "target.png"
    target_file.write_bytes(b"target-bytes")
    content = json.dumps({"source_face_id": "up-1", "target_image_id": "up-2"})

    result = asyncio.run(face_swap_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver({"up-1": str(source_file), "up-2": str(target_file)}),
        swapper=failing_swapper,
        gallery_saver=_fake_gallery_saver(),
    ))

    assert "error" in result
    assert "face" in result["error"].lower()


def test_gallery_save_failure_falls_back_to_inline_data_uri(tmp_path):
    import base64

    source_file = tmp_path / "source.png"
    source_file.write_bytes(b"source-bytes")
    target_file = tmp_path / "target.png"
    target_file.write_bytes(b"target-bytes")
    content = json.dumps({"source_face_id": "up-1", "target_image_id": "up-2"})

    def failing_saver(image_bytes, owner):
        raise RuntimeError("db unavailable")

    result = asyncio.run(face_swap_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver({"up-1": str(source_file), "up-2": str(target_file)}),
        swapper=_fake_swapper(output=b"swapped-bytes"),
        gallery_saver=failing_saver,
    ))

    assert result["image_url"].startswith("data:image/png;base64,")
    decoded = base64.b64decode(result["image_url"].split(",", 1)[1])
    assert decoded == b"swapped-bytes"
    assert "gallery_image_id" not in result


def test_tool_class_delegates_to_module_function():
    tool = FaceSwapTool()
    content = json.dumps({"target_image_id": "up-2"})
    result = asyncio.run(tool.execute(content, {"owner": "alice"}))
    assert "error" in result
```

Create `tests/test_face_swap_registration.py`:

```python
"""face_swap must be registered everywhere a builtin tool needs to be,
applying every lesson sub-project 1's whole-branch review found (dispatcher
owner-threading, the can_generate_images privilege gate, the plan-mode
backstop) from the start. Mirrors
tests/test_edit_image_prompt_registration.py's structure. See
docs/superpowers/specs/2026-08-13-image-editing-face-swap-design.md."""
import asyncio
from pathlib import Path

_CHAT_ROUTES = Path(__file__).resolve().parent.parent / "routes" / "chat_routes.py"


def test_face_swap_registered_everywhere():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    from src.agent_loop import TOOL_SECTIONS, _DOMAIN_TOOL_MAP
    from src.tool_index import BUILTIN_TOOL_DESCRIPTIONS

    assert "face_swap" in TOOL_HANDLERS
    assert "face_swap" in TOOL_TAGS
    names = [(s.get("function") or {}).get("name") for s in FUNCTION_TOOL_SCHEMAS]
    assert "face_swap" in names
    assert "face_swap" in TOOL_SECTIONS
    assert "face_swap" in _DOMAIN_TOOL_MAP["desktop"]
    assert "face_swap" in BUILTIN_TOOL_DESCRIPTIONS


def test_face_swap_matches_generate_image_admin_gating():
    from src.tool_security import NON_ADMIN_BLOCKED_TOOLS
    generate_image_blocked = "generate_image" in NON_ADMIN_BLOCKED_TOOLS
    face_swap_blocked = "face_swap" in NON_ADMIN_BLOCKED_TOOLS
    assert face_swap_blocked == generate_image_blocked, (
        "face_swap's NON_ADMIN_BLOCKED_TOOLS membership must mirror "
        "generate_image's, not default to admin-only"
    )


def test_face_swap_blocked_when_can_generate_images_disabled():
    source = _CHAT_ROUTES.read_text(encoding="utf-8")
    assert 'if not _privs.get("can_generate_images", True):' in source
    idx = source.index('if not _privs.get("can_generate_images", True):')
    following = source[idx: idx + 300]
    assert "face_swap" in following, (
        "face_swap must be added to disabled_tools in the same "
        "can_generate_images privilege branch as generate_image/remove_background/edit_image_prompt"
    )


def test_dispatcher_threads_owner_and_session_into_tool_ctx(monkeypatch):
    """The REAL dispatcher (execute_tool_block, not a mock of it) must
    thread owner= into face_swap's ctx -- otherwise the tool falls into the
    generic dynamic_handlers catch-all, which never threads owner, and
    resolve_upload denies every real (owned) attachment for BOTH images
    this tool resolves."""
    import src.agent_tools as agent_tools
    from src.agent_tools import ToolBlock
    from src.tool_execution import execute_tool_block

    seen = {}

    async def spy(content, ctx):
        seen["content"] = content
        seen["ctx"] = ctx
        return {"output": "Face swapped.", "exit_code": 0}

    monkeypatch.setitem(agent_tools.TOOL_HANDLERS, "face_swap", spy)

    desc, result = asyncio.run(execute_tool_block(
        ToolBlock("face_swap", '{"source_face_id": "up-1", "target_image_id": "up-2"}'),
        session_id="sess-1",
        owner="alice",
    ))

    assert result.get("exit_code") == 0
    assert seen["ctx"].get("owner") == "alice", (
        "face_swap's ctx lost the owner -- resolve_upload will deny every owned attachment"
    )
    assert seen["ctx"].get("session_id") == "sess-1"


def test_face_swap_in_plan_mode_known_mutators():
    """face_swap writes a PNG to disk and inserts a Gallery DB row -- the
    same class of mutator as generate_image/edit_image/remove_background/
    edit_image_prompt, all members of _PLAN_MODE_KNOWN_MUTATORS."""
    import src.tool_security as ts
    assert "face_swap" in ts._PLAN_MODE_KNOWN_MUTATORS
    disabled = ts.plan_mode_disabled_tools()
    assert "face_swap" in disabled
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_face_swap_tool.py tests/test_face_swap_registration.py -v --import-mode=importlib`
Expected: FAIL — `ImportError: cannot import name 'face_swap_tool' from 'src.agent_tools.image_tools'`.

- [ ] **Step 3: Append to `src/agent_tools/image_tools.py`**

At the end of the file (after `EditImagePromptTool`'s class definition), append:

```python
async def face_swap_tool(content, ctx, *, swapper=None, upload_resolver=None, gallery_saver=None):
    ctx = ctx or {}
    owner = ctx.get("owner")

    try:
        args = json.loads(content) if content and content.strip() else {}
        if not isinstance(args, dict):
            return {"error": "face_swap: arguments must be a JSON object"}
    except (ValueError, TypeError):
        return {"error": "face_swap: arguments must be valid JSON"}

    source_face_id = args.get("source_face_id")
    if not isinstance(source_face_id, str) or not source_face_id.strip():
        return {"error": "face_swap: a 'source_face_id' is required"}

    target_image_id = args.get("target_image_id")
    if not isinstance(target_image_id, str) or not target_image_id.strip():
        return {"error": "face_swap: a 'target_image_id' is required"}

    if upload_resolver is None:
        from src.constants import DATA_DIR, UPLOAD_DIR
        from src.upload_handler import UploadHandler
        upload_resolver = UploadHandler(DATA_DIR, UPLOAD_DIR).resolve_upload

    source_bytes, err = await _resolve_attachment_bytes("face_swap", source_face_id, owner, upload_resolver)
    if err:
        return err
    target_bytes, err = await _resolve_attachment_bytes("face_swap", target_image_id, owner, upload_resolver)
    if err:
        return err

    if swapper is None:
        from src.face_swap import swap_face as swapper

    try:
        result_bytes = await asyncio.to_thread(swapper, source_bytes, target_bytes)
    except Exception as e:
        return {"error": f"face_swap: {e}"}

    saver = gallery_saver or _default_gallery_saver
    try:
        saved = saver(result_bytes, owner)
    except Exception:
        logger.warning("face_swap: failed to save result to Gallery", exc_info=True)
        saved = None

    return _image_result("Face swapped.", result_bytes, saved)


class FaceSwapTool:
    async def execute(self, content, ctx):
        return await face_swap_tool(content, ctx)
```

- [ ] **Step 4: Register in `src/agent_tools/__init__.py`**

Find this line:

```python
from .image_tools import RemoveBackgroundTool, EditImagePromptTool
```

Replace it with:

```python
from .image_tools import RemoveBackgroundTool, EditImagePromptTool, FaceSwapTool
```

Find this line:

```python
    "edit_image_prompt": EditImagePromptTool().execute,
```

Immediately after it, insert:

```python
    "face_swap": FaceSwapTool().execute,
```

Find the `TOOL_TAGS` set's `"edit_image_prompt",` entry and add `"face_swap",` immediately after it, matching the set's existing formatting.

- [ ] **Step 5: Add the schema in `src/tool_schemas.py`**

Find the `edit_image_prompt` entry in `FUNCTION_TOOL_SCHEMAS` (its closing `},` is immediately followed by the `ui_control` entry) and insert this new entry immediately after it:

```python
    {
        "type": "function",
        "function": {
            "name": "face_swap",
            "description": "Swap a face from one uploaded image (the source face) into another uploaded image (the target). Returns the target image with the face swapped in.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_face_id": {"type": "string", "description": "The id of the uploaded image containing the source face"},
                    "target_image_id": {"type": "string", "description": "The id of the uploaded image to swap the face into"},
                },
                "required": ["source_face_id", "target_image_id"],
            },
        },
    },
```

- [ ] **Step 6: Add to `TOOL_SECTIONS` and `_DOMAIN_TOOL_MAP` in `src/agent_loop.py`**

Find the `TOOL_SECTIONS` dict's `"edit_image_prompt"` entry and insert this new entry immediately after it (after its closing `""",`):

```python
    "face_swap": """```face_swap
{"source_face_id": "<id of the uploaded source-face image>", "target_image_id": "<id of the uploaded target image>"}
```
Swap the face from the source image into the target image. Requires the user to have accepted the face-swap model's license in Settings first. Returns the result shown inline in your response.""",
```

Find this exact block:

```python
    "desktop": {"launch_app", "find_files", "list_windows", "control_window", "capture_screen",
                "webcam_look", "diagnose_equipment", "remove_background", "edit_image_prompt",
                "ingest_equipment_manual", "search_equipment_manual",
                "list_ui_elements", "click_element", "set_element_text", "mouse", "keyboard"},
```

Replace it with:

```python
    "desktop": {"launch_app", "find_files", "list_windows", "control_window", "capture_screen",
                "webcam_look", "diagnose_equipment", "remove_background", "edit_image_prompt", "face_swap",
                "ingest_equipment_manual", "search_equipment_manual",
                "list_ui_elements", "click_element", "set_element_text", "mouse", "keyboard"},
```

- [ ] **Step 7: Add to `BUILTIN_TOOL_DESCRIPTIONS` in `src/tool_index.py`**

Find the `"edit_image_prompt"` entry and insert this new entry immediately after it:

```python
    "face_swap": "Swap a face from one uploaded image into another uploaded image. Requires accepting the model's license in Settings first.",
```

- [ ] **Step 8: Add the owner-threading dispatch branch in `src/tool_execution.py`**

Find this exact block:

```python
    elif tool == "edit_image_prompt":
        # Registry-dispatched (agent_tools.image_tools); owner threaded for the
        # exact same reason as remove_background just above — the tool resolves
        # the caller's OWN chat attachment via upload_handler.resolve_upload(),
        # which denies any owned upload record when called with owner=None.
        desc = f"edit_image_prompt: {content.split(chr(10))[0][:80]}"
        result = await _direct_fallback(tool, content, session_id=session_id, owner=owner) \
            or {"error": "edit_image_prompt: execution failed", "exit_code": 1}
```

Immediately after it, insert:

```python
    elif tool == "face_swap":
        # Registry-dispatched (agent_tools.image_tools); owner threaded for the
        # exact same reason as remove_background/edit_image_prompt just above —
        # the tool resolves TWO of the caller's OWN chat attachments via
        # upload_handler.resolve_upload(), which denies any owned upload record
        # when called with owner=None.
        desc = f"face_swap: {content.split(chr(10))[0][:80]}"
        result = await _direct_fallback(tool, content, session_id=session_id, owner=owner) \
            or {"error": "face_swap: execution failed", "exit_code": 1}
```

- [ ] **Step 9: Add the privilege gate in `routes/chat_routes.py`**

Find this exact line:

```python
                disabled_tools.update({"generate_image", "remove_background", "edit_image_prompt"})
```

Replace it with:

```python
                disabled_tools.update({"generate_image", "remove_background", "edit_image_prompt", "face_swap"})
```

- [ ] **Step 10: Add to `_PLAN_MODE_KNOWN_MUTATORS` in `src/tool_security.py`**

Find this exact line:

```python
    "remove_background", "edit_image_prompt",
```

Replace it with:

```python
    "remove_background", "edit_image_prompt", "face_swap",
```

Do NOT add `face_swap` to `NON_ADMIN_BLOCKED_TOOLS` — `generate_image` is confirmed absent from that set, and `face_swap` mirrors it (enforced by `test_face_swap_matches_generate_image_admin_gating`, Step 1).

- [ ] **Step 11: Run tests to verify they pass**

Run: `pytest tests/test_face_swap_tool.py tests/test_face_swap_registration.py -v --import-mode=importlib`
Expected: PASS (10 + 5 = 15 tests).

Run: `pytest tests/test_edit_image_prompt_tool.py tests/test_edit_image_prompt_registration.py tests/test_remove_background_tool.py tests/test_remove_background_registration.py -v --import-mode=importlib` to confirm no regression to the two prior sub-projects' tools (this task does not modify `_default_gallery_saver`, `_resolve_attachment_bytes`, or `_image_result` — only calls them — so this should be a pure regression check with no expected behavior change).

Run: `pytest tests/test_desktop_registration.py -v --import-mode=importlib` to confirm no regression to the pre-existing desktop-tool parity test.

Run: `python -c "import app"` to confirm the app still imports cleanly with the new tool wired in.

- [ ] **Step 12: Commit**

```bash
git add src/agent_tools/image_tools.py src/agent_tools/__init__.py src/tool_schemas.py src/agent_loop.py src/tool_index.py src/tool_execution.py routes/chat_routes.py src/tool_security.py tests/test_face_swap_tool.py tests/test_face_swap_registration.py
git commit -m "feat(face-swap): face_swap builtin agent tool

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: License-acceptance setting + Settings UI

**Files:**
- Modify: `src/settings.py` (add `face_swap_license_accepted` to `DEFAULT_SETTINGS`)
- Modify: `static/index.html` (new admin-card toggle, mirroring the `image_gen_enabled` card exactly)
- Modify: `static/js/settings.js` (load/save wiring for the new toggle)
- Test: `tests/test_face_swap_license_setting.py`

**Interfaces:**
- Consumes: `src.settings.DEFAULT_SETTINGS`, `get_setting`, `load_settings`, `save_settings` (existing); the existing generic `POST /api/auth/settings` route (`routes/auth_routes.py`, admin-only, already updates any key present in `DEFAULT_SETTINGS` — confirmed by direct read, no new backend route needed for the accept action itself).
- Produces: `face_swap_license_accepted: False` as a new `DEFAULT_SETTINGS` key, readable via `get_setting("face_swap_license_accepted", False)` (this is exactly what `src.face_swap._ensure_models_available()`, Task 2, already reads).

**On which existing pattern to mirror**: `shell_exec_enabled` (in `static/js/shellExec.js`) looked like the obvious analog at first glance, but is the WRONG one to copy — it's a per-session sidebar quick-toggle that resets to off on every app restart (`shellExec.js`'s own comment: "Defaults off and is reset off server-side on every restart"), matching capabilities like screen/camera/input access. `face_swap_license_accepted` is the opposite: a one-time acceptance that must persist across restarts. The correct analog, confirmed by direct research, is `image_gen_enabled`'s admin Settings-page card — a genuine persisted `DEFAULT_SETTINGS` boolean with its own toggle in the real Settings page, not a session-reset sidebar control.

**Verify this task does NOT wire the new key into any of the `reset_screen_access`/`reset_input_control`/`reset_camera_access`/`reset_shell_exec`-style startup reset functions in `src/settings.py`** — those exist specifically for per-session re-consent of active capabilities; a one-time license acknowledgment should persist across restarts, unlike those. Confirm by reading `src/settings.py`'s reset functions before writing this task's code, and do not add a `reset_face_swap_license` function or add this key to any existing reset function.

- [ ] **Step 1: Write the failing test**

Create `tests/test_face_swap_license_setting.py`:

```python
"""face_swap_license_accepted is a plain, persistent DEFAULT_SETTINGS key --
unlike screen/camera/input/shell access, it must NOT be wired into any
startup reset function, since a one-time license acknowledgment should
persist across restarts. See
docs/superpowers/specs/2026-08-13-image-editing-face-swap-design.md."""
import inspect

from src.settings import DEFAULT_SETTINGS


def test_face_swap_license_accepted_defaults_false():
    assert DEFAULT_SETTINGS["face_swap_license_accepted"] is False


def test_face_swap_license_accepted_not_reset_at_startup():
    import src.settings as settings_module
    reset_fns = [
        obj for name, obj in vars(settings_module).items()
        if name.startswith("reset_") and inspect.isfunction(obj)
    ]
    assert reset_fns, "expected at least one reset_* function to exist as a baseline"
    for fn in reset_fns:
        src = inspect.getsource(fn)
        assert "face_swap_license_accepted" not in src, (
            f"{fn.__name__} must not touch face_swap_license_accepted -- "
            "it's a persistent one-time acceptance, not a per-session capability"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_face_swap_license_setting.py -v --import-mode=importlib`
Expected: FAIL — `KeyError: 'face_swap_license_accepted'`.

- [ ] **Step 3: Add the setting**

In `src/settings.py`, find the `DEFAULT_SETTINGS` dict's `"shell_exec_enabled": False,` line and insert this immediately after it:

```python
    # One-time acknowledgment that the user has read and accepted
    # InsightFace's model license (covering both the buffalo_l detection
    # pack and the inswapper_128.onnx swap model -- both are non-commercial/
    # research-only). Unlike screen/camera/input/shell access, this is a
    # persistent acceptance, not a per-session capability grant -- it must
    # NOT be wired into any reset_* startup function. See
    # src/face_swap.py's _ensure_models_available().
    "face_swap_license_accepted": False,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_face_swap_license_setting.py -v --import-mode=importlib`
Expected: PASS (both tests).

- [ ] **Step 5: Add the Settings UI toggle**

In `static/index.html`, find the `image_gen_enabled` admin-card (search for `set-imgEnabledToggle`):

```html
          <div class="admin-card">
            <h2 style="display:flex;align-items:center;gap:6px;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:1px;opacity:0.6;flex-shrink:0"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>Image Generation<span style="flex:1"></span><label class="admin-switch"><input type="checkbox" id="set-imgEnabledToggle"><span class="admin-slider"></span></label></h2>
            <div class="admin-toggle-sub" style="margin-bottom:8px">Configure which model to use for image generation.</div>
```

Immediately after that card's closing `</div>`, insert a new card following the exact same structure, but — unlike a plain on/off capability toggle — its body must show InsightFace's actual license terms before the checkbox represents an informed acceptance, not a bare feature flag:

```html
          <div class="admin-card">
            <h2 style="display:flex;align-items:center;gap:6px;">Face Swap Model License<span style="flex:1"></span><label class="admin-switch"><input type="checkbox" id="set-faceSwapLicenseToggle"><span class="admin-slider"></span></label></h2>
            <div class="admin-toggle-sub" style="margin-bottom:8px">
              Face-swap uses InsightFace's face-detection (buffalo_l) and face-swap
              (InSwapper) models, both licensed by InsightFace for non-commercial
              research use only — see
              <a href="https://github.com/deepinsight/insightface" target="_blank" rel="noopener">github.com/deepinsight/insightface</a>
              for their current license terms. Checking this box downloads both models
              (a few hundred MB total) for that purpose. Commercial use requires a
              separate license directly from InsightFace.
            </div>
          </div>
```

- [ ] **Step 6: Wire the toggle in `static/js/settings.js`**

Find where `set-imgEnabledToggle` is read and saved (search for `set-imgEnabledToggle`) — the load side reads `el('set-imgEnabledToggle')` and sets `.checked` from the fetched settings object; the save side reads `.checked` and includes it in the `JSON.stringify({...})` body posted to `/api/auth/settings`. Add the analogous load/save wiring for `set-faceSwapLicenseToggle` → `face_swap_license_accepted`, in whichever existing load/save function block is most appropriate (a new small dedicated block is fine if the `image_gen_enabled` block's surrounding function is scoped specifically to image-generation settings and doesn't naturally extend to an unrelated key — check the function's scope before deciding whether to extend it or add a small sibling block).

- [ ] **Step 7: Manually verify the Settings UI**

Start the app, open Settings, confirm the new Face Swap license card renders with the license text visible and the checkbox unchecked by default, check it, save, reload the page, and confirm it's still checked (persisted). This is a manual, owed verification step — no automated frontend test framework runs a real browser in this repo's test suite for Settings toggles specifically (confirm this assumption against whatever existing Settings-toggle tests already exist in `tests/` before treating this as manual-only; if an automated pattern already exists for other toggles, mirror it instead).

- [ ] **Step 8: Commit**

```bash
git add src/settings.py static/index.html static/js/settings.js tests/test_face_swap_license_setting.py
git commit -m "feat(face-swap): license-acceptance setting + Settings UI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Gallery editor addition

**Files:**
- Modify: `routes/gallery/gallery_routes.py` (new route)
- Create: `static/js/editor/ai-face-swap.js` (new frontend wiring — no existing Gallery editor action takes a secondary uploaded file, confirmed by direct research; this needs new frontend surface, not a drop-in reuse of `applyImageTool`)
- Modify: whichever file registers Gallery editor toolbar buttons (confirm exact file by reading how `ai-rembg.js`'s button is registered — search for where `ai-rembg.js` or `ai-tools-misc.js` gets included/wired into the editor's toolbar)
- Test: `tests/test_gallery_face_swap_route.py`

**Interfaces:**
- Consumes: `src.face_swap.swap_face(source_face_bytes, target_image_bytes, *, analyzer=None, swapper=None)` (Task 2).
- Produces: a new route, `POST /api/image/face-swap`, taking the currently-open editor canvas (flattened, as every other Gallery AI action already does) as the *target* image and a newly-uploaded file as the *source* face — following Style Transfer's established `FormData` pattern (the one existing Gallery AI action that already sends a file via `FormData` rather than a JSON body), extended with a second field for the source face.

**Context — read before writing this task**: direct research already confirmed every existing Gallery editor AI action (`ai-rembg.js`, `ai-tool-runner.js`'s shared `applyImageTool`, `ai-tools-misc.js`'s Style Transfer, `ai-inpaint.js`) operates on exactly one image (the flattened editor canvas) with no secondary file input. `wire-import.js` has a file input, but for an unrelated purpose (importing an external image as a new layer, not calling an AI tool). This task's frontend work is therefore genuinely new, not a call site added to an existing shared function — confirm this is still accurate by re-reading `static/js/editor/ai-tool-runner.js` and `static/js/editor/ai-tools-misc.js`'s Style Transfer wiring directly before writing code, since some time has passed since that research and the exact line numbers/structure may have shifted.

**Style Transfer's route** (`routes/gallery/gallery_routes.py`, `POST /api/gallery/style-transfer`) is the exact backend pattern to mirror, confirmed by direct read — **both fields are real file uploads via `request.form()`, not base64-JSON** (this corrects an earlier draft of this plan, which wrongly assumed the target image would be base64 like `remove-bg`'s route):

```python
    @router.post("/api/gallery/style-transfer")
    async def gallery_style_transfer(request: Request):
        """Style transfer using img2img with the diffusion server."""
        import base64, httpx

        user = require_privilege(request, "can_generate_images")
        form = await request.form()
        file = form.get("image")
        prompt = form.get("prompt", "")
        strength = float(form.get("strength", "0.55"))
        if not file: raise HTTPException(400, "No image")

        image_bytes = await read_upload_limited(file, GALLERY_TRANSFORM_UPLOAD_MAX_BYTES, "Image upload")
```

`read_upload_limited(upload, limit, label="Upload") -> bytes` (`src/upload_limits.py:64`, already imported in `gallery_routes.py`) calls `await upload.read(limit + 1)` internally — the uploaded-file object needs an async `.read(n)` method accepting a size argument, not a bare no-arg `.read()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gallery_face_swap_route.py`:

```python
"""POST /api/image/face-swap accepts BOTH the flattened editor canvas
(target) and an uploaded source-face image as real file uploads via
request.form(), mirroring Style Transfer's established FormData pattern
(routes/gallery/gallery_routes.py's /api/gallery/style-transfer) exactly --
NOT the base64-JSON pattern remove-bg's route uses, since this route needs
two real file uploads, not one base64 field. Calls src.face_swap and
returns the result. See
docs/superpowers/specs/2026-08-13-image-editing-face-swap-design.md.
"""
import asyncio
import base64
import io

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


class _FakeUploadFile:
    """Matches read_upload_limited's contract: async .read(n) accepting a
    size argument (it calls upload.read(limit + 1)), not a bare .read()."""
    def __init__(self, data: bytes):
        self._data = data

    async def read(self, n=None):
        return self._data


class _FakeFormData(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


class _FakeRequest:
    def __init__(self, form_data):
        self._form_data = form_data

    async def form(self):
        return self._form_data


def test_face_swap_route_calls_face_swap_module(monkeypatch):
    monkeypatch.setattr(gallery_routes_module, "require_privilege", lambda request, priv: "alice")

    captured = {}

    def fake_swap_face(source_face_bytes, target_image_bytes, **kwargs):
        captured["source"] = source_face_bytes
        captured["target"] = target_image_bytes
        out = Image.open(io.BytesIO(target_image_bytes)).convert("RGBA")
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()

    monkeypatch.setattr(gallery_routes_module, "swap_face", fake_swap_face)

    router = gallery_routes_module.setup_gallery_routes()
    handler = _route(router, "/api/image/face-swap", "POST")

    form = _FakeFormData({
        "image": _FakeUploadFile(_png_bytes()),
        "source_face": _FakeUploadFile(_png_bytes(color=(50, 60, 70, 255))),
    })
    out = asyncio.run(handler(_FakeRequest(form)))

    assert captured["source"] == _png_bytes(color=(50, 60, 70, 255))
    assert captured["target"] == _png_bytes()
    assert "image" in out
    result_img = Image.open(io.BytesIO(base64.b64decode(out["image"])))
    assert result_img.mode == "RGBA"


def test_face_swap_route_returns_error_on_model_failure(monkeypatch):
    monkeypatch.setattr(gallery_routes_module, "require_privilege", lambda request, priv: "alice")

    def failing_swap_face(source_face_bytes, target_image_bytes, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(gallery_routes_module, "swap_face", failing_swap_face)

    router = gallery_routes_module.setup_gallery_routes()
    handler = _route(router, "/api/image/face-swap", "POST")

    form = _FakeFormData({
        "image": _FakeUploadFile(_png_bytes()),
        "source_face": _FakeUploadFile(_png_bytes()),
    })
    out = asyncio.run(handler(_FakeRequest(form)))

    assert "error" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gallery_face_swap_route.py -v --import-mode=importlib`
Expected: FAIL — `AssertionError` (no route registered at `/api/image/face-swap`).

- [ ] **Step 3: Add the route**

Add a new route to `routes/gallery/gallery_routes.py`, mirroring Style Transfer's exact `request.form()` structure shown above:

```python
    # ---- POST /api/image/face-swap ----
    @router.post("/api/image/face-swap")
    async def gallery_face_swap(request: Request):
        """Swap a face from an uploaded source image into the currently-open
        editor canvas (target). Both are real file uploads, matching
        style-transfer's FormData pattern -- not remove-bg's base64-JSON
        pattern, since this route needs two files, not one base64 field."""
        import base64

        user = require_privilege(request, "can_generate_images")
        form = await request.form()
        target_file = form.get("image")
        source_file = form.get("source_face")
        if not target_file or not source_file:
            raise HTTPException(400, "Both image and source_face are required")

        target_bytes = await read_upload_limited(target_file, GALLERY_TRANSFORM_UPLOAD_MAX_BYTES, "Target image")
        source_bytes = await read_upload_limited(source_file, GALLERY_TRANSFORM_UPLOAD_MAX_BYTES, "Source face image")

        try:
            result_bytes = await asyncio.to_thread(run_face_swap_model, source_bytes, target_bytes)
        except Exception as e:
            logger.warning("Face swap failed", exc_info=True)
            return {"error": "Face swap failed"}

        return {"image": base64.b64encode(result_bytes).decode()}
```

Check whether `asyncio` is already imported at module scope in `gallery_routes.py` (it is used by the `remove-bg` route's fix from sub-project 1's whole-branch review — confirm this import survived, don't add a duplicate).

Add the import near the top of the file, aliased to avoid any name collision the way sub-project 1's Task 2 aliased `remove_background` (check whether `gallery_routes.py` has a local variable, function, or import already named `swap_face` before deciding whether the alias is actually necessary — don't alias defensively if nothing collides):

```python
from src.face_swap import swap_face as run_face_swap_model
```

This mirrors the established `test_no_raw_exception_string_in_client_responses` hardening requirement (a static "Face swap failed" client message, real exception only in the server log) and the Important-1 lesson from sub-project 2's whole-branch review (`asyncio.to_thread` around the blocking model call, applied from day one here rather than discovered in review).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gallery_face_swap_route.py -v --import-mode=importlib`
Expected: PASS (both tests).

Run the broader Gallery route regression sweep (`pytest tests/test_gallery_*.py -v --import-mode=importlib`) to confirm no regression, matching how sub-project 1's Task 2 verified its own Gallery route change.

- [ ] **Step 5: Add the frontend file input + button**

Create `static/js/editor/ai-face-swap.js`, following whatever button-registration pattern this step's own investigation (per this task's Context note) finds `ai-rembg.js` uses, but extended with a file `<input type="file">` (or drag-drop target) specifically for the source-face image, and a `FormData`-based request mirroring Style Transfer's actual frontend call in `ai-tools-misc.js` (`fd.append('image', blob, ...)` for the flattened canvas, extended with `fd.append('source_face', sourceFile, sourceFile.name)` for the newly-uploaded source) rather than `applyImageTool`'s single-image JSON contract. Wire it into whichever file registers Gallery editor toolbar buttons (found during this step's own investigation).

- [ ] **Step 6: Manually verify the Gallery UI**

Start the app, open an image in the Gallery editor, use the new face-swap action with a real source-face upload, confirm the result renders as a new layer. This is a manual, owed verification step, same as every other feature this session that touches real model inference or live UI interaction.

- [ ] **Step 7: Commit**

```bash
git add routes/gallery/gallery_routes.py static/js/editor/ai-face-swap.js tests/test_gallery_face_swap_route.py
# Add whichever toolbar-registration file Step 5 modified.
git commit -m "feat(face-swap): Gallery editor face-swap action

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
