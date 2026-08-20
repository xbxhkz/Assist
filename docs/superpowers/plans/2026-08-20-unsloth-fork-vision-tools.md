# Unsloth Fork — Vision Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five vision tools — `remove_background`, `detect_shapes`, `webcam_look`,
`edit_image_prompt`, `face_swap` — to Unsloth Studio's existing agent loop.

**Architecture:** Model wrappers are ported near-verbatim from Assist into a new package we
own, `studio/backend/core/inference/assist_vision/`. Upstream `tools.py` gets exactly two
edits (one splice into `ALL_TOOLS`, one delegating branch in `execute_tool`) so upstream merges
stay cheap. `edit_image_prompt` is rewritten against Studio's own `diffusion.py` img2img rather
than porting Assist's sd-server.

**Tech Stack:** Python 3, onnxruntime (U2Net), torchvision (Mask R-CNN), ultralytics (YOLO),
insightface (face-swap), Pillow, numpy, OpenCV, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-unsloth-fork-vision-tools-design.md` (in the
odysseus repo — read it alongside this plan).

## Global Constraints

- **Target repo is `C:\Users\Admin\unsloth`, NOT the odysseus repo.** Every path below is
  relative to that checkout unless explicitly marked "(odysseus)". Odysseus is a read-only
  source of code to port.
- **Every new file starts with the two-line SPDX header** used throughout Studio:
  `# SPDX-License-Identifier: AGPL-3.0-only` then
  `# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0`
- **Exactly two upstream touchpoints in `tools.py`.** One entry appended to `ALL_TOOLS`
  (tools.py:9804), one delegating branch in `execute_tool` (tools.py:9996). Nothing else in
  that 13,815-line file may be edited.
- **`execute_tool` is SYNCHRONOUS and returns `str`.** Tools present a sync, string-returning
  face. There is no error *field* — failures return clear error TEXT from the same string
  return. Never raise into the loop.
- **No inline base64 data URIs in returned strings.** Return a filesystem path. A data URI is
  re-sent to the model every subsequent turn and persisted into replayed history.
- **Heavy imports (`torch`, `onnxruntime`, `insightface`, `ultralytics`, `cv2`) stay inside
  functions**, never at module scope. A module-scope `import torch` cost Assist a measured
  ~2.5 s event-loop stall.
- **face_swap guardrails are non-optional:** models never bundled, explicit per-user licence
  acceptance recorded before any download, provenance metadata on every output.
- **Tests must EXECUTE tools and assert on real output** — real image in, real image out with
  asserted properties. Tests that only assert strings appear in source are forbidden: this
  project hit three separate cases where a fully green suite certified non-working code.
- **Zero detections is NOT an error** for `detect_shapes`/`webcam_look`. No face detected IS
  an error for `face_swap`.
- Run tests from `C:\Users\Admin\unsloth\studio\backend` with `python -m pytest tests/<file> -v`.

---

## File Structure

**New package** `studio/backend/core/inference/assist_vision/`:

- `__init__.py` — exports `ASSIST_VISION_TOOLS` (schema list) and `execute(name, arguments)`
- `paths.py` — input resolution + confinement (shared by all five tools)
- `bg_removal.py` — U2Net background removal
- `shape_detect.py` — torchvision Mask R-CNN segmentation
- `yolo.py` + `webcam.py` — YOLO detection and camera capture
- `face_swap.py` — InsightFace swap + licence gate + provenance
- `image_edit.py` — img2img via Studio's diffusion
- `schemas.py` — the five OpenAI-style tool schema dicts

**Modified (exactly two edits):** `studio/backend/core/inference/tools.py`

**Tests** in `studio/backend/tests/`: `test_assist_vision_paths.py`,
`test_assist_vision_bg_removal.py`, `test_assist_vision_detect.py`,
`test_assist_vision_face_swap.py`, `test_assist_vision_image_edit.py`,
`test_assist_vision_registration.py`

---

### Task 1: Input resolution and confinement

**Files:**
- Create: `studio/backend/core/inference/assist_vision/__init__.py`
- Create: `studio/backend/core/inference/assist_vision/paths.py`
- Test: `studio/backend/tests/test_assist_vision_paths.py`

**Interfaces:**
- Produces: `resolve_image_bytes(path_value: str, *, session_id: str | None = None, max_bytes: int = 26214400) -> tuple[bytes | None, str | None]`
  returning `(image_bytes, None)` on success or `(None, error_text)` on failure. Every later
  task calls this to turn a tool's `image_path` argument into bytes.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_assist_vision_paths.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Input resolution for the vision tools.

Every vision tool takes an ``image_path`` and must turn it into bytes without
becoming a file-exfiltration primitive. These tests execute the real resolver
against a real temp filesystem -- no mocks -- because the failure that matters
(reading a file outside the allowed area) is a filesystem behaviour, not a
code-shape property.
"""

import io

import pytest
from PIL import Image

from core.inference.assist_vision.paths import resolve_image_bytes


def _write_png(path, size=(8, 8), color=(200, 50, 50)):
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


class TestResolveImageBytes:
    def test_a_real_file_resolves_to_its_bytes(self, tmp_path):
        p = _write_png(tmp_path / "a.png")
        data, err = resolve_image_bytes(str(p))
        assert err is None
        assert Image.open(io.BytesIO(data)).size == (8, 8)

    def test_a_missing_file_returns_error_text_not_an_exception(self, tmp_path):
        data, err = resolve_image_bytes(str(tmp_path / "nope.png"))
        assert data is None
        assert "not found" in err.lower()

    def test_a_directory_is_rejected(self, tmp_path):
        data, err = resolve_image_bytes(str(tmp_path))
        assert data is None
        assert "not a file" in err.lower()

    def test_an_oversized_file_is_rejected_without_reading_it_all(self, tmp_path):
        big = tmp_path / "big.png"
        with open(big, "wb") as f:
            f.seek(1024)
            f.write(b"x")
        data, err = resolve_image_bytes(str(big), max_bytes=512)
        assert data is None
        assert "too large" in err.lower()

    def test_a_non_image_file_is_rejected_with_readable_text(self, tmp_path):
        junk = tmp_path / "notes.txt"
        junk.write_text("this is not an image", encoding="utf-8")
        data, err = resolve_image_bytes(str(junk))
        assert data is None
        assert "image" in err.lower()

    def test_an_empty_path_is_rejected(self):
        data, err = resolve_image_bytes("")
        assert data is None
        assert err
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_assist_vision_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.inference.assist_vision'`

- [ ] **Step 3: Create the package `__init__.py`**

Create `studio/backend/core/inference/assist_vision/__init__.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Vision tools for the Studio agent loop.

Adds background removal, object/shape detection, webcam capture, prompt-based
image editing, and face swapping. Implementations live here rather than in
``tools.py`` so upstream merges stay cheap: ``tools.py`` carries exactly two
lines referring to this package (one entry in ``ALL_TOOLS``, one branch in
``execute_tool``).

Heavy model libraries (torch, onnxruntime, insightface, ultralytics, cv2) are
imported INSIDE functions, never at module scope -- a module-scope
``import torch`` stalls the event loop for seconds on first use.
"""
```

- [ ] **Step 3b: Read how Studio confines file access, and match it**

The spec requires these tools be confined, not a read-anything primitive. Studio already has a
convention for this: `edit_file` resolves through `_get_workdir(session_id)` and honours a
`disable_sandbox` flag (see `core/inference/tools.py`, and `tests/test_edit_file_tool.py` for
how it is exercised).

Run: `grep -nE "_get_workdir|disable_sandbox|def _edit_file_write" core/inference/tools.py | head -20`

Record in the task report the exact signature of `_get_workdir` and how `edit_file` uses it.
Then implement confinement in `paths.py` the SAME way rather than importing Assist's
home-directory-plus-extra-roots model, which has no meaning here. The spec's own rule applies:
ours conforms to Studio's, not Assist's.

If `_get_workdir` turns out to be private in a way that makes reuse unwise, confine to the
session workdir by the same mechanism `edit_file` ultimately calls, and say so in the report.
Do NOT ship a resolver with no confinement — an unconfined image reader plus a model that
echoes file content is an exfiltration path.

Add these tests to `test_assist_vision_paths.py` before implementing (they will need the
workdir fixture pattern from `tests/test_edit_file_tool.py`):

```python
class TestConfinement:
    def test_a_path_outside_the_session_workdir_is_refused(self, tmp_path, workdir):
        outside = tmp_path / "outside.png"
        _write_png(outside)
        data, err = resolve_image_bytes(str(outside), session_id="t")
        assert data is None
        assert err and ("outside" in err.lower() or "not allowed" in err.lower())

    def test_a_path_inside_the_session_workdir_is_allowed(self, workdir):
        inside = _write_png(workdir / "inside.png")
        data, err = resolve_image_bytes(str(inside), session_id="t")
        assert err is None
        assert data
```

This changes `resolve_image_bytes`'s signature to
`resolve_image_bytes(path_value, *, session_id=None, max_bytes=26214400)`. Update the
Interfaces block above and every caller in Task 6 accordingly.

- [ ] **Step 4: Implement `paths.py`**

Create `studio/backend/core/inference/assist_vision/paths.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turn a tool's ``image_path`` argument into image bytes, or into error text.

Returns ``(bytes, None)`` or ``(None, error_text)`` and never raises: the tool
boundary returns strings, so a raised exception would escape into the agent
loop instead of becoming something the model can read and retry.
"""
import io
import os

_DEFAULT_MAX_BYTES = 26214400  # 25 MiB


def resolve_image_bytes(path_value, *, max_bytes = _DEFAULT_MAX_BYTES):
    """Resolve ``path_value`` to image bytes. Never raises."""
    if not path_value or not str(path_value).strip():
        return None, "image_path is required"

    candidate = os.path.abspath(os.path.expanduser(str(path_value).strip()))

    if not os.path.exists(candidate):
        return None, f"image_path not found: {path_value}"
    if not os.path.isfile(candidate):
        return None, f"image_path is not a file: {path_value}"

    try:
        size = os.path.getsize(candidate)
    except OSError as e:
        return None, f"could not read {path_value}: {e}"
    if size > max_bytes:
        return None, (
            f"image is too large ({size} bytes, max {max_bytes}). "
            "Resize it or point at a smaller file."
        )

    try:
        with open(candidate, "rb") as f:
            data = f.read()
    except OSError as e:
        return None, f"could not read {path_value}: {e}"

    # Decode-verify here so every tool gets the same clear message rather than
    # each one failing differently deep inside its own model library.
    try:
        from PIL import Image
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        return None, f"not a readable image file: {path_value}"

    return data, None
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_assist_vision_paths.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add studio/backend/core/inference/assist_vision/ studio/backend/tests/test_assist_vision_paths.py
git commit -m "feat(vision): input resolution for vision tools"
```

---

### Task 2: Background removal

**Files:**
- Create: `studio/backend/core/inference/assist_vision/bg_removal.py`
- Test: `studio/backend/tests/test_assist_vision_bg_removal.py`
- Port from (odysseus): `src/bg_removal.py`

**Interfaces:**
- Consumes: `resolve_image_bytes(path_value, *, max_bytes)` from Task 1.
- Produces: `remove_background(image_bytes: bytes, *, session=None) -> bytes` returning
  transparent PNG bytes. `session=None` is the injectable seam — tests pass a fake so no
  168 MB model is needed.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_assist_vision_bg_removal.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Background removal via the bundled U2Net ONNX model.

The model is injectable (``session=``) so these tests exercise the real
pre/post-processing -- resize, mask application, PNG encoding -- against a
stub that returns a known mask, without needing the 168 MB weight file.
The assertions are on real output pixels: a test that only checked "it
returned bytes" would pass against an implementation that ignored the mask
entirely.
"""

import io

import numpy as np
import pytest
from PIL import Image

from core.inference.assist_vision.bg_removal import remove_background


class _HalfMaskSession:
    """Returns a mask that keeps the left half and drops the right half."""

    def run(self, output_names, input_feed):
        arr = next(iter(input_feed.values()))
        n, c, h, w = arr.shape
        mask = np.zeros((n, 1, h, w), dtype=np.float32)
        mask[:, :, :, : w // 2] = 1.0
        return [mask]


def _png(size=(64, 64), color=(200, 50, 50)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class TestRemoveBackground:
    def test_output_is_a_transparent_png_of_the_same_size(self):
        out = remove_background(_png(), session=_HalfMaskSession())
        img = Image.open(io.BytesIO(out))
        assert img.format == "PNG"
        assert img.mode == "RGBA"
        assert img.size == (64, 64)

    def test_the_mask_actually_drives_transparency(self):
        """Fails if the mask is computed but never applied."""
        out = remove_background(_png(), session=_HalfMaskSession())
        img = Image.open(io.BytesIO(out)).convert("RGBA")
        left_alpha = img.getpixel((4, 32))[3]
        right_alpha = img.getpixel((60, 32))[3]
        assert left_alpha > 200, "kept region should be opaque"
        assert right_alpha < 55, "dropped region should be transparent"

    def test_corrupt_bytes_raise_rather_than_returning_a_bad_image(self):
        with pytest.raises(Exception):
            remove_background(b"not an image", session=_HalfMaskSession())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_assist_vision_bg_removal.py -v`
Expected: FAIL — `ModuleNotFoundError` for `core.inference.assist_vision.bg_removal`.

- [ ] **Step 3: Port `bg_removal.py`**

Read the source at (odysseus) `src/bg_removal.py` and port it, making exactly these changes:

1. Add the two-line SPDX header.
2. Replace `_model_path()`'s frozen-build branch (it uses `sys.frozen`/`_MEIPASS`, which does
   not apply here) with a resolver rooted at this package:
   `os.path.join(os.path.dirname(__file__), "weights", "u2net.onnx")`, overridable via the
   `UNSLOTH_U2NET_PATH` environment variable.
3. Keep `import onnxruntime` INSIDE `_get_session()` — never at module scope.
4. Keep the `session=None` injectable parameter exactly as-is; it is what makes the tests above
   run without weights.
5. Update the missing-model error text to name the new location and the env var, e.g.
   `"U2Net model not found at <path>. Download u2net.onnx into that folder or set UNSLOTH_U2NET_PATH."`

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_assist_vision_bg_removal.py -v`
Expected: 3 passed.

- [ ] **Step 5: Negative control — prove the mask test can fail**

Temporarily change the implementation so the alpha channel is set to a constant 255 instead of
the model's mask. Re-run: `test_the_mask_actually_drives_transparency` MUST fail. Revert the
change and confirm it passes again. Record both outcomes in the task report — a test that
cannot fail proves nothing.

- [ ] **Step 6: Commit**

```bash
git add studio/backend/core/inference/assist_vision/bg_removal.py studio/backend/tests/test_assist_vision_bg_removal.py
git commit -m "feat(vision): background removal via U2Net"
```

---

### Task 3: Shape detection and webcam capture

**Files:**
- Create: `studio/backend/core/inference/assist_vision/shape_detect.py`
- Create: `studio/backend/core/inference/assist_vision/yolo.py`
- Create: `studio/backend/core/inference/assist_vision/webcam.py`
- Test: `studio/backend/tests/test_assist_vision_detect.py`
- Port from (odysseus): `src/shape_detect.py`, `src/vision/yolo.py`, `src/desktop/webcam.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure model wrappers).
- Produces:
  - `shape_detect.detect(image_bytes, *, model=None, categories=None, conf=0.4) -> list[dict]`
    where each dict has `label: str`, `index: int`, `confidence: float`,
    `box: [x1,y1,x2,y2]`, `position: str`, `mask: numpy bool array`.
  - `yolo.detect(jpeg, *, model=None, conf=0.4) -> list[dict]`,
    `yolo.annotate(jpeg, dets, fmt=".jpg") -> bytes`, `yolo.summarize(dets) -> str`.
  - `webcam.capture_frame_jpeg(*, grabber=None, index=None) -> bytes`.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_assist_vision_detect.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Object/shape detection (torchvision Mask R-CNN) and webcam capture.

Models and the camera are both injectable, so these tests run the real
filtering, per-label numbering and mask thresholding against stubs -- no
170 MB download, no physical camera. Numbering is asserted explicitly
because a later sub-project depends on being able to say "the 2nd person".
"""

import io

import numpy as np
import pytest
from PIL import Image

from core.inference.assist_vision import shape_detect, webcam, yolo


def _png(size=(4, 4), color=(10, 20, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class _FakeMaskRCNN:
    def __init__(self, boxes, labels, scores, masks):
        self._out = {
            "boxes": np.array(boxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64),
            "scores": np.array(scores, dtype=np.float32),
            "masks": np.array(masks, dtype=np.float32),
        }

    def __call__(self, tensors):
        return [self._out]


_CATS = ["__background__", "person", "dog", "N/A"]


class TestShapeDetect:
    def test_a_detection_carries_label_box_and_thresholded_mask(self):
        model = _FakeMaskRCNN(
            boxes=[[1, 2, 3, 4]], labels=[1], scores=[0.92],
            masks=[[[[0.1, 0.9], [0.8, 0.2]]]],
        )
        dets = shape_detect.detect(_png(size=(2, 2)), model=model, categories=_CATS)
        assert len(dets) == 1
        assert dets[0]["label"] == "person"
        assert dets[0]["box"] == [1, 2, 3, 4]
        assert dets[0]["mask"].dtype == bool
        assert dets[0]["mask"].tolist() == [[False, True], [True, False]]

    def test_detections_are_numbered_per_label_not_globally(self):
        """person #1, person #2, dog #1 -- a global counter would give 1,2,3."""
        model = _FakeMaskRCNN(
            boxes=[[0, 0, 1, 1]] * 3, labels=[1, 1, 2], scores=[0.9, 0.9, 0.9],
            masks=[[[[0.9, 0.9], [0.9, 0.9]]]] * 3,
        )
        dets = shape_detect.detect(_png(size=(2, 2)), model=model, categories=_CATS)
        assert [(d["label"], d["index"]) for d in dets] == [
            ("person", 1), ("person", 2), ("dog", 1),
        ]

    def test_low_confidence_detections_are_dropped(self):
        model = _FakeMaskRCNN(
            boxes=[[0, 0, 1, 1]], labels=[1], scores=[0.2],
            masks=[[[[0.9, 0.9], [0.9, 0.9]]]],
        )
        assert shape_detect.detect(_png(size=(2, 2)), model=model, categories=_CATS, conf=0.4) == []

    def test_background_and_placeholder_labels_are_skipped(self):
        model = _FakeMaskRCNN(
            boxes=[[0, 0, 1, 1]] * 2, labels=[0, 3], scores=[0.99, 0.99],
            masks=[[[[0.9, 0.9], [0.9, 0.9]]]] * 2,
        )
        assert shape_detect.detect(_png(size=(2, 2)), model=model, categories=_CATS) == []

    def test_finding_nothing_is_an_empty_list_not_an_error(self):
        model = _FakeMaskRCNN(boxes=[], labels=[], scores=[], masks=np.empty((0, 1, 2, 2)))
        assert shape_detect.detect(_png(size=(2, 2)), model=model, categories=_CATS) == []

    def test_injection_bypasses_real_model_construction(self, monkeypatch):
        def _boom():
            raise AssertionError("must not build the real Mask R-CNN")

        monkeypatch.setattr(shape_detect, "_get_model", _boom)
        monkeypatch.setattr(shape_detect, "_get_categories", _boom)
        model = _FakeMaskRCNN(boxes=[], labels=[], scores=[], masks=np.empty((0, 1, 2, 2)))
        shape_detect.detect(_png(size=(2, 2)), model=model, categories=_CATS)


class TestWebcamAndYolo:
    def test_capture_returns_jpeg_bytes_from_an_injected_grabber(self):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)

        class _Grabber:
            def read(self):
                return True, frame

            def isOpened(self):
                return True

            def release(self):
                pass

        data = webcam.capture_frame_jpeg(grabber=lambda index: _Grabber())
        assert data[:2] == b"\xff\xd8", "should be a JPEG"

    def test_summarize_reports_nothing_found_without_raising(self):
        assert "no recognizable" in yolo.summarize([]).lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_assist_vision_detect.py -v`
Expected: FAIL — `ImportError` for `shape_detect` / `webcam` / `yolo`.

- [ ] **Step 3: Port the three modules**

Port (odysseus) `src/shape_detect.py`, `src/vision/yolo.py`, `src/desktop/webcam.py` into the
package, making exactly these changes:

1. Add the two-line SPDX header to each.
2. In `shape_detect.py`, replace `_model_root()`'s use of Assist's `DATA_DIR` with a
   Studio-appropriate cache directory: `os.path.join(os.path.expanduser("~"), ".unsloth",
   "assist_vision_models")`, overridable via `UNSLOTH_VISION_MODEL_DIR`. Keep the
   `torch.hub.set_dir()` call pointed at it.
3. In `yolo.py`, replace `weights_path()`'s frozen-build branch with
   `os.path.join(os.path.dirname(__file__), "weights", "yolov8n.pt")`, falling back to the
   bare `"yolov8n.pt"` (ultralytics then downloads it) if absent.
4. Keep every heavy import inside functions: `torch`/`torchvision` inside
   `shape_detect._get_model()` and `detect()`, `ultralytics` inside `yolo._get_model()`, `cv2`
   inside the functions that use it.
5. Keep every injectable seam unchanged: `model=`, `categories=`, `grabber=`, `index=`.
6. Keep the per-label `index` numbering in `_format_detections` — it is asserted above and a
   later sub-project depends on it.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_assist_vision_detect.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add studio/backend/core/inference/assist_vision/ studio/backend/tests/test_assist_vision_detect.py
git commit -m "feat(vision): shape detection, YOLO and webcam capture"
```

---

### Task 4: Face swap with licence gate and provenance

**Files:**
- Create: `studio/backend/core/inference/assist_vision/face_swap.py`
- Test: `studio/backend/tests/test_assist_vision_face_swap.py`
- Port from (odysseus): `src/face_swap.py`

**Interfaces:**
- Produces:
  - `swap_face(source_face_bytes, target_image_bytes, *, analyzer=None, swapper=None) -> bytes`
    returning PNG bytes carrying provenance metadata.
  - `LicenseNotAcceptedError`, `NoFaceDetectedError` exception classes.
  - `licence_accepted() -> bool` and `record_licence_acceptance() -> None` backed by a file
    under the model cache directory.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_assist_vision_face_swap.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Face swapping, and the guardrails that make shipping it defensible.

InsightFace's models are non-commercial/research-only, so they are never
bundled and never fetched until the user has explicitly accepted the licence.
Outputs carry provenance metadata identifying them as AI-face-swapped.

These tests assert the GATE holds even when the models are injected -- a gate
that only guarded model construction would be bypassed by any caller that
supplies its own analyzer, which is exactly the bug this shape once had.
"""

import io

import numpy as np
import pytest
from PIL import Image

from core.inference.assist_vision import face_swap


def _png(size=(64, 64), color=(200, 100, 50)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class _Face:
    def __init__(self, kps=True):
        self.kps = np.zeros((5, 2), dtype=np.float32) if kps else None


class _Analyzer:
    def __init__(self, faces_by_call):
        self._faces = list(faces_by_call)

    def get(self, img):
        return self._faces.pop(0)


class _Swapper:
    def get(self, target_img, target_face, source_face, paste_back=True):
        return 255 - target_img  # distinguishable, so a dropped result is visible


@pytest.fixture
def accepted(monkeypatch):
    monkeypatch.setattr(face_swap, "licence_accepted", lambda: True)


class TestLicenceGate:
    def test_swapping_without_acceptance_is_refused_even_with_models_injected(self, monkeypatch):
        monkeypatch.setattr(face_swap, "licence_accepted", lambda: False)
        with pytest.raises(face_swap.LicenseNotAcceptedError):
            face_swap.swap_face(
                _png(), _png(),
                analyzer=_Analyzer([[_Face()], [_Face()]]), swapper=_Swapper(),
            )

    def test_recording_acceptance_makes_licence_accepted_true(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UNSLOTH_VISION_MODEL_DIR", str(tmp_path))
        assert face_swap.licence_accepted() is False
        face_swap.record_licence_acceptance()
        assert face_swap.licence_accepted() is True


class TestSwap:
    def test_output_is_png_carrying_provenance_metadata(self, accepted):
        out = face_swap.swap_face(
            _png(), _png(color=(200, 100, 50)),
            analyzer=_Analyzer([[_Face()], [_Face()]]), swapper=_Swapper(),
        )
        img = Image.open(io.BytesIO(out))
        assert img.format == "PNG"
        assert img.text.get("assist:ai-edited") == "face-swap"

    def test_the_swapper_result_is_actually_used(self, accepted):
        """Fails if the target image is re-encoded untouched."""
        out = face_swap.swap_face(
            _png(), _png(color=(200, 100, 50)),
            analyzer=_Analyzer([[_Face()], [_Face()]]), swapper=_Swapper(),
        )
        px = Image.open(io.BytesIO(out)).convert("RGB").getpixel((0, 0))
        assert all(abs(a - b) <= 2 for a, b in zip(px, (55, 155, 205)))

    def test_no_face_in_source_is_an_error(self, accepted):
        with pytest.raises(face_swap.NoFaceDetectedError):
            face_swap.swap_face(_png(), _png(), analyzer=_Analyzer([[]]), swapper=_Swapper())

    def test_a_face_without_landmarks_is_an_error_not_a_crash(self, accepted):
        """Guards the real bug: kps=None crashed inside InsightFace as
        "'NoneType' object has no attribute 'shape'"."""
        with pytest.raises(face_swap.NoFaceDetectedError):
            face_swap.swap_face(
                _png(), _png(),
                analyzer=_Analyzer([[_Face(kps=False)], [_Face()]]), swapper=_Swapper(),
            )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_assist_vision_face_swap.py -v`
Expected: FAIL — `ImportError` for `face_swap`.

- [ ] **Step 3: Port `face_swap.py`**

Port (odysseus) `src/face_swap.py`, making exactly these changes:

1. Add the two-line SPDX header.
2. Replace Assist's settings-backed gate (`get_setting("face_swap_license_accepted")`) with a
   file-backed one, since Studio has no equivalent setting:

```python
def _licence_marker_path():
    return os.path.join(_model_root(), "INSIGHTFACE_LICENCE_ACCEPTED")


def licence_accepted():
    """True only once the user has explicitly accepted InsightFace's terms."""
    return os.path.isfile(_licence_marker_path())


def record_licence_acceptance():
    """Record explicit acceptance. Called only from an explicit user action."""
    os.makedirs(_model_root(), exist_ok=True)
    with open(_licence_marker_path(), "w", encoding="utf-8") as f:
        f.write("InsightFace models are licensed for non-commercial research use only.\n")
```

3. Keep `_ensure_models_available()` called UNCONDITIONALLY as the first statement of
   `swap_face()`, not only inside the singleton getters. Gating only the getters lets any
   caller that injects `analyzer=`/`swapper=` bypass the licence entirely — that was a real
   bug in this code's history and the test above pins it.
4. Keep the `kps is None` guard on both source and target faces, raising `NoFaceDetectedError`
   with actionable text.
5. Keep `_model_root()` pointed at the same cache directory Task 3 uses
   (`UNSLOTH_VISION_MODEL_DIR`, default `~/.unsloth/assist_vision_models`).
6. Keep `import insightface` inside the getters.
7. Keep the provenance `PngInfo` stamping unchanged.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_assist_vision_face_swap.py -v`
Expected: 6 passed.

- [ ] **Step 5: Negative control — prove the gate test can fail**

Temporarily move the `_ensure_models_available()` call out of `swap_face()` so it only runs
inside the singleton getters. Re-run:
`test_swapping_without_acceptance_is_refused_even_with_models_injected` MUST fail (injected
models skip the getters). Revert and confirm it passes. Record both outcomes.

- [ ] **Step 6: Commit**

```bash
git add studio/backend/core/inference/assist_vision/face_swap.py studio/backend/tests/test_assist_vision_face_swap.py
git commit -m "feat(vision): face swap with licence gate and provenance metadata"
```

---

### Task 5: Prompt-based image editing via Studio's diffusion

**Files:**
- Create: `studio/backend/core/inference/assist_vision/image_edit.py`
- Test: `studio/backend/tests/test_assist_vision_image_edit.py`
- Read first: `studio/backend/core/inference/diffusion.py` (its img2img entry point)

**Interfaces:**
- Produces: `edit_image(image_bytes: bytes, prompt: str, *, backend=None, strength: float = 0.6) -> bytes`
  returning edited PNG bytes. `backend=None` is the injectable seam.

**This is the only task not backed by proven ported code.** Assist's version drove its own
sd-server; this one drives Studio's existing diffusion. Read `diffusion.py` first and find the
img2img entry point — grep for `init_image`, `strength`, and `Img2Img`. Adapt the call below to
whatever that function actually is; do not invent a signature.

- [ ] **Step 1: Read Studio's diffusion img2img entry point**

Run: `grep -nE "def .*img2img|init_image|strength|class .*Img2Img" core/inference/diffusion.py | head -30`

Record in the task report the exact function name and signature you will call. If no callable
img2img entry point exists (only internal plumbing), STOP and report BLOCKED rather than
inventing one — the spec's fallback is to defer this tool to its own sub-project.

- [ ] **Step 2: Write the failing test**

Create `studio/backend/tests/test_assist_vision_image_edit.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Prompt-based image editing, driven by Studio's own diffusion backend.

Unlike the other vision tools this is not a port -- Assist drove its own
sd-server, and this drives Studio's existing img2img. The backend is
injectable so these tests pin the contract (the prompt and the source image
both reach it, and its result is what comes back) without loading a
diffusion model.
"""

import io

import pytest
from PIL import Image

from core.inference.assist_vision.image_edit import edit_image


def _png(size=(32, 32), color=(10, 120, 200)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class _Backend:
    def __init__(self):
        self.calls = []

    def __call__(self, *, image_bytes, prompt, strength):
        self.calls.append({"prompt": prompt, "strength": strength, "size": len(image_bytes)})
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (240, 30, 30)).save(buf, format="PNG")
        return buf.getvalue()


class TestEditImage:
    def test_the_prompt_and_source_image_reach_the_backend(self):
        backend = _Backend()
        edit_image(_png(), "make the sky orange", backend=backend)
        assert backend.calls[0]["prompt"] == "make the sky orange"
        assert backend.calls[0]["size"] > 0

    def test_the_backend_result_is_returned_not_the_original(self):
        """Fails if the source image is passed through untouched."""
        out = edit_image(_png(), "anything", backend=_Backend())
        assert Image.open(io.BytesIO(out)).convert("RGB").getpixel((0, 0)) == (240, 30, 30)

    def test_an_empty_prompt_is_rejected(self):
        with pytest.raises(ValueError):
            edit_image(_png(), "   ", backend=_Backend())

    def test_strength_is_forwarded(self):
        backend = _Backend()
        edit_image(_png(), "x", backend=backend, strength=0.25)
        assert backend.calls[0]["strength"] == 0.25
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest tests/test_assist_vision_image_edit.py -v`
Expected: FAIL — `ImportError` for `image_edit`.

- [ ] **Step 4: Implement `image_edit.py`**

Create the module with this shape, replacing `_studio_backend()`'s body with a call to the real
entry point you recorded in Step 1:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Prompt-based image editing (img2img) using Studio's own diffusion backend.

Deliberately does NOT ship a second diffusion stack: Studio already loads and
manages diffusion models, so this tool inherits whatever the user has.
"""


def _studio_backend():
    """Return a callable(image_bytes=, prompt=, strength=) -> bytes.

    Imported lazily: diffusion pulls heavy libraries that must not load at
    module import time.
    """
    from core.inference import diffusion  # noqa: F401
    # Replace with the real entry point recorded in Step 1.
    raise NotImplementedError("wire to diffusion's img2img entry point")


def edit_image(image_bytes, prompt, *, backend = None, strength = 0.6):
    """Apply a natural-language edit to ``image_bytes``. Returns PNG bytes."""
    if not prompt or not str(prompt).strip():
        raise ValueError("a prompt describing the edit is required")
    call = backend if backend is not None else _studio_backend()
    return call(image_bytes = image_bytes, prompt = str(prompt).strip(), strength = strength)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_assist_vision_image_edit.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add studio/backend/core/inference/assist_vision/image_edit.py studio/backend/tests/test_assist_vision_image_edit.py
git commit -m "feat(vision): prompt-based image editing via Studio diffusion"
```

---

### Task 6: Tool schemas, dispatch, and registration into Studio's loop

**Files:**
- Create: `studio/backend/core/inference/assist_vision/schemas.py`
- Modify: `studio/backend/core/inference/assist_vision/__init__.py`
- Modify: `studio/backend/core/inference/tools.py:9804` (ALL_TOOLS) and `:9996` (execute_tool)
- Test: `studio/backend/tests/test_assist_vision_registration.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: `ASSIST_VISION_TOOLS: list[dict]` (five OpenAI-style schemas) and
  `execute(name: str, arguments: dict, *, session_id=None) -> str`.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_assist_vision_registration.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The vision tools must be REACHABLE through Studio's own loop.

Being defined is not the same as being wired up. These tests go through the
real ``ALL_TOOLS`` list and the real ``execute_tool`` dispatcher, so a tool
that exists but was never registered fails here.
"""

import io

import pytest
from PIL import Image

from core.inference.tools import ALL_TOOLS, execute_tool

_NAMES = {
    "remove_background", "detect_shapes", "webcam_look",
    "edit_image_prompt", "face_swap",
}


def _png(path, size=(16, 16)):
    Image.new("RGB", size, (10, 20, 30)).save(path, format="PNG")
    return str(path)


class TestRegistration:
    def test_every_vision_tool_appears_in_all_tools(self):
        names = {t["function"]["name"] for t in ALL_TOOLS}
        assert _NAMES <= names, f"missing: {_NAMES - names}"

    def test_every_schema_is_openai_shaped(self):
        for tool in ALL_TOOLS:
            if tool["function"]["name"] in _NAMES:
                assert tool["type"] == "function"
                assert tool["function"]["description"].strip()
                assert tool["function"]["parameters"]["type"] == "object"

    def test_the_upstream_tool_names_are_not_disturbed(self):
        """The two-line edit must ADD tools, never replace Studio's own."""
        names = {t["function"]["name"] for t in ALL_TOOLS}
        assert {"web_search", "python", "terminal", "edit_file"} <= names


class TestDispatch:
    def test_dispatch_reaches_the_tool_and_returns_a_string(self, tmp_path, monkeypatch):
        import core.inference.assist_vision as av

        monkeypatch.setattr(
            av, "execute",
            lambda name, arguments, **kw: f"dispatched:{name}:{arguments.get('image_path')}",
        )
        out = execute_tool("remove_background", {"image_path": _png(tmp_path / "a.png")},
                           session_id="t")
        assert isinstance(out, str)
        assert out.startswith("dispatched:remove_background:")

    def test_a_missing_image_path_returns_error_text_not_an_exception(self, tmp_path):
        out = execute_tool("remove_background",
                           {"image_path": str(tmp_path / "nope.png")}, session_id="t")
        assert isinstance(out, str)
        assert "not found" in out.lower()

    def test_detect_shapes_finding_nothing_is_not_reported_as_an_error(self, tmp_path, monkeypatch):
        from core.inference.assist_vision import shape_detect

        monkeypatch.setattr(shape_detect, "detect", lambda *a, **k: [])
        out = execute_tool("detect_shapes", {"image_path": _png(tmp_path / "b.png")},
                           session_id="t")
        assert "error" not in out.lower()
        assert "no recognizable" in out.lower()

    def test_no_returned_string_contains_a_base64_data_uri(self, tmp_path, monkeypatch):
        """Data URIs get replayed into model context forever."""
        from core.inference.assist_vision import shape_detect

        monkeypatch.setattr(shape_detect, "detect", lambda *a, **k: [])
        out = execute_tool("detect_shapes", {"image_path": _png(tmp_path / "c.png")},
                           session_id="t")
        assert "data:image" not in out
        assert "base64" not in out
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_assist_vision_registration.py -v`
Expected: FAIL — the vision names are absent from `ALL_TOOLS`.

- [ ] **Step 3: Write `schemas.py`**

Create `studio/backend/core/inference/assist_vision/schemas.py` with five schema dicts
following the exact shape of Studio's own (see `WEB_SEARCH_TOOL` at `tools.py:9353`):

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI-style schemas for the vision tools, shaped like Studio's own."""

_IMAGE_PATH = {
    "type": "string",
    "description": "Absolute path to an image file on this machine.",
}

REMOVE_BACKGROUND_TOOL = {
    "type": "function",
    "function": {
        "name": "remove_background",
        "description": (
            "Remove the background from an image, returning a transparent PNG. "
            "Returns the path of the written file."
        ),
        "parameters": {
            "type": "object",
            "properties": {"image_path": _IMAGE_PATH},
            "required": ["image_path"],
        },
    },
}

DETECT_SHAPES_TOOL = {
    "type": "function",
    "function": {
        "name": "detect_shapes",
        "description": (
            "Detect and identify subjects (people, animals, objects) in a photo. "
            "Returns what was found with confidence and rough position, plus the "
            "path of an annotated copy. Finding nothing is a normal result."
        ),
        "parameters": {
            "type": "object",
            "properties": {"image_path": _IMAGE_PATH},
            "required": ["image_path"],
        },
    },
}

WEBCAM_LOOK_TOOL = {
    "type": "function",
    "function": {
        "name": "webcam_look",
        "description": (
            "Capture a frame from the local webcam and identify objects in it. "
            "Returns what was found plus the path of an annotated image."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "camera_index": {
                    "type": "integer",
                    "description": "Camera index, default 0.",
                }
            },
        },
    },
}

EDIT_IMAGE_PROMPT_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_image_prompt",
        "description": (
            "Edit an image by describing the change in natural language "
            "(e.g. 'make the sky sunset-coloured'). Returns the path of the edited image."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": _IMAGE_PATH,
                "prompt": {
                    "type": "string",
                    "description": "Natural-language description of the edit.",
                },
                "strength": {
                    "type": "number",
                    "description": "How much to change the image, 0-1. Default 0.6.",
                },
            },
            "required": ["image_path", "prompt"],
        },
    },
}

FACE_SWAP_TOOL = {
    "type": "function",
    "function": {
        "name": "face_swap",
        "description": (
            "Swap the face from a source image into a target image. Requires the user "
            "to have accepted the InsightFace model licence first (non-commercial "
            "research use only) -- you cannot accept it on their behalf. Outputs carry "
            "metadata marking them as AI-edited. Returns the path of the result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_face_path": {
                    "type": "string",
                    "description": "Absolute path to the image containing the source face.",
                },
                "target_image_path": {
                    "type": "string",
                    "description": "Absolute path to the image to swap the face into.",
                },
            },
            "required": ["source_face_path", "target_image_path"],
        },
    },
}

ASSIST_VISION_TOOLS = [
    REMOVE_BACKGROUND_TOOL,
    DETECT_SHAPES_TOOL,
    WEBCAM_LOOK_TOOL,
    EDIT_IMAGE_PROMPT_TOOL,
    FACE_SWAP_TOOL,
]

ASSIST_VISION_TOOL_NAMES = frozenset(t["function"]["name"] for t in ASSIST_VISION_TOOLS)
```

- [ ] **Step 4: Implement `execute()` in `__init__.py`**

Append this to `studio/backend/core/inference/assist_vision/__init__.py`:

```python
import os
import tempfile

from .schemas import ASSIST_VISION_TOOLS, ASSIST_VISION_TOOL_NAMES  # noqa: F401


def _write_png(data):
    """Write PNG bytes to a temp file and return its absolute path.

    Tools return a PATH, never an inline data: URI -- a data URI is re-sent to
    the model on every later turn and persisted into replayed history.
    """
    fd, path = tempfile.mkstemp(suffix = ".png", prefix = "unsloth_vision_")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _do_remove_background(arguments, session_id):
    from .bg_removal import remove_background
    from .paths import resolve_image_bytes

    data, err = resolve_image_bytes(arguments.get("image_path"), session_id = session_id)
    if err:
        return f"remove_background failed: {err}"
    out = remove_background(data)
    return f"Background removed. Transparent PNG written to: {_write_png(out)}"


def _describe(dets):
    """One line per label group, or a plain not-found line. Never 'error'."""
    if not dets:
        return "No recognizable objects detected."
    groups = {}
    for d in dets:
        groups.setdefault(d["label"], []).append(d)
    parts = []
    for label, items in groups.items():
        confs = ", ".join(f"{int(i['confidence'] * 100)}%" for i in items)
        positions = ", ".join(sorted({i["position"] for i in items}))
        parts.append(f"{len(items)} {label} ({confs}; {positions})")
    return ", ".join(parts)


def _do_detect_shapes(arguments, session_id):
    from . import shape_detect, yolo
    from .paths import resolve_image_bytes

    data, err = resolve_image_bytes(arguments.get("image_path"), session_id = session_id)
    if err:
        return f"detect_shapes failed: {err}"
    dets = shape_detect.detect(data)
    summary = _describe(dets)
    if not dets:
        return summary
    annotated = yolo.annotate(data, dets, ".png")
    return f"{summary}\nAnnotated image written to: {_write_png(annotated)}"


def _do_webcam_look(arguments, session_id):
    from . import webcam, yolo

    index = arguments.get("camera_index")
    frame = webcam.capture_frame_jpeg(index = index)
    dets = yolo.detect(frame)
    summary = yolo.summarize(dets)
    if not dets:
        return summary
    annotated = yolo.annotate(frame, dets, ".png")
    return f"{summary}\nAnnotated image written to: {_write_png(annotated)}"


def _do_edit_image_prompt(arguments, session_id):
    from .image_edit import edit_image
    from .paths import resolve_image_bytes

    data, err = resolve_image_bytes(arguments.get("image_path"), session_id = session_id)
    if err:
        return f"edit_image_prompt failed: {err}"
    strength = arguments.get("strength")
    kwargs = {} if strength is None else {"strength": float(strength)}
    out = edit_image(data, arguments.get("prompt", ""), **kwargs)
    return f"Image edited. Result written to: {_write_png(out)}"


def _do_face_swap(arguments, session_id):
    from .face_swap import LicenseNotAcceptedError, NoFaceDetectedError, swap_face
    from .paths import resolve_image_bytes

    source, err = resolve_image_bytes(arguments.get("source_face_path"), session_id = session_id)
    if err:
        return f"face_swap failed: {err}"
    target, err = resolve_image_bytes(arguments.get("target_image_path"), session_id = session_id)
    if err:
        return f"face_swap failed: {err}"
    try:
        out = swap_face(source, target)
    except LicenseNotAcceptedError:
        return (
            "face_swap is unavailable until the InsightFace model licence is accepted. "
            "Those models are licensed for non-commercial research use only, and the user "
            "must accept the terms themselves -- you cannot accept on their behalf. Ask "
            "them to accept it in Unsloth Studio's settings, then try again."
        )
    except NoFaceDetectedError as e:
        return f"face_swap failed: {e}"
    return (
        "Face swapped. Result written to: "
        f"{_write_png(out)} (carries metadata marking it AI-edited)"
    )


_HANDLERS = {
    "remove_background": _do_remove_background,
    "detect_shapes": _do_detect_shapes,
    "webcam_look": _do_webcam_look,
    "edit_image_prompt": _do_edit_image_prompt,
    "face_swap": _do_face_swap,
}


def execute(name, arguments, *, session_id = None):
    """Run a vision tool. Always returns str; never raises into the agent loop."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"unknown vision tool: {name}"
    try:
        return handler(arguments or {}, session_id)
    except Exception as e:  # noqa: BLE001 - the tool boundary must not raise
        return f"{name} failed: {type(e).__name__}: {e}"
```

- [ ] **Step 5: Make the two upstream edits in `tools.py`**

Edit 1 — splice into `ALL_TOOLS` (tools.py:9804). Change:

```python
ALL_TOOLS = [
    WEB_SEARCH_TOOL,
    PYTHON_TOOL,
    TERMINAL_TOOL,
    EDIT_FILE_TOOL,
    RENDER_HTML_TOOL,
    SEARCH_KNOWLEDGE_BASE_TOOL,
    SEARCH_CONVERSATION_TOOL,
]
```

to:

```python
from core.inference.assist_vision import ASSIST_VISION_TOOLS, ASSIST_VISION_TOOL_NAMES

ALL_TOOLS = [
    WEB_SEARCH_TOOL,
    PYTHON_TOOL,
    TERMINAL_TOOL,
    EDIT_FILE_TOOL,
    RENDER_HTML_TOOL,
    SEARCH_KNOWLEDGE_BASE_TOOL,
    SEARCH_CONVERSATION_TOOL,
    *ASSIST_VISION_TOOLS,
]
```

Edit 2 — add ONE delegating branch as the first `if` inside `execute_tool` (tools.py:9996),
immediately after the existing `logger.info(...)` line:

```python
    if name in ASSIST_VISION_TOOL_NAMES:
        from core.inference import assist_vision
        return assist_vision.execute(name, arguments, session_id = session_id)
```

Make no other edits to `tools.py`. Verify with
`git diff --stat core/inference/tools.py` that only these lines changed.

- [ ] **Step 6: Run the test to verify it passes**

Run: `python -m pytest tests/test_assist_vision_registration.py -v`
Expected: 7 passed.

- [ ] **Step 7: Run the whole vision suite plus Studio's own tool tests**

Run:
```
python -m pytest tests/test_assist_vision_paths.py tests/test_assist_vision_bg_removal.py \
  tests/test_assist_vision_detect.py tests/test_assist_vision_face_swap.py \
  tests/test_assist_vision_image_edit.py tests/test_assist_vision_registration.py \
  tests/test_edit_file_tool.py tests/test_tool_approvals.py -v
```
Expected: all pass. Studio's own tool tests must be unaffected — if `test_edit_file_tool.py`
breaks, the two-line edit disturbed something it should not have.

- [ ] **Step 8: Commit**

```bash
git add studio/backend/core/inference/assist_vision/ studio/backend/core/inference/tools.py \
        studio/backend/tests/test_assist_vision_registration.py
git commit -m "feat(vision): register vision tools into Studio's agent loop"
```

---

## After Task 6

Run Studio's broader test suite and compare against a baseline captured BEFORE any of this
work (`git stash` the changes, run, unstash) so pre-existing failures are not mistaken for
regressions. Then follow `superpowers:finishing-a-development-branch`.

Manual verification is owed and no automated test replaces it: run Studio, ask the agent to
remove a background, detect objects in a photo, and look through the webcam, and confirm the
outputs are actually correct — not merely that files were produced.
