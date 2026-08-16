# Shape Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `detect_shapes`, a builtin agent tool that detects labeled subjects (people,
animals, other objects) in a photo — via chat attachment or a real filesystem path/filename —
and returns a text summary plus the photo shown back with boxes and labels drawn on. The
underlying detection produces real per-instance segmentation masks (not exposed through the
tool's own output) so a future `swap_shape` sub-project can reuse the same pipeline directly.

**Architecture:** A new module `src/shape_detect.py` mirrors `src/face_swap.py`/
`src/bg_removal.py`'s shape — a lazy-loaded model singleton (`torchvision`'s pretrained
`maskrcnn_resnet50_fpn`) behind an injectable-model test seam — wrapped by a new builtin agent
tool in `src/agent_tools/image_tools.py` that reuses the `_resolve_image_source`/
`_image_result`/`_default_gallery_saver` helpers already shipped this session, and is registered
with the same 8-point checklist `face_swap`/`remove_background`/`edit_image_prompt` already went
through.

**Tech Stack:** `torchvision.models.detection.maskrcnn_resnet50_fpn` (BSD-3-Clause, COCO-
pretrained, ~170MB, downloaded on first use — not bundled), PIL, numpy, reused pieces of
`src/vision/yolo.py` (`_position`, `summarize`, `annotate`).

**Spec:** `docs/superpowers/specs/2026-08-15-shape-detection-design.md`

## Global Constraints

- No license-acceptance gate — `torchvision` and its pretrained weights are permissively
  licensed (verified live against their actual LICENSE file), unlike `face_swap`'s InsightFace
  models. This is a plain download-if-missing step, nothing to gate behind consent.
- Model weights are downloaded on first use, never bundled in the installer (installer-size
  tradeoff, not a consent question — see spec Context).
- No tunable parameters exposed to the model or user (confidence threshold stays the same
  default `src/vision/yolo.py` already uses: `0.4`) — matches this app's "no knobs" precedent
  for v1 image tools.
- Zero detections is a normal, non-error result, not an exception — unlike `face_swap`'s
  `NoFaceDetectedError`, "nothing recognizable here" is a valid answer for a detector.
- Privilege gate matches `remove_background`/`edit_image_prompt`/`face_swap` exactly
  (`can_generate_images`), not `webcam_look`'s stricter admin-only gate — this tool never
  touches a live camera, only a photo it was already given.
- Input resolution reuses `_resolve_image_source`/`_resolve_path_bytes`
  (`src/agent_tools/image_tools.py`) unchanged — accepts `attachment_id` OR `image_path`
  (exactly one), any common image format, no new resolution logic to write.
- Swapping itself is explicitly out of scope — a deliberate follow-up sub-project, not part of
  this plan.

---

### Task 1: Bundle `torchvision` as a declared dependency

**Files:**
- Modify: `requirements.txt` (after the `insightface` block, ~line 69)
- Modify: `Assist.spec` (the `collect_all` tuple, ~line 10-29)
- Test: `tests/test_torchvision_packaging.py`

**Interfaces:**
- Produces: nothing new callable — this task only makes `torchvision` a real, declared,
  PyInstaller-collected dependency instead of an implicit transitive one (currently pulled in
  only via `ultralytics`'s own dependency chain).

- [ ] **Step 1: Write the failing test**

Create `tests/test_torchvision_packaging.py`:

```python
"""torchvision must be a real, declared dependency (not just an implicit
transitive of ultralytics) and must be collected by PyInstaller in
Assist.spec, mirroring onnxruntime/ultralytics/insightface. See
docs/superpowers/specs/2026-08-15-shape-detection-design.md."""
from pathlib import Path

_SPEC_FILE = Path(__file__).resolve().parent.parent / "Assist.spec"
_REQUIREMENTS_FILE = Path(__file__).resolve().parent.parent / "requirements.txt"


def test_torchvision_importable():
    import torchvision  # noqa: F401


def test_maskrcnn_importable():
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    assert callable(maskrcnn_resnet50_fpn)
    assert MaskRCNN_ResNet50_FPN_Weights.DEFAULT is not None


def test_requirements_declares_torchvision():
    text = _REQUIREMENTS_FILE.read_text(encoding="utf-8")
    assert "torchvision" in text


def test_assist_spec_collects_torchvision():
    text = _SPEC_FILE.read_text(encoding="utf-8")
    assert '"torchvision"' in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_torchvision_packaging.py -v --import-mode=importlib`
Expected: `test_requirements_declares_torchvision` and `test_assist_spec_collects_torchvision`
FAIL (the string isn't in either file yet). The two importability tests will already PASS in
this dev environment since `torchvision` is present transitively — that's fine and expected;
they exist to catch the frozen-build packaging gap the other two tests check for, and to give
a clear signal in an environment where torchvision genuinely isn't installed.

- [ ] **Step 3: Declare the dependency**

In `requirements.txt`, immediately after the existing `insightface` line (~line 69), add:

```
# Shape detection (the detect_shapes tool): torchvision's pretrained Mask
# R-CNN provides labeled, per-instance segmentation (masks, not just boxes)
# -- BSD-3-Clause, unlike webcam_look's AGPL-3.0 ultralytics/YOLO pipeline
# (see docs/superpowers/specs/2026-08-15-shape-detection-design.md for why
# that matters). Declared explicitly rather than relying on it being pulled
# in transitively via ultralytics. The frozen build bundles the package
# itself via collect_all in Assist.spec; the ~170MB pretrained weights are
# downloaded on first use, not part of that bundle (see src/shape_detect.py).
torchvision
```

In `Assist.spec`, add `"torchvision"` to the `collect_all` tuple (the loop over
`chromadb`/`onnxruntime`/.../`insightface`), immediately after `"insightface"`:

```python
             "insightface",
             # Shape detection (detect_shapes tool): torchvision's Mask
             # R-CNN. collect_all pulls its submodules/data (torch already
             # handled via ultralytics' own entry above). Pretrained weights
             # are downloaded on first use, not part of this bundle.
             "torchvision"):
```

(This replaces the existing `"insightface"):` closing line — the tuple's trailing `):` moves to
follow the new `"torchvision"` entry.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_torchvision_packaging.py -v --import-mode=importlib`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt Assist.spec tests/test_torchvision_packaging.py
git commit -m "build: declare torchvision as a real dependency for shape detection"
```

---

### Task 2: `src/shape_detect.py` core module

**Files:**
- Create: `src/shape_detect.py`
- Modify: `src/vision/yolo.py:82-96` (`annotate()` gets an optional `fmt` parameter)
- Test: `tests/test_shape_detect.py`
- Test: `tests/test_yolo_detect.py` (one new test for the `fmt` parameter)

**Interfaces:**
- Consumes: `src.vision.yolo._position(cx, cy, w, h) -> str` (existing, unchanged — imported
  directly; it's underscore-prefixed but this is the same codebase reusing genuinely shared pure
  logic, not a rename-worthy public-API change).
- Produces:
  - `detect(image_bytes: bytes, *, model=None, categories=None, conf: float = 0.4) -> list[dict]`
    — each dict: `{"label": str, "confidence": float, "box": [x1,y1,x2,y2], "position": str,
    "mask": np.ndarray[bool, (H, W)]}`. Later tasks (the tool) call this via
    `asyncio.to_thread`.
  - `src.vision.yolo.annotate(image_bytes, dets, fmt=".jpg") -> bytes` — now accepts an optional
    third positional/keyword `fmt` (default unchanged, preserving `webcam_look`'s existing
    un-migrated call site exactly).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_shape_detect.py`:

```python
"""shape_detect.detect() runs torchvision's pretrained Mask R-CNN via an
injectable model + categories list, mirroring src/face_swap.py's
analyzer=None/swapper=None pattern -- tests never need the real (170MB,
downloaded-on-first-use) model weights. See
docs/superpowers/specs/2026-08-15-shape-detection-design.md.
"""
import io

import numpy as np
import pytest
from PIL import Image

from src import shape_detect


def _make_png(size=(4, 4), color=(200, 50, 50)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeModel:
    """Mirrors the real torchvision model's calling convention (called with
    a list of tensors, returns a list of dicts with boxes/labels/scores/
    masks) but returns plain numpy arrays -- detect()'s _to_numpy() helper
    accepts either, so fakes never need to construct real torch tensors."""

    def __init__(self, boxes, labels, scores, masks):
        self._out = {
            "boxes": np.array(boxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64),
            "scores": np.array(scores, dtype=np.float32),
            "masks": np.array(masks, dtype=np.float32),
        }

    def __call__(self, tensors):
        return [self._out]


_CATEGORIES = ["__background__", "person", "dog", "N/A"]


def test_detect_returns_labeled_detection_with_thresholded_mask():
    # 2x2 toy mask: top-right and bottom-left pixels are "on" (>= 0.5).
    model = _FakeModel(
        boxes=[[1, 2, 3, 4]], labels=[1], scores=[0.92],
        masks=[[[[0.1, 0.9], [0.8, 0.2]]]],
    )

    dets = shape_detect.detect(_make_png(size=(2, 2)), model=model, categories=_CATEGORIES)

    assert len(dets) == 1
    d = dets[0]
    assert d["label"] == "person"
    assert d["confidence"] == 0.92
    assert d["box"] == [1, 2, 3, 4]
    assert d["position"]  # non-empty; exact grid label is _position()'s own concern
    assert d["mask"].dtype == bool
    assert d["mask"].tolist() == [[False, True], [True, False]]


def test_detect_filters_below_confidence_threshold():
    model = _FakeModel(
        boxes=[[0, 0, 1, 1]], labels=[1], scores=[0.2],
        masks=[[[[0.9, 0.9], [0.9, 0.9]]]],
    )

    dets = shape_detect.detect(_make_png(size=(2, 2)), model=model, categories=_CATEGORIES, conf=0.4)

    assert dets == []


def test_detect_skips_background_and_na_labels():
    model = _FakeModel(
        boxes=[[0, 0, 1, 1], [0, 0, 1, 1]], labels=[0, 3], scores=[0.99, 0.99],
        masks=[[[[0.9, 0.9], [0.9, 0.9]]], [[[0.9, 0.9], [0.9, 0.9]]]],
    )

    dets = shape_detect.detect(_make_png(size=(2, 2)), model=model, categories=_CATEGORIES)

    assert dets == []


def test_detect_does_not_construct_real_model_when_injected(monkeypatch):
    """Injection must genuinely bypass real (170MB, network-fetching) model
    construction, not just the inference call."""
    def _fail_if_called():
        raise AssertionError("should not construct the real Mask R-CNN model")

    monkeypatch.setattr(shape_detect, "_get_model", _fail_if_called)
    monkeypatch.setattr(shape_detect, "_get_categories", _fail_if_called)

    model = _FakeModel(boxes=[], labels=[], scores=[], masks=np.empty((0, 1, 2, 2)))
    shape_detect.detect(_make_png(size=(2, 2)), model=model, categories=_CATEGORIES)


def test_detect_returns_empty_list_when_nothing_detected():
    model = _FakeModel(boxes=[], labels=[], scores=[], masks=np.empty((0, 1, 2, 2)))

    dets = shape_detect.detect(_make_png(size=(2, 2)), model=model, categories=_CATEGORIES)

    assert dets == []
```

Add to `tests/test_yolo_detect.py` (alongside the existing `test_annotate_returns_decodable_jpeg`):

```python
def test_annotate_supports_png_format():
    jpeg = _jpeg()
    dets = [{"label": "cup", "confidence": 0.7, "box": [5, 5, 40, 40], "position": "left"}]
    out = yolo.annotate(jpeg, dets, fmt=".png")
    assert out[:8] == b"\x89PNG\r\n\x1a\n"
    assert cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR) is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_shape_detect.py tests/test_yolo_detect.py -v --import-mode=importlib`
Expected: `test_shape_detect.py` FAILS with `ModuleNotFoundError: No module named 'src.shape_detect'`;
`test_annotate_supports_png_format` FAILS with `TypeError: annotate() takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Add the `fmt` parameter to `yolo.py::annotate()`**

In `src/vision/yolo.py`, replace the existing `annotate` function (lines 82-96):

```python
def annotate(jpeg, dets, fmt=".jpg"):
    """Draw boxes + labels -> encoded image bytes in `fmt` (default JPEG,
    matching webcam_look's original convention; pass fmt=".png" for callers
    that need PNG, e.g. shape_detect's Gallery-save convention, which -- like
    every other image tool in this app -- persists PNGs). Returns the input
    on decode failure."""
    import cv2
    import numpy as np
    arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return jpeg
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(arr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        cv2.putText(arr, f"{d['label']} {int(d['confidence'] * 100)}%",
                    (int(x1), max(0, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 200, 0), 1)
    ok, buf = cv2.imencode(fmt, arr)
    return bytes(buf.tobytes()) if ok else jpeg
```

- [ ] **Step 4: Write `src/shape_detect.py`**

```python
"""Shape detection via a locally-run torchvision Mask R-CNN pipeline
(labeled, per-instance segmentation -- masks, not just boxes). Chosen over
webcam_look's existing ultralytics/YOLO pipeline specifically because
ultralytics is AGPL-3.0 (confirmed against its LICENSE file); torchvision is
BSD-3-Clause with no field-of-use restriction on its pretrained weights, so
this needs no license-acceptance gate the way src/face_swap.py's InsightFace
models do -- just a plain download-if-missing step. Masks are produced (and
returned in each detection) even though this module's own tool only reports
labels/boxes/positions in text -- a future swap_shape sub-project needs real
per-instance masks for a clean cutout, and building the right pipeline once
now avoids redoing this work later. See
docs/superpowers/specs/2026-08-15-shape-detection-design.md.

API verified directly against the installed torchvision==0.26.0 package
(site-packages, not web search / training-data memory) before writing this
module:
  - maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
    weights_backbone=None) -- passing weights= makes the function skip the
    separate ImageNet backbone download entirely (confirmed via source:
    `if weights is not None: weights_backbone = None`), so only ONE download
    happens (the full COCO-pretrained state dict).
  - Inference (eval mode, no_grad) on a list of one [C,H,W] float tensor in
    0-1 range (via torchvision.transforms.functional.to_tensor) returns a
    list of one dict: {"boxes": FloatTensor[N,4], "labels": Int64Tensor[N],
    "scores": FloatTensor[N], "masks": FloatTensor[N,1,H,W] in 0-1 range,
    threshold at 0.5 per torchvision's own docs}. Masks come back at the
    SAME (H, W) as the input tensor -- no resizing needed.
  - MaskRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"] is a 91-entry
    COCO category list (index 0 is "__background__"; a few interior indices
    are the literal string "N/A", COCO's original non-contiguous numbering)
    -- readable without triggering any download, since it's plain Enum
    metadata, not part of the state dict fetch.
  - The weights download (torch.hub.load_state_dict_from_url under the
    hood) is cached under torch.hub.get_dir() -- explicitly redirected to
    DATA_DIR/shape_detect_models via torch.hub.set_dir() before model
    construction, mirroring face_swap.py's _model_root() convention rather
    than leaving it to scatter into the user's global ~/.cache/torch.
"""
import io
import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

_model = None
_categories = None


def _model_root() -> str:
    from src.constants import DATA_DIR
    return os.path.join(DATA_DIR, "shape_detect_models")


def _get_model():
    global _model, _categories
    if _model is None:
        os.makedirs(_model_root(), exist_ok=True)
        torch.hub.set_dir(_model_root())
        from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        _categories = list(weights.meta["categories"])
        _model = maskrcnn_resnet50_fpn(weights=weights, weights_backbone=None)
        _model.eval()
    return _model


def _get_categories():
    if _categories is None:
        _get_model()  # populates _categories as a side effect
    return _categories


def _to_numpy(x):
    """Accepts either a real torch.Tensor (real model output) or a plain
    numpy array / list (test doubles) -- lets injected fakes skip
    constructing real torch tensors entirely."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _format_detections(raw, w, h, conf=0.4):
    """raw: list of (label, confidence, x1, y1, x2, y2, mask). Filter by
    conf, round the box, add grid position (reusing yolo.py's own
    _position() so the phrasing matches webcam_look's established language
    instead of inventing a second convention) and the boolean instance
    mask. Pure."""
    from src.vision.yolo import _position
    dets = []
    for label, c, x1, y1, x2, y2, mask in raw:
        if float(c) < conf:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dets.append({
            "label": label,
            "confidence": round(float(c), 2),
            "box": [round(x1), round(y1), round(x2), round(y2)],
            "position": _position(cx, cy, w, h),
            "mask": mask,
        })
    return dets


def detect(image_bytes, *, model=None, categories=None, conf=0.4):
    """Detect labeled subjects (people, animals, other objects) in
    image_bytes -> list[Detection]. Accepts any PIL-readable format (PNG/
    JPEG/WEBP/...), matching this app's other image tools -- no format
    restriction. Never raises for "nothing detected" (returns []); model
    construction/inference errors propagate to the caller, which applies
    the never-raises discipline at its own boundary (matching every other
    image tool's convention)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    tensor = to_tensor(img)

    m = model if model is not None else _get_model()
    cats = categories if categories is not None else _get_categories()

    with torch.no_grad():
        raw_out = m([tensor])[0]

    boxes = _to_numpy(raw_out["boxes"])
    labels = _to_numpy(raw_out["labels"])
    scores = _to_numpy(raw_out["scores"])
    masks = _to_numpy(raw_out["masks"])

    raw = []
    for i in range(len(boxes)):
        label_idx = int(labels[i])
        if label_idx <= 0 or label_idx >= len(cats):
            continue
        label = cats[label_idx]
        if not label or label == "N/A":
            continue
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        mask = masks[i, 0] >= 0.5
        raw.append((label, float(scores[i]), x1, y1, x2, y2, mask))

    return _format_detections(raw, w, h, conf=conf)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_shape_detect.py tests/test_yolo_detect.py -v --import-mode=importlib`
Expected: all passed (5 new in `test_shape_detect.py`, 1 new in `test_yolo_detect.py`, plus the
existing `test_annotate_returns_decodable_jpeg` still passing unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/shape_detect.py src/vision/yolo.py tests/test_shape_detect.py tests/test_yolo_detect.py
git commit -m "feat(shape-detect): core detection module (torchvision Mask R-CNN)"
```

---

### Task 3: `detect_shapes` tool + full registration

**Files:**
- Modify: `src/agent_tools/image_tools.py` (append `detect_shapes_tool`/`DetectShapesTool`;
  update module docstring)
- Modify: `src/agent_tools/__init__.py` (import, `TOOL_HANDLERS`, `TOOL_TAGS`)
- Modify: `src/tool_schemas.py` (`FUNCTION_TOOL_SCHEMAS` entry, after `face_swap`'s)
- Modify: `src/agent_loop.py` (`TOOL_SECTIONS` entry; `_DOMAIN_TOOL_MAP["desktop"]:306-309`)
- Modify: `src/tool_index.py` (`BUILTIN_TOOL_DESCRIPTIONS` entry)
- Modify: `src/tool_execution.py` (dispatcher owner-threading branch, ~line 890)
- Modify: `routes/chat_routes.py` (`can_generate_images` privilege gate, ~line 843-844)
- Modify: `src/tool_security.py` (`_PLAN_MODE_KNOWN_MUTATORS`, ~line 187)
- Test: `tests/test_detect_shapes_tool.py`
- Test: `tests/test_detect_shapes_registration.py`

**Interfaces:**
- Consumes: `shape_detect.detect(image_bytes, *, model=None, categories=None, conf=0.4) ->
  list[dict]` (Task 2); `src.vision.yolo.summarize(dets) -> str`,
  `src.vision.yolo.annotate(image_bytes, dets, fmt=".jpg") -> bytes` (Task 2 extended this with
  `fmt`); `_resolve_image_source(tool_name, id_field, path_field, id_value, path_value, owner,
  upload_resolver)`, `_image_result(output_message, result_bytes, saved)`,
  `_default_gallery_saver(image_bytes, owner, *, prompt=..., model=...)` (all pre-existing in
  `src/agent_tools/image_tools.py`).
- Produces: `detect_shapes_tool(content, ctx, *, detector=None, upload_resolver=None,
  gallery_saver=None) -> dict`; `DetectShapesTool` class wrapping it.

- [ ] **Step 1: Write the failing tool tests**

Create `tests/test_detect_shapes_tool.py`:

```python
"""detect_shapes_tool resolves a chat attachment or filesystem path/
filename, runs it through src.shape_detect, and returns a text summary +
annotated image via the established image_url convention. Never raises
into the agent loop, matching every other image tool's established
pattern. See docs/superpowers/specs/2026-08-15-shape-detection-design.md.
"""
import asyncio
import json

from src.agent_tools.image_tools import DetectShapesTool, detect_shapes_tool


def _fake_upload_resolver(found=True, path="/tmp/fake.png"):
    def resolver(upload_id, owner=None):
        if not found:
            return None
        return {"id": upload_id, "path": path, "name": "fake.png", "mime": "image/png"}
    return resolver


def _fake_gallery_saver(image_id="gallery-img-1", filename="abc123def456.png"):
    calls = []

    def saver(image_bytes, owner):
        calls.append((image_bytes, owner))
        return {"id": image_id, "filename": filename}

    saver.calls = calls
    return saver


_ONE_PERSON = [{"label": "person", "confidence": 0.92, "box": [1, 2, 3, 4], "position": "left"}]


def test_missing_attachment_id_and_image_path_returns_error():
    result = asyncio.run(detect_shapes_tool("{}", {"owner": "alice"}))
    assert "error" in result


def test_invalid_json_returns_error():
    result = asyncio.run(detect_shapes_tool("not json", {"owner": "alice"}))
    assert "error" in result


def test_unresolvable_attachment_returns_error():
    content = json.dumps({"attachment_id": "missing-id"})
    result = asyncio.run(detect_shapes_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=False),
        detector=lambda image_bytes: _ONE_PERSON,
    ))
    assert "error" in result


def test_successful_detection_returns_summary_and_short_url(tmp_path):
    real_file = tmp_path / "photo.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})
    gallery_saver = _fake_gallery_saver(image_id="gallery-img-1", filename="abc123def456.png")

    result = asyncio.run(detect_shapes_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        detector=lambda image_bytes: _ONE_PERSON,
        gallery_saver=gallery_saver,
    ))

    assert "error" not in result
    assert "person" in result["output"]
    assert result["image_url"] == "/api/generated-image/abc123def456.png"
    assert result["gallery_image_id"] == "gallery-img-1"
    assert gallery_saver.calls[0][1] == "alice"


def test_zero_detections_is_not_an_error(tmp_path):
    real_file = tmp_path / "photo.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})

    result = asyncio.run(detect_shapes_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        detector=lambda image_bytes: [],
        gallery_saver=_fake_gallery_saver(),
    ))

    assert "error" not in result
    assert "No recognizable objects detected" in result["output"]


def test_detector_failure_returns_error_not_raise(tmp_path):
    real_file = tmp_path / "photo.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})

    def failing_detector(image_bytes):
        raise RuntimeError("model download failed")

    result = asyncio.run(detect_shapes_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        detector=failing_detector,
        gallery_saver=_fake_gallery_saver(),
    ))

    assert "error" in result


def test_gallery_saver_receives_detect_shapes_prompt_and_model_via_default_saver(tmp_path, monkeypatch):
    import src.agent_tools.image_tools as image_tools

    captured = {}

    def fake_default_saver(image_bytes, owner, *, prompt="Background removed", model="remove_background"):
        captured["prompt"] = prompt
        captured["model"] = model
        return {"id": "gid", "filename": "f.png"}

    monkeypatch.setattr(image_tools, "_default_gallery_saver", fake_default_saver)

    real_file = tmp_path / "photo.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})

    asyncio.run(detect_shapes_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        detector=lambda image_bytes: _ONE_PERSON,
    ))

    assert captured["prompt"] == "Shapes detected"
    assert captured["model"] == "detect_shapes"


def test_gallery_save_failure_falls_back_to_inline_data_uri(tmp_path):
    real_file = tmp_path / "photo.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1"})

    def failing_saver(image_bytes, owner):
        raise RuntimeError("db unavailable")

    result = asyncio.run(detect_shapes_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        detector=lambda image_bytes: _ONE_PERSON,
        gallery_saver=failing_saver,
    ))

    assert result["image_url"].startswith("data:image/png;base64,")
    assert "gallery_image_id" not in result


def test_tool_class_delegates_to_module_function():
    tool = DetectShapesTool()
    result = asyncio.run(tool.execute("{}", {"owner": "alice"}))
    assert "error" in result
```

Create `tests/test_detect_shapes_registration.py` (mirrors `tests/test_face_swap_registration.py`
exactly, same structure, `detect_shapes` in place of `face_swap`):

```python
"""detect_shapes must be registered everywhere a builtin tool needs to be,
applying every lesson prior image-tool sub-projects' whole-branch reviews
found (dispatcher owner-threading, the can_generate_images privilege gate,
the plan-mode backstop) from the start. Mirrors
tests/test_face_swap_registration.py's structure. See
docs/superpowers/specs/2026-08-15-shape-detection-design.md."""
import asyncio
from pathlib import Path

_CHAT_ROUTES = Path(__file__).resolve().parent.parent / "routes" / "chat_routes.py"


def test_detect_shapes_registered_everywhere():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    from src.agent_loop import TOOL_SECTIONS, _DOMAIN_TOOL_MAP
    from src.tool_index import BUILTIN_TOOL_DESCRIPTIONS

    assert "detect_shapes" in TOOL_HANDLERS
    assert "detect_shapes" in TOOL_TAGS
    names = [(s.get("function") or {}).get("name") for s in FUNCTION_TOOL_SCHEMAS]
    assert "detect_shapes" in names
    assert "detect_shapes" in TOOL_SECTIONS
    assert "detect_shapes" in _DOMAIN_TOOL_MAP["desktop"]
    assert "detect_shapes" in BUILTIN_TOOL_DESCRIPTIONS


def test_detect_shapes_matches_generate_image_admin_gating():
    from src.tool_security import NON_ADMIN_BLOCKED_TOOLS
    generate_image_blocked = "generate_image" in NON_ADMIN_BLOCKED_TOOLS
    detect_shapes_blocked = "detect_shapes" in NON_ADMIN_BLOCKED_TOOLS
    assert detect_shapes_blocked == generate_image_blocked, (
        "detect_shapes's NON_ADMIN_BLOCKED_TOOLS membership must mirror "
        "generate_image's, not default to admin-only"
    )


def test_detect_shapes_blocked_when_can_generate_images_disabled():
    source = _CHAT_ROUTES.read_text(encoding="utf-8")
    assert 'if not _privs.get("can_generate_images", True):' in source
    idx = source.index('if not _privs.get("can_generate_images", True):')
    following = source[idx: idx + 300]
    assert "detect_shapes" in following, (
        "detect_shapes must be added to disabled_tools in the same "
        "can_generate_images privilege branch as generate_image/remove_background/"
        "edit_image_prompt/face_swap"
    )


def test_dispatcher_threads_owner_and_session_into_tool_ctx(monkeypatch):
    """The REAL dispatcher (execute_tool_block, not a mock of it) must
    thread owner= into detect_shapes's ctx -- otherwise the tool falls into
    the generic dynamic_handlers catch-all, which never threads owner, and
    resolve_upload denies every real (owned) attachment."""
    import src.agent_tools as agent_tools
    from src.agent_tools import ToolBlock
    from src.tool_execution import execute_tool_block

    seen = {}

    async def spy(content, ctx):
        seen["content"] = content
        seen["ctx"] = ctx
        return {"output": "detected", "exit_code": 0}

    monkeypatch.setitem(agent_tools.TOOL_HANDLERS, "detect_shapes", spy)

    desc, result = asyncio.run(execute_tool_block(
        ToolBlock("detect_shapes", '{"attachment_id": "up-1"}'),
        session_id="sess-1",
        owner="alice",
    ))

    assert result.get("exit_code") == 0
    assert seen["ctx"].get("owner") == "alice", (
        "detect_shapes's ctx lost the owner -- resolve_upload will deny every owned attachment"
    )
    assert seen["ctx"].get("session_id") == "sess-1"


def test_detect_shapes_in_plan_mode_known_mutators():
    """detect_shapes writes an annotated PNG to disk and inserts a Gallery
    DB row (best-effort) -- the same class of mutator as
    generate_image/remove_background/edit_image_prompt/face_swap, all
    members of _PLAN_MODE_KNOWN_MUTATORS."""
    import src.tool_security as ts
    assert "detect_shapes" in ts._PLAN_MODE_KNOWN_MUTATORS
    disabled = ts.plan_mode_disabled_tools()
    assert "detect_shapes" in disabled
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_detect_shapes_tool.py tests/test_detect_shapes_registration.py -v --import-mode=importlib`
Expected: `test_detect_shapes_tool.py` FAILS with `ImportError` (`detect_shapes_tool`/
`DetectShapesTool` don't exist yet); `test_detect_shapes_registration.py` FAILS on the first
assertion in `test_detect_shapes_registered_everywhere` (`"detect_shapes" in TOOL_HANDLERS`).

- [ ] **Step 3: Append the tool to `src/agent_tools/image_tools.py`**

First, replace the ENTIRE module docstring (lines 1-31 as they currently stand) — the current
text is already slightly stale (it still names the pre-rename `_resolve_face_swap_source` helper,
renamed to `_resolve_image_source` when `remove_background`/`edit_image_prompt` were extended
with path/filename support earlier this session) — with:

```python
"""The builtin image-editing tools that operate on an already-uploaded chat
image attachment: `remove_background` (strips the background via the bundled
U2Net ONNX model, src/bg_removal.py -- no rembg/transformers dependency),
`edit_image_prompt` (applies a natural-language edit via img2img on the
bundled sd-server, src/image_edit.py), `face_swap` (swaps a face from
one uploaded image into another via the bundled InsightFace pipeline,
src/face_swap.py -- the only one of the four that resolves TWO chat
attachments), and `detect_shapes` (detects labeled subjects -- people,
animals, other objects -- via the bundled torchvision Mask R-CNN pipeline,
src/shape_detect.py; read-only, never edits the source image). All four
accept their image input EITHER as a chat attachment id OR as a real
filesystem path/bare filename, via the shared _resolve_image_source /
_resolve_path_bytes helpers, confined to the same roots find_files already
searches (src/agent_tools/desktop_tools.py's FindFilesTool) -- this extends
an existing trust boundary to these tools, not a new capability class. All
four share three more helpers here -- _resolve_attachment_bytes (attachment
-> bytes), _image_result (result shaping) and _default_gallery_saver
(best-effort Gallery persistence).

All four return their result via the established image_url convention -- as
a SHORT /api/generated-image/<file> URL (like generate_image), falling back
to an inline data: URI only when the Gallery save failed. This is the first
builtin tool module to call
upload_handler.resolve_upload() directly; no existing accessor for the
app's UploadHandler singleton is reachable from a Tool's ctx, so this
constructs its own throwaway instance, mirroring
routes/document_helpers.py's existing precedent for the same reason (the
read path needs no cross-request state). NEVER raises into the agent --
every failure returns {"error": ...}, matching diagnose_equipment's
established pattern. See
docs/superpowers/specs/2026-08-12-image-editing-background-removal-design.md,
docs/superpowers/specs/2026-08-12-image-editing-prompt-edit-design.md,
docs/superpowers/specs/2026-08-13-image-editing-face-swap-design.md, and
docs/superpowers/specs/2026-08-15-shape-detection-design.md.
"""
```

Then append at the end of the file, after the existing `FaceSwapTool` class:

```python
async def detect_shapes_tool(content, ctx, *, detector=None, upload_resolver=None, gallery_saver=None):
    ctx = ctx or {}
    owner = ctx.get("owner")

    try:
        args = json.loads(content) if content and content.strip() else {}
        if not isinstance(args, dict):
            return {"error": "detect_shapes: arguments must be a JSON object"}
    except (ValueError, TypeError):
        return {"error": "detect_shapes: arguments must be valid JSON"}

    attachment_id = args.get("attachment_id")
    image_path = args.get("image_path")

    if upload_resolver is None:
        from src.constants import DATA_DIR, UPLOAD_DIR
        from src.upload_handler import UploadHandler
        upload_resolver = UploadHandler(DATA_DIR, UPLOAD_DIR).resolve_upload

    image_bytes, err = await _resolve_image_source(
        "detect_shapes", "attachment_id", "image_path", attachment_id, image_path, owner, upload_resolver)
    if err:
        return err

    if detector is None:
        from src.shape_detect import detect as detector

    try:
        dets = await asyncio.to_thread(detector, image_bytes)
    except Exception as e:
        return {"error": f"detect_shapes: {e}"}

    from src.vision.yolo import annotate, summarize
    out = summarize(dets)
    annotated = await asyncio.to_thread(annotate, image_bytes, dets, ".png")

    def _saver(image_bytes, owner):
        return _default_gallery_saver(image_bytes, owner, prompt="Shapes detected", model="detect_shapes")

    saver = gallery_saver or _saver
    try:
        saved = saver(annotated, owner)
    except Exception:
        logger.warning("detect_shapes: failed to save result to Gallery", exc_info=True)
        saved = None

    return _image_result(out, annotated, saved)


class DetectShapesTool:
    async def execute(self, content, ctx):
        return await detect_shapes_tool(content, ctx)
```

- [ ] **Step 4: Register in `src/agent_tools/__init__.py`**

Change the import (line 44):

```python
from .image_tools import RemoveBackgroundTool, EditImagePromptTool, FaceSwapTool, DetectShapesTool
```

Add to `TOOL_HANDLERS` (immediately after the `"face_swap"` line, ~96):

```python
    "detect_shapes": DetectShapesTool().execute,
```

Add to `TOOL_TAGS` (immediately after the `"face_swap"` entry, ~136):

```python
             "detect_shapes",
```

- [ ] **Step 5: Register in `src/tool_schemas.py`**

Immediately after the existing `face_swap` schema entry — it ends with:

```python
                    "target_image_path": {"type": "string", "description": "A filesystem path or bare filename for the target image (e.g. 'C:\\Users\\me\\Pictures\\group.jpg' or just 'group.jpg'). Use this OR target_image_id, not both."},
                },
            },
        },
    },
```

(note: `face_swap`'s schema has no `"required"` array — the "exactly one of id/path" constraint
can't be expressed that way and is enforced in the tool's own Python validation instead; don't
add one to `detect_shapes`'s schema either) — and immediately before the `ui_control` entry,
insert:

```python
    {
        "type": "function",
        "function": {
            "name": "detect_shapes",
            "description": "Detect labeled subjects (people, animals, and other objects) in a photo, returning what was found (label, confidence, rough position) plus the photo shown back with boxes and labels drawn on. The image can be given EITHER as an uploaded chat attachment id OR as a real filesystem path/filename (searched under the user's home directory and any configured extra tool roots) -- give exactly one of the two, never both. Any common image format works (PNG, JPEG, WEBP, etc).",
            "parameters": {
                "type": "object",
                "properties": {
                    "attachment_id": {"type": "string", "description": "The id of the uploaded image attachment. Use this OR image_path, not both."},
                    "image_path": {"type": "string", "description": "A filesystem path or bare filename for the image (e.g. 'C:\\Users\\me\\Pictures\\photo.jpg' or just 'photo.jpg'). Use this OR attachment_id, not both."},
                },
            },
        },
    },
```

- [ ] **Step 6: Register in `src/agent_loop.py`**

Immediately after the existing `"face_swap"` entry in `TOOL_SECTIONS` (ends `...inline in your
response.""",`), insert:

```python
    "detect_shapes": """```detect_shapes
{"attachment_id": "<id from an uploaded image>"}
```
Or by filesystem path/filename instead of an upload id:
```detect_shapes
{"image_path": "<full path or just a filename, e.g. photo.jpg>"}
```
Give exactly one of `attachment_id`/`image_path`. A bare filename is searched for under the user's home directory and any configured extra tool roots (same scope as find_files); a full path must resolve within those same folders. Any common image format works. Detect labeled subjects (people, animals, other objects) in the photo -- returns what was found plus the photo shown back with boxes/labels drawn on. Finding nothing is a normal result, not an error.""",
```

In `_DOMAIN_TOOL_MAP["desktop"]` (lines 306-309), add `"detect_shapes"` to the set:

```python
    "desktop": {"launch_app", "find_files", "list_windows", "control_window", "capture_screen",
                "webcam_look", "diagnose_equipment", "remove_background", "edit_image_prompt", "face_swap",
                "detect_shapes",
                "ingest_equipment_manual", "search_equipment_manual",
                "list_ui_elements", "click_element", "set_element_text", "mouse", "keyboard"},
```

- [ ] **Step 7: Register in `src/tool_index.py`**

Immediately after the existing `"face_swap"` entry in `BUILTIN_TOOL_DESCRIPTIONS`, insert:

```python
    "detect_shapes": "Detect labeled subjects (people, animals, objects) in a photo, returning what was found plus an annotated image. The image can be an uploaded chat attachment or a filesystem path/filename.",
```

- [ ] **Step 8: Register in `src/tool_execution.py`**

Immediately after the existing `elif tool == "face_swap":` branch (ends `or {"error": "face_swap:
execution failed", "exit_code": 1}`), insert:

```python
    elif tool == "detect_shapes":
        # Registry-dispatched (agent_tools.image_tools); owner threaded for the
        # exact same reason as remove_background/edit_image_prompt/face_swap
        # just above -- the tool resolves the caller's OWN chat attachment via
        # upload_handler.resolve_upload(), which denies any owned upload
        # record when called with owner=None.
        desc = f"detect_shapes: {content.split(chr(10))[0][:80]}"
        result = await _direct_fallback(tool, content, session_id=session_id, owner=owner) \
            or {"error": "detect_shapes: execution failed", "exit_code": 1}
```

- [ ] **Step 9: Register the privilege gate in `routes/chat_routes.py`**

Change (line 843-844):

```python
            if not _privs.get("can_generate_images", True):
                disabled_tools.update({"generate_image", "remove_background", "edit_image_prompt", "face_swap"})
```

to:

```python
            if not _privs.get("can_generate_images", True):
                disabled_tools.update({"generate_image", "remove_background", "edit_image_prompt", "face_swap", "detect_shapes"})
```

- [ ] **Step 10: Register the plan-mode backstop in `src/tool_security.py`**

Change (line 187):

```python
    "remove_background", "edit_image_prompt", "face_swap",
```

to:

```python
    "remove_background", "edit_image_prompt", "face_swap", "detect_shapes",
```

- [ ] **Step 11: Run tests to verify they pass**

Run: `pytest tests/test_detect_shapes_tool.py tests/test_detect_shapes_registration.py -v --import-mode=importlib`
Expected: all passed.

- [ ] **Step 12: Run the full image-tools + shape-detect regression slice**

Run: `pytest tests/test_shape_detect.py tests/test_yolo_detect.py tests/test_detect_shapes_tool.py tests/test_detect_shapes_registration.py tests/test_torchvision_packaging.py tests/test_face_swap_tool.py tests/test_remove_background_tool.py tests/test_edit_image_prompt_tool.py -v --import-mode=importlib`
Expected: all passed — confirms the shared helpers (`_resolve_image_source`, `_image_result`,
`_default_gallery_saver`) and `yolo.py`'s `annotate()`/`summarize()` still work correctly for
every existing caller after this task's changes.

- [ ] **Step 13: Commit**

```bash
git add src/agent_tools/image_tools.py src/agent_tools/__init__.py src/tool_schemas.py \
        src/agent_loop.py src/tool_index.py src/tool_execution.py routes/chat_routes.py \
        src/tool_security.py tests/test_detect_shapes_tool.py tests/test_detect_shapes_registration.py
git commit -m "feat(shape-detect): detect_shapes agent tool + full registration"
```

---

## After Task 3

Run the full test suite (`pytest tests/ -q --import-mode=importlib`) and confirm no new
failures against the pre-existing baseline (there are ~130 known pre-existing failures in this
dev environment unrelated to this work — compare counts, don't expect zero). Then follow
`superpowers:finishing-a-development-branch`.

A follow-up rebuild of `installer/Output/Assist-Setup.exe` (`.\build-installer.ps1`) is owed
once this plan is complete, same as every other feature this session — `detect_shapes` won't be
usable in the installed app until then.
