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
from PIL import Image

_model = None
_categories = None


def _model_root() -> str:
    from src.constants import DATA_DIR
    return os.path.join(DATA_DIR, "shape_detect_models")


def _get_model():
    global _model, _categories
    if _model is None:
        import torch
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
    mask. Also numbers each detection per label (person #1, person #2, dog
    #1, ...) via a running per-label counter over the already-ordered `raw`
    list -- not used by this module's own text output, but the exact shape
    a future swap_shape sub-project needs to disambiguate "swap the 2nd
    person" without a later breaking change (see spec's Architecture
    section). Pure."""
    from src.vision.yolo import _position
    dets = []
    counts = {}
    for label, c, x1, y1, x2, y2, mask in raw:
        if float(c) < conf:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        counts[label] = counts.get(label, 0) + 1
        dets.append({
            "label": label,
            "index": counts[label],
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
    image tool's convention).

    torch / torchvision.transforms.functional.to_tensor are imported here,
    lazily, rather than at module scope: importing torch costs ~2.3s, and
    this function is the ONLY code path (via detect_shapes_tool's
    asyncio.to_thread(detector, image_bytes) call) where that cost is
    supposed to land -- off the ASGI event loop, inside the thread pool.
    A module-scope `import torch` would instead pay that cost synchronously
    on the event loop the first time src.shape_detect is imported (e.g. by
    detect_shapes_tool's own `from src.shape_detect import detect as
    detector` line, which runs BEFORE the to_thread call), stalling every
    concurrent request (including SSE chat streams) for ~2.5s on first use.
    Mirrors src/face_swap.py's/src/bg_removal.py's own lazy-import-heavy-
    deps-inside-the-getter convention for the same reason."""
    import torch
    from torchvision.transforms.functional import to_tensor

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
