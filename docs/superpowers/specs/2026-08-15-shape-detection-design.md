# Shape Detection — Design Spec

## Context

The user asked for the ability to "detect shapes in photos like of people, animals, etc to be
able to swap them." That request bundles two distinct capabilities: detecting labeled subjects in
a photo, and swapping a detected subject between two photos. Per this session's own established
pattern for multi-part image-editing work (background removal / natural-language editing /
face-swap were each their own sub-project), this was split the same way: **this spec covers
detection only.** Swapping is a deliberate follow-up, designed fresh once detection exists to
build on, not designed blind alongside it.

**Swap mechanic, decided during brainstorming:** cut-and-paste compositing (segment the chosen
subject, cut it out, paste it into the other photo), not a generative/inpainting identity swap.
This matters for this spec because it means the detector needs to produce genuine per-instance
**masks**, not just bounding boxes — a box-only detector (e.g. `webcam_look`'s existing
`yolov8n.pt`) would let a future swap tool paste a rectangle including background, not a clean
cutout of the subject itself.

**Model licensing, live-verified before any design (mirroring face-swap's own research
rigor):** the obvious reuse — extending `webcam_look`'s existing `ultralytics`/YOLO pipeline with
a segmentation variant (`yolov8n-seg.pt`) — was rejected. Ultralytics' YOLO package and pretrained
weights are **AGPL-3.0** (confirmed against the actual LICENSE file and Ultralytics' own licensing
FAQ), which requires publicly releasing the complete corresponding source of any distributed
application that embeds it, or purchasing Ultralytics' Enterprise License. Assist is a publicly
distributed application (installer, GitHub releases), and `ultralytics`/`yolov8n.pt`'s existing
use in `webcam_look` does not appear to have had this addressed when it shipped — a pre-existing
gap this spec does not attempt to retroactively fix, but declines to deepen.

**Model chosen instead: `torchvision.models.detection.maskrcnn_resnet50_fpn`**, COCO-pretrained
(`MaskRCNN_ResNet50_FPN_Weights.DEFAULT`, ~170MB). `torchvision` is **BSD-3-Clause** (confirmed
against its LICENSE file), fully permissive, no source-disclosure obligation, and no separate
license restriction on the pretrained weights. It is also already present in this dev
environment's dependency tree (pulled in transitively via `ultralytics`), so this is not really a
new dependency family — it's the first-party PyTorch ecosystem package, which packages far more
reliably into a frozen Windows/PyInstaller build than the alternatives considered:

- **Detectron2** (Apache-2.0): same underlying approach (Mask R-CNN), but with a historically
  rough Windows/PyInstaller packaging story — no benefit over torchvision's built-in
  implementation for this use case.
- **Segment Anything (SAM, Apache-2.0)**: also permissive, but class-agnostic — it segments
  "things" without labeling what they are, so it would need a separate classifier bolted on to
  answer "is this a person or a dog," more moving parts for no benefit here.

**Model delivery:** downloaded on first use, not bundled unconditionally in the installer.
Because the license is permissive, this is a pure installer-size/UX tradeoff, not a consent
question — 170MB is meaningfully larger than `yolov8n.pt`'s 6MB, so it follows the download-on-
first-use pattern already established by `face_swap`'s (much larger, license-gated) models,
minus the license-acceptance gate itself, which doesn't apply here.

## Goal

A user (or the agent on their behalf) points `detect_shapes` at a photo — via chat attachment or a
real filesystem path/filename, matching the pattern this session already shipped for
`face_swap`/`remove_background`/`edit_image_prompt` — and gets back a text description of what was
found (people, animals, objects — label, confidence, rough position) plus the photo shown back
with boxes and labels drawn on it. The underlying detection also produces real per-instance
segmentation masks, not exposed through the tool's own text output, but structured so a future
`swap_shape` sub-project can reuse the same detection pipeline directly instead of redoing this
work.

## Architecture

**A new module, `src/shape_detect.py`**, mirroring `src/face_swap.py`/`src/bg_removal.py`'s
shape: a lazy-loaded model singleton wrapping `maskrcnn_resnet50_fpn`, with the same injectable-
model test seam used everywhere else in this app (`model=None` parameter, real construction
deferred behind it). Core function: image bytes in, a list of `Detection`s out (`label`,
`confidence`, `box`, `mask`, grid `position`) — `position` reuses `src/vision/yolo.py`'s existing
`_position()` helper so the phrasing ("left", "center-right", etc.) matches `webcam_look`'s
established language instead of inventing a second convention. Detections are numbered per label
(`person #1`, `person #2`, `dog #1`, ...) — not needed by this tool's own output, but the exact
shape a future `swap_shape` needs for disambiguating "swap the 2nd person," so it exists now
rather than getting bolted on as a breaking change later.

**A new builtin agent tool, `detect_shapes`**, registered with the full checklist this session has
now applied three times running: the standard registration points, dispatcher owner-threading,
the `can_generate_images`-equivalent privilege gate (matching `remove_background`/
`edit_image_prompt`/`face_swap` — this tool only ever looks at a photo it was already given, no
live camera access, so `webcam_look`'s stricter admin-only gate doesn't apply), and the plan-mode
mutator backstop.

Input resolution reuses `_resolve_image_source`/`_resolve_path_bytes` (`src/agent_tools/
image_tools.py`, shipped earlier this session) unchanged — `detect_shapes` accepts `image_id` or
`image_path` (single-image field naming, matching `remove_background`'s `attachment_id`/
`image_path`, not `face_swap`'s dual-image naming) from day one.

Output reuses the established `_image_result` best-effort-Gallery-save + short-URL convention (the
same one every other image tool in this app uses) for the annotated image, and a text summary
built the same way `src/vision/yolo.py`'s existing `summarize()` already does for `webcam_look`
(grouped by label, with counts/confidences/positions) — reused directly where possible rather than
reimplemented, since the grouping/formatting logic doesn't differ between a live webcam frame and
an uploaded photo.

## Data Flow

Agent calls `detect_shapes` with `image_id` or `image_path` → resolved to bytes via the shared
resolver → `shape_detect.detect(image_bytes)` runs Mask R-CNN inference off the event loop
(`asyncio.to_thread`, matching every other model-inference tool in this app) → returns numbered,
labeled detections with masks → the tool builds a text summary and an annotated image (boxes +
labels drawn via the same OpenCV drawing approach `src/vision/yolo.py::annotate()` already uses)
→ best-effort Gallery save → short served URL (or inline data URI fallback on save failure) →
tool result.

First real use triggers the model download if the weights aren't already cached locally (mirrors
`face_swap`'s `_ensure_models_available`-style gate, minus the license-acceptance check — this is
a plain download-if-missing step, since the weights carry no licensing restriction to gate
behind).

## Error Handling

Matches this app's established never-raises convention for builtin tools:

- Missing/invalid `image_id`/`image_path`, or both/neither given — clear `{"error": ...}`.
- Unresolvable attachment, path outside allowed roots, bare-filename zero/multiple matches,
  oversized file — same errors `_resolve_image_source`/`_resolve_path_bytes` already produce for
  the other image tools; nothing new to build here.
- Corrupt/unreadable image bytes — caught, clean error, never an unhandled exception.
- Model not yet downloaded and the download fails (no network, etc.) — clear, actionable error.
- **Zero detections found is *not* an error** — unlike `face_swap`'s `NoFaceDetectedError` (where
  "no face" blocks a downstream swap from happening at all), a detector legitimately reporting
  "nothing recognizable here" is a valid, useful answer, not a failure. Returns a normal result
  with a summary like "No recognizable people, animals, or objects detected," not an exception.
- Inference failure (any other model error) — caught, clean error.
- Non-fatal Gallery-save failure — falls back to the inline data URI, matching every other image
  tool.

## Testing

**Backend:** unit tests for `src/shape_detect.py`'s core `detect()` function with an injected fake
model returning canned detections — no real torch/torchvision required to run the suite, matching
`bg_removal.py`/`face_swap.py`'s established test style. Tool-level tests for `detect_shapes_tool`
cover missing/invalid args, unresolvable attachment/path (delegating to the already-tested shared
resolver, not re-testing its internals), zero-detections-is-not-an-error, model-download-failure,
inference-failure, and Gallery-save-failure fallback — mirroring `face_swap_tool`'s/
`remove_background_tool`'s existing test shapes. Registration-parity test applies the same
checklist (dispatcher owner-threading, privilege gate, plan-mode backstop) from day one.

Manual GUI + end-to-end verification (does a real photo actually produce sensible detections, does
the annotated image look right, does the first-use download actually work) is owed by the user,
same as every other feature this session that touches real model inference.

## Out of Scope

- **Swapping itself** — a deliberate follow-up sub-project, designed once this detection
  foundation exists, not designed blind alongside it.
- Video — still images only, consistent with every other image tool in this app.
- Any tunable parameters exposed to the model or user (confidence threshold, max detections) —
  reuses `src/vision/yolo.py`'s existing default (0.4 confidence) rather than introducing new
  knobs, matching this app's established "no tunable params" precedent for v1 image tools.
- Bundling the model weights in the installer — download-on-first-use only (see Context).
- Any consent/identity verification of people depicted in a photo — same deliberate boundary
  `face_swap`'s spec already drew, not revisited here.
- Retroactively addressing `webcam_look`'s existing `ultralytics`/AGPL-3.0 exposure — a real,
  separate question flagged to the user during brainstorming, explicitly left for a decision
  outside this spec's scope.
