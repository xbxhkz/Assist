# AI Image Editing — Sub-project 1: Background Removal — Design Spec

## Context

The user requested a broader "AI image editing" capability: natural-language add/change/remove
edits, background removal, and face-swap tooling (InsightFace/DeepFaceLab/FaceFusion). Live-code
research at brainstorming time found these are three genuinely independent subsystems with very
different feasibility and risk profiles, decomposed here the same way prior multi-part
initiatives this project has shipped were split:

- **Background removal (this spec)** — smallest lift, most implementation-ready. `onnxruntime` is
  already a bundled dependency (pulled in transitively by FastEmbed), so a small background-removal
  ONNX model can be bundled as a build asset and run directly, with no live-server feasibility
  question and no new heavy runtime dependencies. `rembg` (the obvious off-the-shelf library) is a
  dead end on this platform: the app's own Cookbook UI already marks it `winUnsupported`
  (`static/js/cookbook.js`), and the frozen Windows build has no `pip` available at runtime
  anyway (`routes/shell_routes.py`'s own comment: "pip simply isn't bundled there"). This spec
  sidesteps `rembg` entirely by loading a bundled ONNX model directly via `onnxruntime`, mirroring
  the exact pattern already used for the webcam tool's `yolov8n.pt` (fetched at build time via a
  `scripts/fetch_*.py` script, collected into the frozen build via `Assist.spec`).
- **Natural-language image editing (deferred, sub-project 2)** — the bundled `sd-server` binary's
  strings show `/sdapi/v1/img2img`/`init_image`/`mask`/`strength` fields exist, but this project
  already has a direct cautionary precedent: a prior ControlNet sub-project found the binary's
  `control_image` field was silently ignored in practice (output identical regardless of input) —
  string presence was not proof of function. This deferred sub-project needs its own live
  GO/NO-GO feasibility spike (mirroring ControlNet's own Task 1) before any real design work,
  which is out of scope for this spec.
- **Face-swap tooling (deferred, sub-project 3)** — zero existing infrastructure (no catalog
  entry, no feature code; `insightface` appears only in a raw pip-install allowlist). The biggest
  lift of the three, and the one that needs its own explicit conversation about scope and any
  guardrails given the technology's dual-use nature (legitimate creative/VFX/dubbing use
  alongside its association with non-consensual deepfakes) before design begins. Out of scope for
  this spec.

## Goal

A user can remove the background from an image two ways — asking the agent in chat ("remove the
background from this image"), or clicking a button in the Gallery editor — and get back a
transparent-background PNG, saved to the Gallery and, when triggered from chat, shown inline in
that response too.

## Architecture

**A new module, `src/bg_removal.py`**, with one public function:
`remove_background(image_bytes: bytes) -> bytes` — returns RGBA PNG bytes with the background
made transparent. Internally: lazy-loads a `u2net.onnx` model via `onnxruntime.InferenceSession`
(cached after first load, matching this app's established lazy-singleton pattern for other
models), decodes the input via Pillow (already bundled), resizes/normalizes to the model's
expected input shape, runs inference to get a saliency mask, resizes the mask back to the
original image's dimensions, and composites it as the alpha channel of the original image using
Pillow + numpy (both already bundled). No new runtime dependencies.

**Model bundling** mirrors the established `scripts/fetch_*.py` pattern exactly
(`fetch_embedding_model.py`, `fetch_llama_server.py`, `fetch_sd_server.py`): a new
`scripts/fetch_bg_removal_model.py` downloads U2Net's ONNX export (Apache-2.0 licensed — chosen
specifically over alternatives like BRIA RMBG, which carries a non-commercial license, since
Assist is publicly distributed via GitHub releases) into `build_assets/bg_removal/u2net.onnx`,
collected into the frozen build via `Assist.spec`, exactly like `yolov8n.pt` already is. The
exact download URL and a checksum are confirmed against the live, current hosting location at
plan-writing time, not guessed here.

**Two entry points, one shared function:**

1. **A new builtin agent tool, `remove_background`**, taking a reference to an already-uploaded
   chat image attachment. Reuses `src/upload_handler.py`'s existing `resolve_upload()` — the same
   mechanism vision-analysis tools already use to turn a chat attachment id into image bytes.
   Registered as a new builtin tool, which this codebase has a documented gotcha for: adding one
   needs **7 registration points, not the 5 a first pass usually finds** (bit a prior sub-project,
   webcam detection, before it was caught in review) — the plan calls out the exact 7 explicitly
   rather than leaving this to be rediscovered.
2. **A new button in the Gallery editor**, calling a new route (naming matches the Gallery's
   existing `harmonize_image`/`inpaint_proxy` convention, e.g. `POST
   /api/gallery/{image_id}/remove-background`) that calls `src/bg_removal.py` directly on an
   already-stored Gallery image.

Both entry points call the exact same `remove_background()` function — neither reimplements the
model-loading or image-processing logic.

## Data Flow

**Chat path**: user uploads an image and asks to remove its background → the agent calls the
`remove_background` tool with the attachment's id → the tool resolves the attachment to bytes via
`upload_handler.resolve_upload()` → calls `src/bg_removal.py`'s `remove_background()` → the
result is saved as a new Gallery image AND returned inline in the tool's result via this app's
existing `image_url` convention (the same mechanism `generate_image`-style tools already use to
render an image inline in the chat stream).

**Gallery path**: user clicks the new button on an existing Gallery image → the route loads that
image's bytes from wherever the Gallery already stores them → calls `src/bg_removal.py`'s
`remove_background()` → the result is saved as a new Gallery image.

## Error Handling

Matches this app's established "never-raises" convention for builtin tools: a non-image upload,
a corrupt/unreadable file, or a missing model asset (e.g. a dev environment where the build-time
fetch script was never run) all fail with a clear, specific error message — never an unhandled
crash. The Gallery route mirrors the same discipline at the HTTP layer (a clean 4xx/5xx with a
message, not a stack trace).

## Testing

**Backend:** unit tests for `src/bg_removal.py`'s pre/post-processing pipeline with the
`onnxruntime.InferenceSession` mocked — the real `u2net.onnx` file will not exist in the dev
repository, only after the build-time fetch script runs, so no test may depend on the actual
model file being present. Tests for the hostile-input paths (non-image bytes, truncated/corrupt
image data) proving the never-raises guarantee. Tool-level and route-level tests with
`remove_background()` itself mocked, matching this app's established test style for builtin
tools and Gallery routes.

**Frontend:** source-presence tests for the new Gallery button, matching the established style
for other Gallery editor actions.

Manual GUI + end-to-end verification (does a real image actually get its background removed
correctly, does the chat path really show the image inline, does the Gallery button really save
a new image) is owed by the user, same as every other feature this session that touches real
model inference or live UI interaction.

## Out of Scope

- Natural-language image editing (add/change/remove via prompt) — deferred to its own sub-project
  2, which needs a live feasibility spike before design begins (see Context).
- Face-swap tooling (InsightFace/DeepFaceLab/FaceFusion) — deferred to its own sub-project 3,
  which needs its own scope/guardrails conversation before design begins (see Context).
- Any editing beyond background removal (e.g. background replacement with a new image, edge
  refinement controls, batch processing multiple images at once) — v1 is exactly "remove the
  background, get a transparent PNG back," nothing more.
- Model choice beyond U2Net — if quality turns out insufficient in manual testing, evaluating a
  different model is a follow-up, not part of this spec.
