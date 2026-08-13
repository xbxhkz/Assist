# AI Image Editing — Sub-project 3: Face-Swap Tooling — Design Spec

## Context

This is the third and final sub-project in the AI Image Editing initiative (see
`docs/superpowers/specs/2026-08-12-image-editing-background-removal-design.md`). Sub-project 1
(Background Removal) and sub-project 2 (Natural-Language Editing) are both shipped on `dev`. This
one was deliberately deferred from the start, needing its own scope/guardrails conversation before
any design work, given face-swap technology's dual-use nature: legitimate uses (VFX compositing,
film dubbing/localization, historical reenactment, de-aging, privacy-preserving anonymization,
personal creative projects) sit alongside a strong real-world association with non-consensual
deepfakes, harassment, and impersonation.

**That conversation happened before any technical design, and settled two things a purely
technical brainstorm would have missed:**

1. **Assist is a publicly distributed application** (GitHub releases, an installer, a changelog),
   not a private script — so "I personally won't misuse this" doesn't by itself answer what
   guardrails the *shipped feature* needs, since it goes to everyone who installs Assist, not just
   its developer. The user's intent for their own use (personal projects, not nefarious) is taken
   at face value; the guardrails below exist because the feature ships to other people too.
2. **Live research found no cleanly-licensed pretrained face-swap model exists to bundle.**
   Checked against FaceFusion's own model-licensing documentation (the most complete public
   reference for this, since FaceFusion documents every mainstream option): InSwapper, BlendSwap,
   and SimSwap are explicitly non-commercial/research-only; HyperSwap is research-only (RAIL
   license); UniFace has no stated license at all, which legally means no permission is granted,
   not that it's unrestricted. This is a harder licensing problem than sub-project 1 hit (there
   was a clean alternative — U2Net — to fall back to for background removal; there is no equally
   capable, cleanly-licensed alternative here). **Decision: no face-swap model is bundled in
   Assist's installer or auto-fetched at build time.** Instead, the app provides a gated,
   in-app download: the model's actual license terms are shown, the user must explicitly accept
   them, and only then is the model fetched from its original source, under the user's own
   acceptance — not Assist's redistribution.

**Guardrails settled (both resolved during brainstorming, not left open):**

- **Provenance marking**: every face-swap output image gets embedded metadata identifying it as
  AI-face-edited (metadata only — no visible watermark, per the user's explicit choice). This
  doesn't restrict any legitimate use; it means a swapped image can't quietly pass as an unaltered
  photo if it ever leaves the user's machine.
- **Gated model licensing**: covered above.
- **Explicitly not attempted**: identity/consent verification of the people depicted in either
  image. That would be a substantially larger, likely infeasible undertaking — this v1's
  guardrails are provenance marking and licensing consent, not a verification system. This is a
  deliberate, stated boundary, not an oversight.

## Goal

A user supplies a source-face image and a target image and gets back the target image with the
source face swapped in — via chat or the Gallery editor — with the output carrying provenance
metadata, and with the underlying model obtained only through explicit, informed license
acceptance.

## Architecture

**A new module, `src/face_swap.py`**, mirroring `src/bg_removal.py`'s shape: a core function
taking source-face bytes and target-image bytes, returning the swapped result as PNG bytes with
embedded provenance metadata. Internally: runs InsightFace's face-detection/alignment step against
both images, then feeds the result into the InSwapper ONNX model for the swap itself (ONNX,
matching this app's established `onnxruntime` pattern from `bg_removal.py` — chosen over the other
three licensed-alike alternatives on technical merit: it's the most widely used and
highest-quality of the four, per live research).

**A separate small module (or an extension of this app's existing settings infrastructure)
handles gated model licensing**: checks whether the model is present locally AND its license has
been explicitly accepted (a persisted setting, mirroring how other admin-configurable settings
work in this app); if not, surfaces the model's actual license text and requires an explicit
accept action before fetching the model from its original source. No swap runs until both
conditions hold.

**Two entry points, both calling the same core function** — matching sub-projects 1 and 2's
established pattern:

1. A new builtin agent tool taking two chat attachment ids (source face, target image). Registered
   with the full lesson-learned checklist from sub-project 1's whole-branch review applied from
   day one: the 6 standard registration points, the dispatcher owner-threading branch, the
   `can_generate_images`-equivalent privilege gate, and the plan-mode mutator backstop — not
   discovered missing in review this time.
2. A Gallery editor addition, following the established single-image-editor-button pattern (the
   image currently open in the editor is the *target*) — since neither prior sub-project's Gallery
   button needed a second image, and building a full "browse and pick a second Gallery image" UI
   would be a meaningfully larger scope item than either prior sub-project's Gallery addition. The
   *source* face is supplied via a file upload at the point of action, not picked from existing
   Gallery images.

Output reuses the shared helpers `edit_image_prompt` already extracted in sub-project 2
(`_resolve_attachment_bytes`, `_image_result`) — this becomes the third tool through that code,
which is exactly the case those helpers were generalized to handle — and returns the short served
`/api/generated-image/{filename}` URL, not an inline data URI, matching the corrected pattern from
sub-project 2's whole-branch review.

## Data Flow

**Chat path**: user uploads a source-face image and a target image, asks to swap → the agent
calls the tool with both attachment ids → the tool resolves both via `resolve_upload()` (owner
threaded via the dispatcher branch) → checks the model-download+license-acceptance state; if not
met, returns a clear, actionable message pointing at where to accept the license (mirroring
`edit_image_prompt`'s "configure a default image model in Admin" message pattern) → runs the swap
via `asyncio.to_thread` (applied from the start, not discovered in review) → embeds provenance
metadata → best-effort Gallery save via the shared helpers → returns the short served URL.

**Gallery path**: user opens a Gallery image (the target) in the editor and uploads a source-face
image via the new action → same model/license check → same core function → result saved to
Gallery.

The license-acceptance flag is a persisted setting, checked by both entry points before any swap
proceeds — accepting it once (e.g., via a Settings panel) covers both.

## Error Handling

Matches this app's established never-raises convention for builtin tools: a missing/invalid
attachment id (now two, not one), an unresolvable attachment, the model not yet
downloaded/license not yet accepted, **no face detected in the source or target image** (a new
failure mode specific to this feature — face detection can genuinely find nothing usable),
inference failure, and non-fatal Gallery-save failure — all fail with a clear, specific message,
never an unhandled exception into the agent loop.

## Testing

**Backend:** unit tests for `src/face_swap.py`'s core function with a mocked detection+swap
pipeline — no real model required, matching `bg_removal.py`'s established test style. Tool-level
tests extend `edit_image_prompt_tool`'s established style (missing/invalid args, unresolvable
attachment, model-call failure, Gallery-save-failure fallback), plus dedicated tests for the
license-gate check (not-yet-accepted → clear error, never a silent bypass) and the no-face-detected
case. Registration-parity test applies the full lesson-learned checklist (dispatcher
owner-threading, privilege gate, plan-mode backstop) from day one — the way sub-project 2's plan
was written after learning from sub-project 1's review, not discovered missing again.

**Frontend:** Gallery editor addition needs source-presence tests matching the established style
for other Gallery editor actions; the license-acceptance UI (wherever it lives — likely a Settings
panel addition) needs its own tests confirming the swap path is genuinely blocked until acceptance
is recorded, not just visually gated.

Manual GUI + end-to-end verification (does a real swap actually look right, does the license-gate
flow actually block the swap when not accepted and unblock it when accepted, does the metadata
provenance tag actually survive being written and later read back) is owed by the user, same as
every other feature this session that touches real model inference.

## Out of Scope

- Video (deferred; a still-images-only decision made during brainstorming — this hardware's 6GB
  VRAM budget and this app's existing image-processing infrastructure both favor a per-image,
  no-training pretrained-model approach over a video/training-based one like DeepFaceLab).
- Any tunable parameters exposed to the model or user (matching sub-project 2's "no knobs"
  precedent for v1).
- Any bundled celebrity/public-figure face catalog or face library — the tool only ever operates
  on images the user explicitly supplies, nothing pre-loaded or searchable.
- Any auto-publish/sharing integration — stays local-only, consistent with every other feature in
  this app.
- Identity or consent verification of the people depicted in either image (see Context) — a
  deliberate boundary, not a gap to fill later without a fresh conversation about feasibility.
- Visible watermarking of output (the user chose metadata-only provenance marking for v1).
- Any model besides InSwapper — if quality or licensing terms change this calculus later,
  evaluating alternatives is a follow-up, not part of this spec.
