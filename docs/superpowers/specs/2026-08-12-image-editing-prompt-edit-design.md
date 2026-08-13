# AI Image Editing — Sub-project 2: Natural-Language Image Editing — Design Spec

## Context

This is the second of three AI Image Editing sub-projects (see the original decomposition in
`docs/superpowers/specs/2026-08-12-image-editing-background-removal-design.md`). Sub-project 1,
Background Removal, is shipped on `dev`. Sub-project 3, face-swap tooling, remains deferred
pending its own dedicated scope/guardrails conversation given the technology's dual-use nature.

**This sub-project required its own live feasibility spike before design work, per sub-project
1's own Context section**, which flagged a direct cautionary precedent: a prior ControlNet
sub-project found the bundled `sd-server` binary accepted a `control_image` field without error
but silently ignored it — output was identical regardless of input. String/API-surface presence
was not proof the feature worked end-to-end.

**The spike's verdict was GO, with strong empirical evidence — the opposite outcome from
ControlNet.** A live test against the bundled `sd-server.exe` (stable-diffusion.cpp, pinned build
`bb84971`, the same commit ControlNet was tested against) confirmed:
- Two `img2img` calls with identical prompt/seed/strength but different `init_image` inputs (a
  solid blue vs. solid green seed image) produced outputs differing by a mean pixel difference of
  80.36 (0-255 scale) — as different as two unrelated photos — while a determinism control
  (identical repeated calls) produced a difference of exactly 0.0000, ruling out random noise as
  the explanation.
- A low-strength (0.15) retention test showed each output's mean color closely tracked its own
  seed image's color (near-identical to input), confirming graded, not binary, conditioning.
- A mask-region test (solid blue image, white mask over a center square, prompt requesting an
  orange circle) showed the masked region was repainted (mean RGB shifted to orange) while the
  unmasked border was preserved almost exactly (mean pixel difference 2.72) — correct
  masked-inpainting behavior.
- Server-side logs confirmed every `img2img` call took a distinct `IMG2IMG` code path (not
  `TXT2IMG`) and that `denoising_strength` correctly drove the internal step count
  (`t_enc = strength × steps`) in every case.

**A second, independently important finding from the spike**: this codebase already has two
separate, unconnected image-editing subsystems. The Gallery UI's "harmonize" and "inpaint"
buttons call `routes/gallery/gallery_routes.py`, which targets `scripts/diffusion_server.py` — a
separate Python/diffusers server tied to the "Cookbook" feature (per its own test names, e.g.
`test_cookbook_cpu_only_serve.py`, `test_cookbook_remote_windows_diffusers.py`, this looks like
infrastructure for custom/remote/CPU-only workflows, not the app's default always-on backend).
Neither of those routes is wired to chat, and this spike did not test whether that path actually
works. Separately, the bundled `sd-server.exe` — the same binary already running for normal image
generation via `src/ai_interaction.py`'s `/v1/images/generations` calls — has its own working
`/sdapi/v1/img2img` endpoint (now proven above) that nothing in the app currently calls at all.

**Decision (made during brainstorming, not left open)**: build against `sd-server.exe` directly,
not the Cookbook/`diffusion_server.py` path. It's the same binary already treated as this app's
default image backend, requires no separate service, and is now proven to work. The existing
Gallery harmonize/inpaint buttons are left untouched — a separate, already-existing feature with
its own (untested by this spike) backend, out of scope here.

**The existing chat-facing `edit_image` tool** (`src/tools/image.py`, confirmed broken in
sub-project 1's own research — it proxies to `/api/gallery/{action}` routes that mostly don't
exist) remains untouched and out of scope, exactly as sub-project 1 left it. This sub-project adds
a new, separately-named tool rather than trying to fix or reuse it.

## Goal

A user uploads an image in chat and describes an edit in natural language ("add a red hat", "make
the sky sunset-colored", "remove the car in the background") and gets back an edited image, shown
inline in that response and saved to the Gallery.

## Architecture

**A new module, `src/image_edit.py`**, mirroring `src/bg_removal.py`'s shape, with one public
function: `edit_image(image_bytes: bytes, prompt: str, *, strength=0.6) -> bytes`. Internally:
ensures the default image model is being served (reusing the existing `ensure_image_served`
auto-routing logic already used for `generate_image` — no separate service, no new model to
bundle), then POSTs to the already-running `sd-server.exe`'s `/sdapi/v1/img2img` endpoint with the
input image as `init_images[0]`, the user's prompt, and `denoising_strength=0.6` (the balance
point the spike validated — a real, visible edit that still resembles the original), `steps=20`
(matching the spike's own successful test configuration). Returns the resulting PNG bytes.
`strength` is a fixed default per the brainstorming decision below — not exposed as a tunable
parameter to the model or user in v1.

**A new builtin agent tool, `edit_image_prompt`** — deliberately named to avoid any collision or
confusion with the existing, broken, out-of-scope `edit_image` tool. Takes `attachment_id` +
`prompt`, resolves the chat attachment the same way `remove_background` does
(`upload_handler.resolve_upload(attachment_id, owner=owner)`), reads the bytes, calls
`edit_image()` off the event loop via `asyncio.to_thread` (applying sub-project 1's whole-branch
review finding from day one this time, not after the fact — synchronous ONNX/diffusion inference
must never run directly on the event loop), and returns the result.

**Registration applies every lesson sub-project 1's whole-branch review surfaced, from the start**
rather than risking a repeat of the same gaps:
- The 6 standard builtin-tool registration points (import + `TOOL_HANDLERS` + `TOOL_TAGS` in
  `src/agent_tools/__init__.py`; `FUNCTION_TOOL_SCHEMAS` in `src/tool_schemas.py`; `TOOL_SECTIONS`
  + `_DOMAIN_TOOL_MAP` in `src/agent_loop.py`; `BUILTIN_TOOL_DESCRIPTIONS` in `src/tool_index.py`).
- **The `ctx["owner"]` dispatcher branch in `src/tool_execution.py`** — `remove_background` was
  originally missing this (it silently fell into a generic catch-all that never threads `owner`,
  making the tool fail-closed for every real user in any auth-enabled install, undetected by 17
  passing tests because none exercised the real dispatcher). `edit_image_prompt` needs the same
  explicit dispatch branch, threading `owner=`/`session_id=`, mirroring the pattern used for other
  owner-dependent tools — written from the start, with a dispatcher-level test (not just a
  callee-level test) proving it, exactly like the fix that closed this gap for `remove_background`.
- **The `can_generate_images` privilege gate in `routes/chat_routes.py`**, mirroring
  `generate_image`'s own gating exactly — this is a generation-adjacent capability, so it should
  be gated the same way from day one, not discovered missing in a later review.
- **The `_PLAN_MODE_KNOWN_MUTATORS` entry in `src/tool_security.py`** — this tool writes a PNG to
  disk and inserts a DB row, the same class of mutator `generate_image`/`edit_image`/
  `remove_background` already are.
- `NON_ADMIN_BLOCKED_TOOLS` membership mirrors whatever `generate_image`'s actual membership is at
  implementation time (confirmed by `remove_background`'s own plan: `generate_image` is NOT a
  member as of that plan's writing) — verified again at plan-writing time for this sub-project,
  not assumed to still hold.

## Data Flow

User uploads an image in chat and describes an edit → the agent calls `edit_image_prompt` with
`attachment_id` + `prompt` → the tool resolves the attachment via `resolve_upload(attachment_id,
owner=owner)` (with `owner` correctly threaded via the dispatcher branch above) → reads the image
bytes and calls `edit_image()` via `asyncio.to_thread` → `edit_image()` ensures the default image
model is being served, then calls `sd-server.exe`'s `/sdapi/v1/img2img` → returns the edited PNG
bytes → a best-effort Gallery save, **reusing `remove_background`'s existing
`_default_gallery_saver` helper** (extended or parameterized as needed for this tool's different
`prompt`/`model` field values, rather than duplicating near-identical logic) → the tool returns
the short served `/api/generated-image/{filename}` URL as `image_url` (the corrected pattern from
sub-project 1's whole-branch review, applied from the start), falling back to the full base64
data URI only if the Gallery save itself fails.

## Error Handling

Matches this app's established never-raises convention for builtin tools, identical in spirit to
`remove_background`: a missing/invalid `attachment_id`, an unresolvable attachment, the default
image model being unavailable or failing to serve, the `sd-server` request itself failing or
timing out, or a non-image upload — all fail with a clear, specific error message returned as
`{"error": ...}`, never an unhandled exception into the agent loop. Gallery-save failure remains
non-fatal, matching `remove_background`'s corrected best-effort behavior.

## Testing

**Backend:** unit tests for `edit_image()` with the HTTP call to `sd-server` mocked — no real
model or GPU required, matching how `bg_removal.py`'s tests injected a fake ONNX session rather
than depending on the real model file. Tool-level tests mirror `remove_background`'s established
style (missing/invalid args, unresolvable attachment, model-call failure, Gallery-save-failure
fallback to data URI). **The registration-parity test is written to check the full
lesson-learned list from day one** — the dispatcher's owner-threading branch (via a real,
non-mocked dispatch call, not just asserting the callee reads `ctx["owner"]` correctly), the
`can_generate_images` privilege gate, and the `_PLAN_MODE_KNOWN_MUTATORS` entry — as first-class
assertions, not just the original 6 points sub-project 1 started with before its review found the
gaps.

**Frontend:** none required — this is a chat-only tool with no new UI surface (mirrors
`remove_background`'s chat-tool path; no Gallery button is added).

Manual GUI + end-to-end verification (does a real edit actually look like what was asked for, is
edit quality acceptable, does the inline image really render in chat) is owed by the user, same as
every other feature this session that touches real model inference.

## Out of Scope

- Mask/region-targeted editing from a text description (e.g. "just the hat") — deferred; the user
  chose to ship whole-image editing first and revisit region-targeting later. Would need its own
  feasibility check for whether automatic region-inference from text is reliable.
- A tunable `strength` parameter — deferred; v1 uses one fixed default (0.6) validated by the
  spike, not exposed to the model or user.
- The Gallery's existing "harmonize"/"inpaint" buttons and their `diffusion_server.py`/Cookbook
  backend — untouched, separate feature, not confirmed broken or working by this spike. Could
  become its own future sub-project if ever found broken.
- Face-swap tooling — still sub-project 3, still needs its own scope/guardrails conversation
  before design begins.
- Batching multiple edits in one call, multi-turn iterative refinement of the same image, or any
  editing beyond one whole-image edit per tool call.
- Fixing or reusing the existing broken `edit_image` chat tool — left untouched, exactly as
  sub-project 1 left it.
