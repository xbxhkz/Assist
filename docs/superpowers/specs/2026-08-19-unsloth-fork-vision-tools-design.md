# Unsloth Fork — Sub-project 1: Vision Tools — Design Spec

## Context

This is the first sub-project of an initiative to fork
[Unsloth](https://github.com/unslothai/unsloth/) and add capabilities from Assist that Unsloth
lacks. The scope of that initiative was established by investigation, not assumption, and two
findings reshaped it substantially:

**Unsloth is no longer a training library.** It ships a Tauri desktop app, a web UI ("Unsloth
Studio"), a CLI, and the Python library, and it performs inference as well as training across
LLM, diffusion, TTS, and embedding models. Studio's backend is Python/FastAPI with a
`routes/` layout — architecturally very close to Assist, which makes porting plausible.

**Studio already has an agent loop.** A spike into `studio/backend/core/inference/` found
`studio_tool_loop.py`, `tools.py` (604 KB), `tool_call_parser.py` (140 KB) and
`mcp_client.py` (48 KB). It ships seven tools today: `edit_file`, `python`, `terminal`,
`web_search`, `search_knowledge_base`, `search_conversation`, `render_html`. Its 24 backend
routes also already cover chat history, datasets, RAG, research runs, providers/credentials,
Whisper STT, local llama serving, and export.

That killed the initiative's original first sub-project. Porting Assist's agent loop would
have replaced a working agent loop with another working agent loop — weeks of migration for
no new capability, against a 25,000-line inference module under active development. The
remaining gaps are narrower and real.

**Licensing is compatible.** Unsloth is dual-licensed: Apache 2.0 for the core, AGPL-3.0 for
Studio (`studio/LICENSE.AGPL-3.0`). Assist is itself AGPL-3.0. Combining them is permissible;
the fork inherits AGPL-3.0.

This also resolves the licence status of the tools' own dependencies. `webcam_look` uses
ultralytics YOLO, which is **AGPL-3.0** — compatible here precisely because the fork is
AGPL-3.0. (Assist's shape-detection work earlier moved off ultralytics on the mistaken belief
that Assist was closed-source and would be forced open by an AGPL dependency; Assist was
already AGPL-3.0, so no conflict existed. `detect_shapes`'s torchvision Mask R-CNN is
BSD-3-Clause and unaffected either way.) The one genuine restriction is InsightFace's
face-swap weights, which are non-commercial/research-only regardless of this project's
licence — hence the gating described below.

## Goal

Add five vision tools to Unsloth Studio's existing agent loop: `remove_background`,
`detect_shapes`, `webcam_look`, `edit_image_prompt`, and `face_swap`. Studio can generate
images but cannot edit them, identify objects in them, or see through a camera.

## Guiding constraint: merge-friendliness

This forks an actively-developed upstream, so every upstream release must be merged into these
changes. Studio has no plugin system (`backend/plugins/` holds only data-designer seeds), so
tools cannot be registered without touching upstream files. The design therefore minimises
that surface rather than eliminating it:

- Implementations live in a **new package we own**, `studio/backend/core/inference/assist_vision/`.
- Upstream `tools.py` receives the **smallest possible edit** — confining merge conflicts to a
  couple of lines rather than scattering them through a 13,800-line file.

**The registration mechanism, read from the real code** (`studio/backend/core/inference/tools.py`):

```python
# tools.py:9804 — the registry
ALL_TOOLS = [WEB_SEARCH_TOOL, PYTHON_TOOL, TERMINAL_TOOL, EDIT_FILE_TOOL,
             RENDER_HTML_TOOL, SEARCH_KNOWLEDGE_BASE_TOOL, SEARCH_CONVERSATION_TOOL]

# tools.py:9996 — the dispatcher (an if/elif chain on `name`)
def execute_tool(name: str, arguments: dict, cancel_event=None, timeout=..., session_id=None,
                 ..., output_callback=None, ...) -> str:
```

Schemas are module-level OpenAI-style dicts (`{"type": "function", "function": {...}}`). So the
two touchpoints are: one entry spliced into `ALL_TOOLS` (`*ASSIST_VISION_TOOLS`) and one
delegating branch in `execute_tool` that forwards our tool names to our package. Roughly two
lines of upstream change.

**Two adaptations this forces, and they are not cosmetic:**

1. **`execute_tool` is synchronous and returns `str`.** Assist's image tools are `async def`
   returning dicts (`{"output": ..., "image_url": ..., "error": ...}`). Ported tools must
   present a sync, string-returning face. Async model work still runs off the caller's thread
   internally, but the tool boundary is synchronous.
2. **Results are strings, so there is no `image_url` field.** The output convention becomes a
   path (or short URL) written into the returned text, which suits Studio's file-centric tools
   and reinforces the no-inline-data-URI rule below.

## Architecture

Five tools added to Studio's existing agent loop. No new loop, no new UI, no second model
stack.

**Ported near-verbatim from Assist** (already standalone, with injectable-model test seams):

- `bg_removal.py` — U2Net ONNX, 168 MB, bundled
- `shape_detect.py` — torchvision Mask R-CNN, ~170 MB, downloaded on first use
- `face_swap.py` — InsightFace detection + InSwapper, models never bundled
- `webcam.py` + `yolo.py` — camera capture and YOLO detection, 6 MB weights bundled

**Rewritten, not ported:** `edit_image_prompt` calls Studio's existing `diffusion.py` img2img
path rather than porting Assist's sd-server. Verified present: `init_image`, `strength`, and
an `Img2Img` code path. This avoids a duplicate diffusion stack and lets the tool inherit
whatever diffusion models the user already has in Studio.

**face_swap ships with its guardrails intact and non-optional.** InsightFace's models are
non-commercial/research-only, so: models are never bundled, an explicit per-user license
acceptance is required before any download, and every output carries embedded provenance
metadata identifying it as AI-face-swapped. In a public AGPL fork distributed under the
user's own name, that gating is what makes shipping the capability defensible. It is not a
configurable option.

## Data flow

**Input.** Filesystem paths are the primary input, resolved through the same confinement Assist
uses: restricted to allowed roots, sensitive-path filtering, and a size cap. This matches
Studio's file-centric tool set (`edit_file`, `terminal`), which means it works today. Whether
Studio's chat supports image *uploads* is unconfirmed; if it does, attachment-id input is a
small additive follow-on, deliberately not designed blind here.

**Execution.** Studio's tool loop dispatches to the tool function, which resolves input to
bytes and calls the model wrapper **off the event loop** via `asyncio.to_thread`. The ported
wrappers keep their lazy heavy imports (`torch`, `onnxruntime`, `insightface`) inside the
functions that run in that thread. This is not stylistic: a module-scope `import torch` in
Assist produced a measured ~2.5 s ASGI event-loop stall on first call, freezing all concurrent
requests.

**Output.** Results are written to a Studio-managed location (via its `storage/`/`state/`
modules) and the tool returns a path or short URL — **never an inline base64 data URI**. A
data URI is copied verbatim into the model's own context on every later turn and persisted
into session history that replays forever; for multi-megabyte images that is ruinous.

## Components

| Component | Origin |
|---|---|
| `bg_removal.py`, `shape_detect.py`, `face_swap.py`, `webcam.py`, `yolo.py` | Port ~verbatim from Assist |
| Tool wrappers (arg parsing, input resolution, result shaping) | Port, adapted to Studio's tool signature |
| `edit_image_prompt` | Rewritten against Studio's `diffusion.py` img2img |
| Face-swap license gate + provenance stamping | Port unchanged, non-optional |
| Registration into `tools.py` | New, minimal |

The only component not backed by proven ported code is `edit_image_prompt`, since it depends
on how Studio's diffusion entry point wants to be called.

## Error handling

**A tool never raises into the agent loop.** Every failure returns a structured error the model
can read and act on: unresolvable path, path outside allowed roots, oversized file, corrupt
image bytes, model-download failure, inference failure, and (face_swap) license-not-accepted
each produce a distinct, actionable message.

Two specifics that are easy to get wrong:

- **Zero detections is not an error.** `detect_shapes` finding nothing is a valid answer,
  unlike `face_swap` finding no face, which genuinely blocks the operation.
- **Failures name their own cause.** Assist shipped a face-swap bug that surfaced as
  `'NoneType' object has no attribute 'shape'` from inside InsightFace, because a detected
  face had no usable landmarks. It is caught at our boundary with an actionable message.

**Studio's convention, now known:** `execute_tool` returns a plain `str` — there is no error
*field* to populate. Failures are therefore returned as clear, human-and-model-readable error
text from the same string return, not raised and not signalled structurally. Ours conforms to
that rather than importing Assist's dict shape.

## Testing

Every model wrapper keeps its **injectable-model seam** (`model=None`, `analyzer=None`,
`session=None`) so the suite runs without downloading any weights.

**Tests must execute the tool and assert on real output, not assert that strings appear in
source.** This is stated explicitly because the Assist project hit three separate cases where
a fully green suite certified non-working code: wizard tests asserting JavaScript substrings
against a wizard that rendered as an empty box; packaging tests passing against a build
silently missing nine required files; and an image whose every test passed while four MCP
servers were dead. In each case the tests checked text, not behaviour.

Required coverage:

- Real image in, real image out, with asserted properties (dimensions, format, transparency,
  provenance metadata present).
- A registration test proving each tool is reachable through Studio's loop, not merely defined.
- A negative control on at least the critical paths: reintroduce the bug, confirm the test
  fails.

Manual verification of output *quality* — whether a background was removed well — remains the
user's, as no automated test covers it.

## Out of scope

- Porting Assist's agent loop, tool registry, or model-provider layer. Studio has all three.
- Chat-attachment input, pending confirmation that Studio's chat accepts image uploads.
- Any new UI. These are agent tools invoked through Studio's existing chat surface.
- The other sub-projects in this initiative: code intelligence (LSP + build/run for
  Rust/C++/C#/JavaScript — net-new work, neither app has it), multi-persona Crew, workflow
  engine + node editor, and optionally Assist's tool-RAG retrieval and privilege gating.
  Each gets its own spec.
