# Unslothed Capability Map — Design Spec

## Context

Sub-project 6, piece 3b, of the initiative to fork [Unsloth](https://github.com/unslothai/unsloth/)
into a local AI agent platform. Piece 3a (tool readiness) merged and was pushed on 2026-09-17 at
`6b552aba`. When discovery was scoped the owner chose "both readiness and a capability map"; this is
the capability-map half.

The master specification asks for it in three places:

- **§37 Tool Discovery Protocol** — the AI should be able to ask "What tools can manipulate images?
  … edit videos? … analyze PDFs?"
- **§38 Capability Discovery** — a capability map: `CAPABILITY: OCR / Available: Tesseract, Vision
  Model / Best option: Vision Model / Reason: Better layout understanding / Status: Installed`. "The
  agent should select tools based on capabilities rather than remembering specific tool names."
- **§4 Automatic Tool Acquisition** — "If the AI determines that it needs a capability that is not
  currently available, it should be capable of acquiring it." Acquisition cannot start until absence is
  detectable.

## What investigation established

**Nothing in the codebase models capabilities.** No tool carries a category, tag or capability; the
only metadata is each schema's free-text description. MCP tool schemas carry none either.

**§38's own example is not about tools.** Its two OCR providers — Tesseract (a program) and "Vision
Model" (the loaded model's own ability) — are neither of them among the 18 tools. A map of tools alone
would report "no OCR" on a machine that can do it.

**The `python` tool runs the Studio environment itself** (`subprocess.Popen([sys.executable, ...])`),
so everything importable there is reachable by the model. On the owner's machine that includes PyMuPDF
and pymupdf4llm (PDF), Pillow and OpenCV (images), Whisper (speech), python-docx (Word), onnxruntime,
torch/torchvision, insightface and ultralytics. Tesseract, FFmpeg and EasyOCR are absent.

**Cheap import checks can lie.** Whisper imports cleanly, but its `load_audio` states "Requires the
ffmpeg CLI in PATH" — so on this machine speech-to-text would read `ready` while every transcription
fails. A provider must be able to declare several requirements.

**The loaded model's abilities are readable without editing do-not-edit code.** `LlamaCppBackend`
exposes read-only `is_loaded`, `is_vision` and `model_identifier`; the transformers orchestrator keeps
`is_vision` per model. llama.cpp's `_is_audio` routes audio *output* (TTS), not audio understanding.

**Getters in this codebase create what they are asked about.** `get_inference_backend()`,
`get_diffusion_backend()` and `get_sd_cpp_backend()` each construct an instance when none exists.
`get_llama_cpp_backend()` lives in `routes/inference.py`, a ~20,000-line route module that core code
must not import. A probe calling any of these would start or couple to the subsystem it was asking
about.

**`edit_image_prompt` depends on which diffusion engine is active.** It routes through
`diffusion_engine_router.get_active_diffusion_engine()`; the native sd.cpp engine raises on
`init_image` ("img2img / inpaint / reference / upscale are not yet supported", `sd_cpp_backend.py:2153`).

**A parked 3a residual was disproven.** The claim that `tests/test_tool_readiness_probes.py` opens the
user's real `~/.unsloth/studio/rag/rag.db` is false: logging every `sqlite3.connect` during that file
showed only pytest temporary homes, and the real file's timestamp and size were unchanged. No fix is
needed and none is in scope.

## Decisions taken by the owner

| Question | Decision |
|---|---|
| What counts as a provider | **Tools + installed software + the loaded model's abilities** |
| How "Best option" is chosen | **Curated preference order** — best = first provider that is actually ready |
| Which capabilities exist | **§37/§38's named examples + a capability for everything current tools do** |
| How the model queries it | **A new `find_capability` tool** |

## Goal

Let the model ask what can accomplish a task — across tools, installed software and its own abilities —
which option is best, and whether it works right now; and make missing capabilities visible so §4 has
something to act on.

## Architecture

### Package

`core/inference/capability_map/`, beside `tool_readiness/` rather than inside it. The map consumes
readiness; readiness must never depend on the map.

| File | Responsibility |
|---|---|
| `vocabulary.py` | the curated data: every capability and its providers, in preference order |
| `providers.py` | requirement checks for software and model abilities |
| `__init__.py` | resolve one capability or the whole map; the tool's `execute()` |
| `schemas.py` | the `find_capability` schema + `CAPABILITY_TOOLS` / `CAPABILITY_TOOL_NAMES` |

3a's `tool_readiness/probes.py` gains probes for the three tools that still report `unknown`.

### Data model

```python
@dataclass(frozen = True)
class Requirement:
    kind:  str   # "tool" | "module" | "binary" | "model" | "cached_model"
    name:  str   # "python" | "fitz" | "ffmpeg" | "vision" | "whisper"

@dataclass(frozen = True)
class Provider:
    kind:     str                    # "tool" | "software" | "model" — for display
    name:     str                    # "detect_shapes" | "PyMuPDF" | "vision model"
    via:      str | None             # "python" / "terminal" for software; None otherwise
    requires: tuple[Requirement, ...]
    reason:   str                    # why it sits at this rank

@dataclass(frozen = True)
class Capability:
    name:      str                   # "ocr"
    title:     str                   # "Read text in images"
    aliases:   tuple[str, ...]       # ("text recognition", "read text from image")
    providers: tuple[Provider, ...]  # curated preference order; may be empty
    acquire:   str | None            # §4 hook — descriptive text only
```

A software provider's `via` tool is an implicit requirement: PyMuPDF is only usable if `python` is.

### Status rules

Every requirement resolves to 3a's `Readiness` (`ready` / `missing` / `unknown`). A provider is:

- **ready** — every requirement is ready
- **missing** — at least one requirement is missing
- **unknown** — otherwise

A capability is:

- **ready** — at least one provider is ready; **best option** = the first ready provider in curated
  order, with that provider's `reason`
- **missing** — every provider is missing, **or the capability has no providers**
- **unknown** — otherwise

**An `unknown` never makes anything `ready`**, at either level — 3a's central claim carried up.

### Matching a query

`find_capability("OCR")` matches a capability `name` or any `alias`, case-insensitively, after trimming.
No fuzzy or model-based matching. An unmatched query returns the list of capability names and titles —
not an error — so the model can pick one.

## Checking requirements

All checks are cheap: an import-spec lookup, a PATH lookup, a file stat, or reading state already in
memory. **Nothing is imported with side effects, nothing is loaded or created, nothing touches the
network.**

| Kind | Check |
|---|---|
| `tool` | 3a's `tool_readiness.resolve(name)`, unchanged — with its 60 s cache and guard |
| `module` | `importlib.util.find_spec(name)` (3a's `_module_present`) |
| `binary` | `shutil.which(name)` |
| `model` | read-only backend state — see below |
| `cached_model` | at least one model file in that library's download cache — see below |

### Cached models

A library that downloads its model on first use is **`missing` until a model is cached**, matching
3a's `webcam_look` convention (weight absent → `missing`, remedy "downloads on first use"). 3a's final
review marked the opposite inconsistency — one auto-downloading tool `ready`, another `missing` —
Important.

v1 has one entry, `whisper`: any `*.pt` in `os.path.join(os.getenv("XDG_CACHE_HOME", ~/.cache),
"whisper")`. Importing whisper to ask it would load torch, so the path is pinned, and a test reads
`whisper.load_model`'s source and asserts it still builds `download_root` that way.

Verified on the owner's machine on 2026-09-17: after installing FFmpeg 9.0.1, `whisper.audio.load_audio`
decoded a generated 2-second tone into exactly 32,000 samples (16 kHz), but no model is cached, so
speech-to-text is correctly `missing` until the first transcription downloads one.

### New tool probes (added to 3a's `probes.py`)

| Tool | Probe | Delegates to |
|---|---|---|
| `remove_background` | `onnxruntime` importable; weight file present | `bg_removal._model_path()`, which honours `UNSLOTH_VISION_MODEL_DIR` |
| `detect_shapes` | `torch` and `torchvision` importable; Mask R-CNN weight present | weight filename pinned in code (`maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth`), directory from `assist_vision.models.model_root()` + `/checkpoints` |
| `edit_image_prompt` | an image model loaded on the **diffusers** engine | `diffusion_engine_router._active_engine_name` and the engine module's backend global, read via `sys.modules` |

**Corrected 2026-09-17, after a task review caught it.** This table first specified `torch.hub.get_dir()`
for the Mask R-CNN weight, with `unknown` whenever `torch` was not yet imported. That was wrong:
`shape_detect._get_model()` calls `torch.hub.set_dir(model_root())` before the download
(`shape_detect.py:69`, stated in its module docstring), so the weight caches under the app's own vision
model root, not torch's default. Reading torch's default would report `missing` for an already-cached
weight in any process where torch happened to be imported by another tool. `model_root()` needs no
torch, so the `unknown` fallback disappears — and delegating to it is the rule this spec sets for every
other probe.

`edit_image_prompt` states, in order:

| State | Readiness |
|---|---|
| router or engine module not in memory | `unknown` |
| active engine is native sd.cpp | `missing` — "native engine does not support image-to-image" |
| diffusers engine active, no model loaded | `missing` — "no image model loaded in Studio" |
| diffusers engine active with a model loaded | `ready` | Reading `MaskRCNN_ResNet50_FPN_Weights`
at runtime would import torchvision's detection stack, so the filename is pinned and a test asserts it
equals `MaskRCNN_ResNet50_FPN_Weights.DEFAULT.url`'s basename.

### Model abilities

v1 checks one ability: **vision**.

- **GGUF:** `sys.modules.get("routes.inference")` → its `_llama_cpp_backend` → `is_loaded`, `is_vision`,
  `model_identifier`.
- **Transformers:** the orchestrator module's `_inference_backend` global — **never
  `get_inference_backend()`** — → its active model's `is_vision`.

| Backend state | Vision |
|---|---|
| either backend has a loaded vision model | `ready` — the report names it ("Qwen2.5-VL-7B is loaded") |
| a local model is loaded, and no loaded model has vision | `missing` — the report names the loaded model |
| no backend module in memory, or no local model loaded | `unknown` — an external or API model may be serving, with abilities this process cannot see |

Audio is excluded: the only llama.cpp flag routes TTS output, and treating it as audio understanding
would be a false `ready`.

## The capability vocabulary

Providers are listed in curated preference order. "—" means the capability has no provider in v1 and is
therefore `missing`.

| Capability | Title | Providers (order) | Acquire hint (when missing) |
|---|---|---|---|
| `internet_search` | Search the web and read pages | tool `web_search` | Needs the ddgs Python package. |
| `run_python` | Run Python code | tool `python` | — |
| `run_commands` | Run terminal commands | tool `terminal` | — |
| `filesystem` | Read, write and edit files | tool `edit_file` (exact edits); tool `terminal` (list, move, read); tool `python` (bulk or structured) | — |
| `pdf_analysis` | Extract text and tables from PDFs | pymupdf4llm via python (keeps layout as Markdown); PyMuPDF via python (page-level access) | Needs the PyMuPDF Python package. |
| `ocr` | Read text in images | vision model (understands layout, nothing to install); Tesseract via terminal; EasyOCR via python | Load a vision-capable model, or install Tesseract OCR (a free program). |
| `image_understanding` | Describe or answer questions about an image | vision model | Load a vision-capable model. |
| `object_detection` | Find and label objects in a photo | tool `detect_shapes` (boxes and confidences); vision model (descriptive, no boxes) | Needs torch, torchvision and the Mask R-CNN weights. |
| `camera` | See through the webcam | tool `webcam_look` | Needs a webcam, ultralytics, OpenCV and the YOLO weights. |
| `image_processing` | Resize, crop, convert or adjust images precisely | Pillow via python (simple, exact); OpenCV via python (advanced operations) | Needs the Pillow Python package. |
| `image_editing` | Change an image by describing the edit | tool `edit_image_prompt` | Load an image model in Studio on the diffusers engine. |
| `background_removal` | Remove an image's background | tool `remove_background` | Needs onnxruntime and the u2net model. |
| `face_swap` | Swap a face between two images | tool `face_swap` | Needs the InsightFace licence accepted and its models downloaded. |
| `image_generation` | Create an image from a text description | — | Studio can generate images on its Images page, but no agent tool exposes text-to-image yet. |
| `video_editing` | Cut, convert or combine video | FFmpeg via terminal | Needs FFmpeg, a free command-line program. |
| `speech_to_text` | Transcribe audio or video speech | Whisper via python (requires module `whisper`, binary `ffmpeg` **and** a cached Whisper model) | Needs the Whisper Python package, FFmpeg (a free command-line program), and a Whisper model, which Whisper downloads on first use (the smallest is about 75 MB). |
| `word_documents` | Read and write Word documents | python-docx via python | Needs the python-docx Python package. |
| `document_search` | Search the user's uploaded documents | tool `search_knowledge_base` | Upload documents to a knowledge base. |
| `conversation_recall` | Recall earlier parts of this conversation | tool `search_conversation` | Needs the conversation archive enabled with RAG available. |
| `code_intelligence` | Definitions, references, types and errors in code | the code tools (requires tools `code_definition`, `code_references`, `code_hover`, `code_symbols`, `code_diagnostics`) | Needs a language server for the project's language. |
| `visual_output` | Show the user an interactive HTML canvas or chart | tool `render_html` | — |
| `computer_control` | Control the mouse, keyboard and applications | — | No provider yet; desktop control is planned for a later piece. |

That is 22 capabilities. Aliases are defined per capability in `vocabulary.py`; the plan lists them.
`check_tool_readiness` and `find_capability` are discovery tools and deliberately map to no capability.

Reasons for each provider's rank are written in `vocabulary.py` beside the provider; the parentheticals
above are their gist.

**Every acquire hint is descriptive, never a runnable command.** §4 requires downloads to be
identified, verified and presented to the user; a paste-ready `winget install` would invite the model
to install through `terminal`, and under the "full" permission mode nothing would stop it. Acquisition
is a later piece with its own security design.

## The tool

`find_capability`, with an optional `capability` argument. Its description is written for the question
the model has: *"Find which tools, installed software or model abilities can do something (read text in
images, edit video, analyse PDFs), which is best, and whether it works right now. Omit `capability` to
see the whole map, including what is missing."*

### Output

One capability, following §38's format:

```
CAPABILITY: ocr — Read text in images
Status:      ready
Best option: vision model (Qwen2.5-VL-7B is loaded)
Reason:      understands layout; nothing to install
Providers:
  1. vision model              ready     Qwen2.5-VL-7B is loaded
  2. Tesseract  via terminal   missing   tesseract not on PATH
  3. EasyOCR    via python     missing   easyocr not installed
```

With no argument: one line per capability — ready first, then unknown, then missing with its acquire
hint — followed by a note that MCP tools are not in the map and their descriptions are in the model's
tool list. Tool providers are always phrased "via tool `web_search`" so the model can match them
against the tools actually offered to it in this request.

### Registration

A new built-in tool needs five registration points (the `_addable` re-add has been missed three times).
Every one of 3a's corresponding lines is fork-inserted, so `find_capability` joins them:

| Point | Change | Seam |
|---|---|---|
| `tools.py` import | new line beside the readiness import (`:9842`) | +1 |
| `ALL_TOOLS` | `*CAPABILITY_TOOLS,` after `*READINESS_TOOLS,` (`:9854`) | +1 |
| dispatch in `execute_tool` | block after the readiness block (`:10085`) | +3 |
| `_ALWAYS_SAFE_TOOLS` | amend the fork's existing rebind (`:4541`) | 0 |
| `_addable` in `routes/inference.py` | add `CAPABILITY_TOOL_NAMES` to the fork's import (`:3674`) and union (`:3676`) | +1 |
| `_ANTHROPIC_UNPROMPTED_SAFE_TOOLS` | amend the fork's existing rebind (`:20049`) | 0 |

`tools.py` goes from **88 to ~93** insertions, `routes/inference.py` from **58 to ~59**, both with **zero
deletions**. The tool is read-only, so it belongs in both safe sets.

## Error handling

Every requirement check runs inside 3a's `resolve()` guard or an equivalent bare `except BaseException`;
a check that raises reports `unknown` carrying its error text. `execute()` never raises, including on a
non-string or unhashable `capability` argument.

## Testing

Every negative control is **demonstrated to fail** before its test counts. Only named test files are
run; **never the full backend suite** (upstream fixtures fabricate GGUF files up to 40 GB).

**Registration and seam**

- Reachability through the **real** `_select_request_tools` with a Studio-style payload: `find_capability`
  survives. Control: remove it from `_addable`. The existing readiness reachability test evaluates the
  real `_addable` expression and must be given the new name in its namespace.
- Both safe-set rebinds asserted by behaviour. Control: remove the name from each.
- `tools.py` passes both hard-coded ceilings of 90 (`tests/test_tool_audit_seam.py`,
  `tests/test_tool_readiness_seam.py`); **both are bumped** as part of the registration task. The
  zero-deletion assertions stay unchanged. `routes/inference.py`'s seam test asserts deletions only.

**Vocabulary integrity**

- Every `tool` requirement names a real `ALL_TOOLS` entry.
- Every `ALL_TOOLS` entry except the two discovery tools appears in at least one capability.
- No alias maps to two capabilities.
- **No acquire hint contains an install command** — `pip install`, `winget`, `choco`, `npm i`,
  `apt install`, `apt-get`, `brew install`, `conda install`, `curl`, `dotnet tool install`.

**Status rules** — each with its control

- Best option is the first *ready* provider, not the first listed.
- An `unknown` provider never makes a capability `ready`.
- A software provider requires its `via` tool.
- Whisper with `ffmpeg` absent → `missing`; with no cached model → `missing`; with all three → `ready`.
- A capability with no providers → `missing`.

**No-side-effect probes**

Model-ability and diffusion tests inject **fake modules into `sys.modules`**; the real heavy modules
are never imported. Each fake's creating getter **fails the test if called**. Control: make a probe call
the getter.

**Pinned facts with drift tests**

- The Mask R-CNN weight filename equals `MaskRCNN_ResNet50_FPN_Weights.DEFAULT.url`'s basename
  (torchvision imported in that test only).
- The pinned Whisper cache directory matches how `whisper.load_model` builds `download_root`.
- An active sd.cpp engine makes `edit_image_prompt` `missing` with its reason.

**Output**

- A single capability renders in §38's format; the full map orders ready, unknown, missing.
- An unmatched query returns the capability list.
- `execute()` never raises on bad arguments.

## Risks

**The vocabulary is hand-maintained.** A new tool without a capability entry fails the integrity test,
which is the intended forcing function; a new package or program does not, and will be invisible until
someone adds it.

**Curated order ignores the task.** "Best OCR" is the vision model for a scanned letter and may be
Tesseract for a thousand-page batch. The `reason` strings make the trade-off readable; the model can
choose another listed provider.

**The report describes the local model, not necessarily the serving one.** Mitigated by naming it.

**Presence is not correctness.** A corrupt weight, a broken `ffmpeg` build or a language server that
crashes on start all pass a cheap check — the same accepted limit as 3a.

**The seam grows again**, by ~5 lines in `tools.py` and ~1 in `routes/inference.py`, all insertions.

## Out of scope

- **Dedicated `transcribe_audio` and `edit_video` tools — the owner's chosen NEXT piece (3c).** This
  piece maps both capabilities through `python` + Whisper and `terminal` + FFmpeg. When 3c ships, those
  tools are added to the vocabulary as each capability's first-preference provider. 3c needs its own
  design: which edit operations, confining file access to the chat's workspace, timeouts and
  cancellation for long media, and consent before a Whisper model download.
- Acquiring anything (§4) — this piece only makes absence visible and describes what would fill it
- MCP tools in the map
- Audio understanding as a model ability
- Parameterised, per-argument readiness
- Task-aware ranking
- Any UI
- The `rag.db` test-isolation "fix" (the defect does not exist)
