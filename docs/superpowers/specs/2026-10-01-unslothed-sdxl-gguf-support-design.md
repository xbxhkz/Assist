# SDXL GGUF Support — Design

**Status:** Approved by conversational design review, 2026-10-01.

**Repo:** `C:\Users\Admin\unsloth`, `studio/backend`. Base branch: `unslothed-release`.

**Platform constraint (overrides anything that assumes portability):** this is built for exactly
one machine — Windows 11, NVIDIA CUDA 13.0, sm_120 (RTX 5080 Laptop, 16GB VRAM, 64GB system RAM).
Do not design for other accelerators or OSes. The design routes entirely through the existing
diffusers GPU path and deliberately does **not** touch the native sd.cpp engine (see §3).

## 1. Problem

A user tried to load `offgrid-ai/pony-diffusion-v6-xl-GGUF` (Q8_0) from the chat interface and
got the generic "This GGUF carries no llama.cpp model metadata (no general.architecture)..."
refusal instead of a specific one. Investigation found two separate issues:

1. **Cosmetic:** `pony-diffusion-v6-xl` doesn't match any of SDXL's registered name/aliases in
   `core/inference/diffusion_families.py`'s `_FAMILIES` registry (`sdxl`, `sd-xl`, `sd_xl`,
   `sdxl-turbo`, `sdxl-base`, `stable-diffusion-xl`), so the chat-refusal code's family lookup
   (`detect_family`, name-based) fails and falls through to the generic message instead of
   "This is an image-generation model... Open it from the Images page instead."
2. **Structural (the real gap, this spec's subject):** even with the alias fixed, the Images page
   genuinely cannot load this GGUF today. SDXL is the one family in the registry marked
   `single_file_is_pipeline = True` (its checkpoint convention bundles UNet + VAE + text
   encoders into one file, unlike FLUX/Qwen-Image-style transformer-only GGUFs), and
   `diffusion.py`'s loader explicitly refuses any GGUF for such a family
   (`core/inference/diffusion.py:1781-1785`): `"'{fam.name}' checkpoints are whole-pipeline
   single files and have no GGUF transformer variant; load the .safetensors pipeline instead
   of a GGUF."` `family_gguf_loadable()` (`diffusion_families.py:1292-1300`) encodes the same
   exclusion for the picker/listing side.

This spec adds a real GGUF-loading path for this one family shape, so a GGUF like Pony Diffusion
V6 XL can actually be loaded and served from the Images page, not just correctly refused.

## 2. What already exists (do not duplicate)

Investigated before designing this, per this project's standing reconciliation discipline:

- **The transformer-only GGUF path** (`diffusion.py` lines ~4173-4198): used by FLUX,
  Qwen-Image, and every other family with `single_file_is_pipeline = False`. Dequantizes a
  GGUF-quantized **transformer/denoiser component only** via `transformer_cls.from_single_file
  (single_file_path, quantization_config = diffusers.GGUFQuantizationConfig(compute_dtype =
  dtype), ...)`, sourcing VAE/text-encoder(s)/scheduler separately from `fam.base_repo`. This
  spec's new path is architecturally analogous but cannot reuse this function directly: SDXL's
  GGUF is not a transformer-only file (see §3), so `GGUFQuantizationConfig` + `from_single_file`
  on a single component does not apply — the whole file must be read and split first.
- **`_install_gguf_prefix_strip`** (`diffusion.py`, called at line 4194): already handles
  sd.cpp-style tensor key prefixing (`model.diffusion_model.*`) for the transformer-only path,
  confirming this project already has precedent for sd.cpp-converted GGUF naming conventions —
  but it operates on a diffusers component class's `from_single_file`, not a raw multi-component
  checkpoint, so it is not reusable as-is either.
- **The existing remote/local GGUF header peek** (`llama_cpp.py`'s `_remote_non_chat_gguf_verdict`
  / `_remote_non_chat_gguf_refusal`, using `core.inference.diffusion_compat._read_gguf_header`
  and `_read_local_header`): a 256 KiB byte-range read used to answer "does this GGUF carry chat
  metadata" before a multi-GB download, with a documented, measured rationale for the byte
  budget (chat-model KV blocks are megabytes; non-chat media GGUFs finish their whole KV walk in
  under 4 KB). This spec reuses the SAME underlying header-read primitives for content-based
  SDXL detection (§4.2), but needs its OWN byte budget: this file's tensor-INFO array (not its
  KV section) is what must be covered, and a 2513-tensor SDXL checkpoint's tensor table is much
  larger than a typical chat GGUF's KV block. The exact budget is an implementation-time
  measurement (§7), not a value this spec fixes.
- **The native sd.cpp engine** (`core/inference/sd_cpp_backend.py`,
  `diffusion_engine_router.py`'s `predict_engine`): deliberately NOT used by this design.
  `predict_engine` only routes to sd.cpp on a CPU or (opted-in) MPS host, or under an explicit
  forced-native preference (`diffusion_engine_router.py:264-271`). On this project's actual
  target machine (a CUDA GPU), this path would never fire by default, so wiring
  `sd_cpp_vae`/`sd_cpp_text_encoders` for SDXL would not solve "load it from the Images page on
  this machine" — it would require a config change this project doesn't use. `sd_cpp_backend.py`
  also has zero existing SDXL-specific code today; building that is a materially different,
  larger undertaking this spec explicitly defers (§3, Non-goals).
- **Diffusers' own single-file conversion functions**
  (`diffusers.loaders.single_file_utils.convert_ldm_unet_checkpoint`,
  `convert_ldm_vae_checkpoint`, `convert_ldm_clip_checkpoint`, `convert_open_clip_checkpoint`):
  confirmed present and independently callable in the installed `diffusers==0.40.0.dev0`
  (verified via direct `inspect.signature()`, not docs). These are the exact functions
  `StableDiffusionXLPipeline.from_single_file()` already calls internally when loading a plain
  `.safetensors` SDXL checkpoint — this spec's new loader reuses them directly rather than
  reimplementing SDXL checkpoint conversion.

## 3. Non-goals

- **No native sd.cpp engine work.** See §2 — `predict_engine` would not route there on this
  machine's CUDA GPU by default, so it would not solve the reported problem. A future piece
  could add sd.cpp SDXL support for CPU/MPS hosts; out of scope here.
- **No support for a GGUF-quantized transformer-only SDXL variant.** Every SDXL GGUF repo
  checked (see §4.1) packages the whole pipeline (UNet + VAE + dual CLIP) in one file, matching
  the "original checkpoint" convention, not a transformer-only convention. If a
  transformer-only SDXL GGUF convention is ever observed in the wild, it is a different shape
  needing its own design.
- **No change to the existing transformer-only GGUF path.** FLUX/Qwen-Image/etc. are untouched.
- **No attempt to preserve GGUF's quantization through to VRAM savings.** The new path
  dequantizes to the pipeline's working dtype on load, same as the existing transformer-only
  path already does (`GGUFQuantizationConfig(compute_dtype = dtype)` dequantizes too) — a
  GGUF's benefit here is a smaller download, not a smaller resident footprint.
- **No new quantization schemes beyond what `gguf.quants.dequantize()` already supports.**
  Confirmed present in the installed `gguf==0.19.0` package (verified via direct inspection).
  If a future SDXL GGUF uses a ggml type this function doesn't handle, it refuses cleanly
  (§6), it does not attempt a new dequantization implementation.
- **No LoRA support for a GGUF-loaded SDXL pipeline in this piece.** SDXL's existing LoRA
  trainer/deployer continues to target the safetensors path only, matching the
  `train_base_repos` already declared on the family.

## 4. Architecture

Three additive pieces:

### 4.1 Confirmed file shape (verified against the real repo, not assumed)

Read directly from `offgrid-ai/pony-diffusion-v6-xl-GGUF`'s `pony-diffusion-v6-xl-Q8_0.gguf`
(4.18 GB) via an HTTP range request for the first 3 MB (covers the full header: magic, version,
tensor count, KV section, and the complete 2513-entry tensor-info table — verified by a hand
-written parser, not `gguf.GGUFReader`, which requires the full file since it also validates
tensor data):

- `kv_count = 0` — **zero metadata key-value pairs**, confirming the original "no
  `general.architecture`" refusal is correct: this file genuinely carries none.
- `tensor_count = 2513`, falling into exactly three top-level key prefixes:
  - `conditioner.*` (585 tensors) — the dual text encoders. `conditioner.embedders.0.*` is
    CLIP-L (keys like `...self_attn.k_proj.weight`, `...mlp.fc1.weight` — OpenAI CLIP
    convention); `conditioner.embedders.1.*` is CLIP-G/OpenCLIP (SDXL's second encoder).
  - `first_stage_model.*` (248 tensors) — the VAE (keys like `decoder.conv_in.weight`,
    `decoder.mid.attn_1.k.weight` — the standard "original" VAE checkpoint convention).
  - `model.diffusion_model.*` (1680 tensors) — the UNet (keys like
    `input_blocks.0.0.weight`, `...emb_layers.1.weight`, `...in_layers.0.weight` — the
    standard "original" SD/SDXL UNet checkpoint convention).
- Two ggml tensor types present: type `1` (F16, used for biases/norms) and type `8` (Q8_0, used
  for the large weight matrices) — a standard mixed-precision GGUF quantization, confirming
  `gguf.quants.dequantize(data, qtype)` (present in the installed `gguf==0.19.0`, confirmed via
  direct inspection) is sufficient; no custom dequantization code is needed.

This confirms the file uses the exact "original checkpoint" key convention diffusers' existing
`convert_ldm_unet_checkpoint` / `convert_ldm_vae_checkpoint` / `convert_ldm_clip_checkpoint` /
`convert_open_clip_checkpoint` already parse for a plain safetensors SDXL load — the GGUF
format is a different container for the SAME tensor-naming convention, not a different
checkpoint layout.

### 4.2 Content-based SDXL-GGUF detection (new)

Both the picker (deciding whether to list/offer the GGUF) and the chat-refusal message
(explaining why it can't load as chat) need to recognize "this is an SDXL-shaped
whole-pipeline GGUF" — by content, not by repo name. Repo-name aliasing (the cosmetic fix in
§1) does not scale to the many unpredictably-named SDXL community finetunes (Pony, Illustrious,
NoobAI, AnimagineXL, etc.) the way a content signature does, and this project's own chat-GGUF
architecture detection is already content-based (reading `general.architecture` from the header)
— this follows the same precedent.

**Signature:** a GGUF with no `general.architecture` KV, whose tensor names include at least one
each of the `model.diffusion_model.`, `first_stage_model.`, and `conditioner.embedders.` prefixes.
All three must be present — a file with only one or two (e.g. a VAE-only or encoder-only export)
is not a loadable whole pipeline and must not match.

**Where this plugs in:**
- **Local file, already present** (`llama_cpp.py`'s `_non_chat_gguf_refusal`, called after
  `self._architecture` reads `None` for a local GGUF): add this signature check using the
  already-open local reader, before falling through to the generic message (~line 12052).
  Returns the more specific "This is an image-generation GGUF (SDXL), which cannot run as a
  chat model. Open it from the Images page instead." (or the "Studio could not tell which...
  cannot assemble it either" variant, mirroring the existing `_AMBIGUOUS_IMAGE_ARCHES` message
  shape) once `family_gguf_loadable`/`family_buildable_here` (§4.3) confirm the Images page can
  actually take it.
- **Remote, not-yet-downloaded repo** (`llama_cpp.py`'s `_remote_non_chat_gguf_verdict`, using
  `core.inference.diffusion_compat._read_gguf_header`/`_read_local_header`): same signature
  check, applied to the byte-range-fetched prefix instead of a locally-opened file. The existing
  256 KiB budget was measured for a DIFFERENT property (a chat GGUF's KV block size) and must be
  re-measured for this one (the tensor-INFO table's size, which scales with tensor count, not
  metadata) — confirmed empirically during this investigation that 3 MB safely covers a
  2513-tensor file; the exact minimum is an implementation-time measurement (§7), and the
  existing fail-open behavior (too short a prefix → no verdict, defer to the full download) is
  preserved unchanged, so a wrong guess here costs a missed cheap refusal, never a wrong one.
- **Picker/listing** (`diffusion_families.py`'s `detect_family_for_pick`, called from
  `llama_cpp.py`'s `_ambiguous_image_arch_is_pickable` and the Images-page listing route): for a
  remote repo not yet locally present, this needs the SAME remote byte-range signature check
  before the family can resolve to SDXL at listing time. This is the one place the existing
  mechanism's CALLERS (not just its header-reading primitive) need a new consumer — the picker
  currently only resolves families by name (`_token_in_needle` against `fam.name`/`fam.aliases`);
  it needs a second resolution path that tries the content signature when name-matching fails
  and a `.gguf` file is in play.

### 4.3 The new whole-pipeline GGUF loader

New module: `core/inference/diffusion_sdxl_gguf.py`. One entry point:

```python
def load_sdxl_pipeline_from_gguf(
    gguf_path: str,
    *,
    base_repo: str,
    pipeline_class: str,
    dtype,
    local_files_only: bool,
    hf_token: Optional[str],
) -> "StableDiffusionXLPipeline":
```

Steps:
1. Read the full GGUF's tensor-info table (name, dims, ggml type, data offset) — the same
   parsing shape as §4.1's investigation, but reading the whole file this time, not a header
   prefix.
2. For each tensor, dequantize via `gguf.quants.dequantize(raw_bytes, qtype)`, reshape to the
   declared dims, convert to a `torch.Tensor` at `dtype`.
3. Split the resulting flat `{name: tensor}` dict into three checkpoints by the §4.1 prefixes.
4. Fetch `base_repo`'s component configs (UNet, VAE, each CLIP text encoder) the same way the
   existing transformer-only path already fetches `fam.base_repo` for VAE/text-encoder configs
   (reusing that existing fetch machinery, not rebuilding it).
5. Run each checkpoint through diffusers' existing converters: `convert_ldm_unet_checkpoint
   (unet_checkpoint, unet_config)`, `convert_ldm_vae_checkpoint(vae_checkpoint, vae_config)`,
   `convert_ldm_clip_checkpoint(clip_l_checkpoint)` for `conditioner.embedders.0`, and
   `convert_open_clip_checkpoint(clip_g_text_model, clip_g_checkpoint)` for
   `conditioner.embedders.1` (this one needs an already-constructed bare `text_model` instance
   per its signature — construct one from `base_repo`'s CLIP-G config before calling it, the
   same pattern `create_diffusers_clip_model_from_ldm` uses internally).
6. Assemble a `StableDiffusionXLPipeline` (or whichever `fam.pipeline_class` names — passed
   through, not hardcoded, so this stays generically named even though SDXL is the only caller
   today) from the converted components plus a scheduler from `base_repo`.
7. Return the assembled pipeline through the exact same return shape the existing
   `single_file_is_pipeline` safetensors branch (`diffusion.py:4160-4172`) already returns,
   so every downstream consumer (generation, VRAM tracking, eviction, the Unified Memory core
   park/restore system from the prior piece) sees an ordinary resident SDXL pipeline and needs
   no awareness this one came from a GGUF.

### 4.4 Family registry and gate changes

- New `DiffusionFamily` field: `whole_pipeline_gguf_supported: bool = False`
  (`diffusion_families.py`, alongside the other `sd_cpp_*`/capability fields). Set `True` only
  on the SDXL entry (`diffusion_families.py:403-422`) for this piece.
- `family_gguf_loadable()` (`diffusion_families.py:1292-1300`), currently
  `return not fam.single_file_is_pipeline and not fam.pipeline_only`, becomes:
  `return not fam.pipeline_only and (not fam.single_file_is_pipeline or
  fam.whole_pipeline_gguf_supported)`.
- `diffusion.py`'s refusal at line 1781 (`if kind == "gguf" and fam.single_file_is_pipeline:
  raise ValueError(...)`) becomes conditional on `not fam.whole_pipeline_gguf_supported`,
  with the `True` case routing to §4.3's new loader instead of raising.
- `family_buildable_here()` (`diffusion_engine_router.py:279-306`) needs NO change: once the
  gate above stops excluding SDXL, `family_pipeline_available(fam)` already returns `True`
  (diffusers ships `StableDiffusionXLPipeline` unconditionally — it is not a new-enough-diffusers
  problem the way some native-only families are), so the existing `if
  family_pipeline_available(fam): return True` short-circuit already covers this case correctly
  with zero sd.cpp involvement, confirming §3's non-goal is structurally enforced, not just
  a stated intention.

## 5. Data flow (worked example, Pony Diffusion V6 XL)

1. User searches/selects `offgrid-ai/pony-diffusion-v6-xl-GGUF` on the Images page (not yet
   downloaded). The picker's family resolution tries name-matching first (fails, per §1), then
   §4.2's remote content-signature check against a byte-range-fetched prefix, resolves to SDXL,
   confirms `family_gguf_loadable(sdxl_family)` is now `True`, lists the model with its quant
   variants (Q4_K, Q8_0).
2. User picks the Q8_0 variant and loads. `diffusion.py`'s existing load-request validation
   (`validate_load_request`/the loader's kind-dispatch) reaches the `kind == "gguf" and
   fam.single_file_is_pipeline` branch, sees `whole_pipeline_gguf_supported = True`, calls
   §4.3's `load_sdxl_pipeline_from_gguf(...)` instead of raising.
3. The loader downloads the full 4.18 GB file (existing download machinery, unchanged),
   dequantizes, splits, converts, assembles, returns a `StableDiffusionXLPipeline` like any
   other resident SDXL load.
4. Generation, VRAM tracking, and eviction (including the park/restore system from the prior
   Unified Memory core piece) all see an ordinary resident SDXL pipeline — no awareness needed
   that this one's weights originated from a GGUF file.
5. Separately: if the user instead tries to open this same GGUF from the **chat** interface,
   `_non_chat_gguf_refusal`'s local-file path (§4.2) now correctly reports "This is an
   image-generation GGUF (SDXL)... Open it from the Images page instead" — the specific message,
   not the generic fallback that triggered this whole investigation.

## 6. Error handling

- **A `.gguf` file matching the §4.2 signature but with a tensor the converter can't map**
  (wrong shape, an unexpected ggml type `gguf.quants.dequantize` doesn't support): refuse
  cleanly before/during the load with a message naming the specific tensor and what was
  expected, not a bare traceback — matching this project's existing "refuse before/during the
  slow work with a clear reason" pattern used throughout the diffusion loaders.
- **A `single_file_is_pipeline` GGUF that does NOT match the three-prefix signature** (some
  other, unanticipated whole-pipeline convention): falls through to the existing "Studio cannot
  assemble this particular model from a GGUF either" refusal (§4.2's local/remote detection
  simply doesn't match, so nothing new fires) — never a crash, never a false-positive load
  attempt on an incompatible file.
- **Download/network failures** during the full-file fetch: unchanged, existing download
  machinery and its existing error handling already cover this; the new loader only begins
  after a successful download, same as the existing transformer-only GGUF path.
- **Out-of-memory during dequantization or pipeline assembly**: unchanged, existing VRAM/RAM
  preflight and OOM handling already wrap every diffusion load; this path does not bypass it.

## 7. Testing

Per this project's standing test discipline (no real model, GPU, or multi-GB download in
automated tests):

- **The dequantize+split+convert pipeline**: pure-unit tests using a small, hand-built synthetic
  GGUF (a handful of correctly-prefixed, correctly-shaped tiny tensors covering all three
  prefixes and both observed ggml types, F16 and Q8_0) — not a download of the real 4.18 GB
  file. Confirms the three-way prefix split is correct, confirms each slice reaches the right
  diffusers converter with the right argument shape, confirms the assembled pipeline has the
  expected component classes.
- **§4.2's content-signature detection**: pure-unit, synthetic GGUF headers (no real network),
  covering: the real three-prefix match, each of the three prefixes missing one at a time (must
  NOT match), a chat GGUF's real shape (has `general.architecture`, must not be mistaken for
  this signature), and the "too-short prefix, fails open" case for the remote path.
- **The gate changes** (`family_gguf_loadable`, the `diffusion.py` routing branch): unit tests
  confirming SDXL is now `True`/routes to the new loader, and that every OTHER
  `single_file_is_pipeline`-shaped family (hypothetically, since none exists today besides SDXL)
  with `whole_pipeline_gguf_supported` left at its `False` default still raises the original
  refusal unchanged — this gate change must not silently affect any other family.
- **Not automatable, manual verification owed** (matching this project's established pattern for
  AirLLM and Unified Memory core): an actual load of the real `offgrid-ai/pony-diffusion-v6-xl
  -GGUF` Q8_0 file on this machine's RTX 5080, confirming the assembled pipeline actually
  generates a coherent image, not just that the conversion pipeline runs without raising. The
  exact remote-header byte-budget (§4.2) should also be confirmed against the real file during
  this same manual pass, not just the 3 MB figure measured during this design's investigation.

## 8. Review Focus

Failure modes/inputs this spec implies but a task's own tests could still miss, most likely
first:

- **A different SDXL-family GGUF whose text-encoder-2 (CLIP-G) uses a different internal key
  prefix than `conditioner.embedders.1`** (some converters number embedders differently, or a
  model trained without the second text encoder at all, e.g. a hypothetical SD 1.5-shaped file
  sharing the `first_stage_model`/`model.diffusion_model` prefixes but only ONE `conditioner`
  embedder) — the §4.2 signature's "all three must be present" rule should correctly exclude an
  SD1.5-shaped file (which has no `conditioner.embedders.*` prefix at all, using a bare
  `cond_stage_model.*` instead), but this needs an explicit test, not just a reasoned argument.
- **A quant type `gguf.quants.dequantize` doesn't recognize** silently producing garbage instead
  of refusing — needs an explicit test asserting a clean refusal for an unrecognized ggml type,
  not just happy-path coverage of F16/Q8_0.
- **The remote byte-range budget being too small for a larger/future SDXL-shaped GGUF** (more
  tensors than this specific 2513-tensor file, e.g. a future higher-resolution or larger-VAE
  variant) silently returning "no verdict" rather than a wrong one — already fails open by
  design (§4.2), but worth an explicit test with a tensor count larger than what 3 MB covers,
  confirming it still fails open rather than parsing a truncated/corrupt tensor-info table into
  a wrong signature match.
- **A GGUF matching the content signature that ISN'T actually SDXL-compatible** (e.g. a
  differently-shaped UNet with a different number of down/up blocks than `base_repo`'s config
  expects) — the converter functions should raise on a shape mismatch rather than silently
  truncating/padding; confirm this is the real behavior (via the synthetic-GGUF unit tests
  deliberately using a wrong shape for one tensor), not assumed.
- **Listing performance**: the picker's new remote content-signature check (§4.2) adds a
  byte-range request to every `.gguf` file whose name doesn't match an existing family alias,
  on every listing refresh — confirm this doesn't meaningfully slow down a listing page with
  many unrelated non-SDXL GGUF repos in it (the existing chat-refusal remote check already pays
  a similar cost per load attempt, not per listing render, so this is a new call site, not just
  a reused one).
