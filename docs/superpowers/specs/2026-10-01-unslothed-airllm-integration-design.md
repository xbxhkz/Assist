# AirLLM Integration — Design

**Status:** Approved by conversational design review, 2026-10-01. This is piece 3 of 3 from the
user's pasted "unified AI memory" master build prompt: Task Continuity Engine (shipped,
`unslothed-release` @ `549eae1b`) → Unified Memory core (shipped, `unslothed-release` @
`f0027a33`) → **this piece**.

**Repo:** `C:\Users\Admin\unsloth`, `studio/backend`. Base branch: `unslothed-release`.

**Platform constraint (overrides anything in the master prompt that assumes Linux):** this is
built for exactly one machine — Windows 11, NVIDIA CUDA 13.0, sm_120 (RTX 5080 Laptop, 16GB
VRAM, 64GB system RAM). Do not design for portability, other accelerators, or other OSes.

## 1. Problem

Today Studio serves chat/text models two ways, both inside a dedicated subprocess
(`core/inference/orchestrator.py`'s `InferenceOrchestrator` spawns and manages it;
`core/inference/inference.py`'s `InferenceBackend` is what runs inside it): GGUF via
`llama-server` (a separate native process the orchestrator also manages,
`core/inference/llama_cpp.py`), and HF `transformers`-based safetensors loading
(`InferenceBackend.load_model()`, which already imports `AutoProcessor`,
`TextIteratorStreamer`, and runs `transformers.pipeline` for several model types). Both paths
require the model to fit resident in VRAM (with whatever quantization/offload each path
supports) — there is no way to serve a text model too large for this machine's 16GB VRAM even
after aggressive quantization.

[AirLLM](https://github.com/lyogavin/airllm) (installed: version 4.0.0, confirmed via
`pip show airllm`) solves exactly this for autoregressive transformer decoders: it streams one
transformer layer at a time from disk to GPU, runs it, and discards it, so VRAM only ever needs
to hold one layer's weights plus the KV cache — not the whole model. Its own description:
"runs 70B large language models on a single 4GB GPU without quantization, distillation or
pruning." This is dramatically slower than normal resident inference (every forward pass
re-reads layers from disk/RAM), so it is a deliberate last-resort path, not a replacement for
either existing one.

## 2. What already exists (do not duplicate)

Investigated before designing this, per this project's standing reconciliation discipline:

- **`InferenceOrchestrator`/`InferenceBackend`'s subprocess lifecycle** (spawn, shutdown,
  command dispatch, streaming) is fully built and battle-tested across the two existing chat
  paths. This piece adds a third *loading mode* inside the same subprocess — it does not spawn
  a new subprocess type, and does not touch `InferenceOrchestrator`'s spawn/shutdown machinery
  at all.
- **`gpu_arbiter.py`'s `CHAT` ownership and `_evict_chat()`** already kill the whole chat
  subprocess on eviction, regardless of which of the two existing paths was active. This piece
  needs no change here: an AirLLM-loaded model is just a third thing that can be running inside
  the same subprocess `_evict_chat()` already knows how to kill.
- **`core/inference/memory_residency.py` (the just-shipped park/restore coordinator) and
  `core/inference/diffusion_memory.py`'s offload-tier system are both diffusion/video-only and
  irrelevant here.** Chat has never used either — its VRAM story has always been "the whole
  subprocess, resident or not, killed on eviction" — and AirLLM doesn't change that; it only
  changes what "resident" means for one single-layer-at-a-time load.
- **llama.cpp's "ctx auto-fit to GGUF+VRAM" logic is NOT reused.** It assumes the model's own
  weights compete with the KV cache for the same VRAM budget — true for GGUF/safetensors
  resident loading, false for AirLLM, where the model's weights are (almost) never resident.
  Reusing that estimator would badly under-size AirLLM's context. This piece needs its own,
  much simpler sizing heuristic (§4.3).

## 3. Non-goals

- **AirLLM is never auto-selected.** It is dramatically slower than either existing path (every
  forward pass re-reads layers from disk/RAM), so a user must explicitly choose it for a given
  load, knowing the tradeoff. No size-vs-VRAM auto-fallback heuristic is built.
- **Not a diffusion/video feature.** AirLLM is specifically a layer-streaming loader for
  autoregressive transformer decoders (confirmed via its exported class list: `AirLLMLlama2`,
  `AirLLMQWen2`, `AirLLMMistral`, etc. — all text-generation architectures). It has no role in
  image or video generation, and nothing in this piece touches `diffusion.py`/`video.py`.
- **No new subprocess type, no new arbiter owner.** AirLLM loads inside the existing chat
  subprocess under the existing `CHAT` owner.
- **No LoRA support for AirLLM loads in this piece.** The existing HF orchestrator path already
  has its own LoRA handling; AirLLM has a separate `AirLLMLoRA`/`AirLLMLoRAQwen4Exp` class
  family that is a materially different integration surface. Deferred — v1 is base-model
  inference only.
- **No multi-GPU layer distribution.** This machine has one GPU. `device` is always a single
  CUDA ordinal string.

## 4. Architecture

Three additive pieces, all inside the existing chat subprocess machinery:

### 4.1 Installing AirLLM into the Studio venv

`airllm` is currently installed in SYSTEM Python (`C:\Python314`, confirmed via
`python -c "import airllm; print(airllm.__file__)"` →
`C:\Python314\Lib\site-packages\airllm\__init__.py`) with CPU-only torch (`2.11.0+cpu`), NOT in
the Studio venv (`C:\Users\Admin\.unsloth\studio\unsloth_studio`, which has the correct
`torch==2.10.0+cu130`). It must be installed into the Studio venv before any of this piece's
code can run.

**Verified safe to install with `--no-deps`:** `airllm==4.0.0`'s declared dependencies
(`pip show airllm`) are `accelerate, huggingface-hub, safetensors, scipy, sentencepiece, torch,
tqdm, transformers`. Every one of these except `torch` is already present in the Studio venv at
a modern, almost-certainly-compatible version (confirmed via `pip list` in the Studio venv:
`accelerate 1.15.0`, `safetensors 0.8.0`, `scipy 1.18.1`, `sentencepiece 0.2.2`, `tqdm 4.70.1`,
`transformers 5.17.0`). `torch` is the one dependency that must NOT be touched — a plain
`pip install airllm` (without `--no-deps`) risks pulling in a generic CPU-only `torch` wheel
from PyPI and silently downgrading/replacing the pinned CUDA build, exactly the kind of
dependency landmine this project has hit before with other ML packages. The install command is
therefore:
```
C:\Users\Admin\.unsloth\studio\unsloth_studio\Scripts\python.exe -m pip install airllm==4.0.0 --no-deps
```
`bitsandbytes` (needed only for the optional `compression` parameter, §4.2) is ALSO already
present in the Studio venv (`0.50.2`, confirmed via `pip show`) — no action needed there.

This is a manual, one-time environment step, not something the implementation plan automates
(matching how this project has handled other one-off environment fixes) — the plan's first task
should perform it and verify with a real `import airllm` inside the Studio venv before writing
any integration code, since compatibility between `airllm==4.0.0`'s internals and
`transformers==5.17.0`/`accelerate==1.15.0` (both newer than what AirLLM was likely tested
against) is unverified until actually imported and exercised.

### 4.2 `InferenceBackend.load_model()` — the AirLLM branch

**Correction from the brainstorming conversation**: `load_model()`'s existing dispatch is
NOT based on a loose "engine" or "model_kind" string — it takes a `config: ModelConfig`
(`utils/models/model_config.py:3615`) whose load path is chosen by boolean discriminator
fields: `is_gguf`, `is_lora`, `is_vision`, `is_audio`, etc. (confirmed by reading the real
dataclass, not assumed). The consistent, idiomatic way to add this selection is a new field on
the SAME dataclass, `is_airllm: bool = False`, following the exact established pattern rather
than inventing a different kind of discriminator. `load_model()`'s own dispatch (confirmed at
`core/inference/inference.py:331`, which also confirms this method is Unsloth's own loading
wrapper — note the `load_in_4bit` parameter and the GGUF-specific `max_seq_length<=0` handling
comment, not bare HF `transformers.AutoModelForCausalLM`) gains a branch checking
`config.is_airllm` before its existing load-path selection, with the AirLLM path entirely
bypassing Unsloth's own `FastLanguageModel`-style loading (AirLLM is a wholesale alternative to
it for this one case, not a variant of it). Explicitly NOT inferred from file size, repo
metadata, or anything else — always a caller-set flag, matching §3's "never auto-selected"
non-goal.

When selected:

```python
import airllm

model = airllm.AutoModel.from_pretrained(
    model_local_path_or_repo_id,      # local path or HF repo id, same as existing loads already resolve
    device = f"cuda:{ordinal}",        # this machine's single GPU; never omit the ordinal
    max_seq_len = resolved_max_seq_len,  # from §4.3, NEVER AirLLM's own tiny 512 default
    compression = compression,         # "4bit" | "8bit" | None, caller-supplied; requires bitsandbytes
                                        # (confirmed present) -- AirLLM itself raises ImportError if
                                        # compression is requested and bitsandbytes can't be imported
    hf_token = hf_token,
    # layer_shards_saving_path intentionally left at AirLLM's own default ("next to the model
    # cache", per its own docstring) -- no new cache-location convention needed for v1.
)
```
Note from AirLLM's own docstring: `compression` disables `prefetching` when both are set
("prefetching is not supported together with compression for now; disabling prefetching") —
this is AirLLM's own internal behavior, not something this integration needs to replicate or
guard against, just be aware of (slower generation when compression is on, already true of
AirLLM's own design).

**The first load of a given model triggers AirLLM's own one-time layer-splitting
preprocessing** (writing per-layer shard files to disk, cached for subsequent loads). This can
take a real amount of time for a large model. AirLLM's own progress-reporting surface for this
step is unverified at spec-writing time — the implementation plan's own task for this should
investigate (read `airllm_base.py`'s splitting code, check for a callback/logging hook) and
design around whatever is actually available, rather than this spec assuming a specific
mechanism. At minimum, the existing load-progress UI must not go stale/stuck-looking during this
step even if fine-grained progress isn't available.

### 4.3 Context-length sizing (new, AirLLM-specific)

A new, dedicated sizing function — NOT a reuse of any existing GGUF/safetensors context-fit
logic (§2). AirLLM's VRAM budget per forward pass is approximately:
`(one layer's weight size) + (KV cache for max_seq_len tokens) + (fixed overhead: embeddings,
lm_head, activations, CUDA context)`. The implementation plan's own task should derive a
concrete formula from:
- the model's hidden size / layer count / attention head config (available from the HF
  `config.json` before any weights load, the same way existing load-planning code already reads
  config metadata without downloading full weights),
- a conservative estimate of one transformer layer's resident size (weights for one decoder
  block, at the model's native dtype unless `compression` is set),
- free VRAM at request time (reuse this project's existing free-VRAM snapshot helper — the same
  one `diffusion_memory.py`'s `snapshot_device_memory`-style logic already uses — do not
  reinvent a second one).

This function's job is specifically to compute a `max_seq_len` far larger than what would fit
under normal resident loading of an equivalently-sized model, since that's the whole point of
this path. The plan's Review Focus should include a test proving this (a model whose full
weights would never fit 16GB resident still gets a non-trivial `max_seq_len`, not AirLLM's
tiny 512 default and not zero).

## 5. Data flow (worked example)

1. User explicitly selects AirLLM for a chat load, naming a repo id or local path.
2. The route (or `InferenceBackend.load_model()` itself, wherever existing context-fit sizing
   already happens for the other two paths — match that location) calls the §4.3 sizing
   function with the model's config metadata and a free-VRAM snapshot, getting back a
   `max_seq_len`.
3. `InferenceBackend.load_model()`'s AirLLM branch builds the model via
   `airllm.AutoModel.from_pretrained(...)` (§4.2). First-ever load of this model triggers the
   one-time layer-split; subsequent loads read the cached shards directly.
4. Generation requests reuse the subprocess's existing per-token streaming loop, calling
   `model.generate(**kwargs)` — the exact kwarg shape AirLLM expects/returns needs confirming
   against the installed version at implementation time (its own signature is the generic
   `generate(self, *args, **kwargs)`, suggesting HF-compatible pass-through, but this must be
   verified empirically, not assumed).
5. Eviction: unchanged. `gpu_arbiter._evict_chat()` kills the whole subprocess exactly as today,
   regardless of which of the three load paths was active.

## 6. Error handling

- **Compression requested without `bitsandbytes` importable in the subprocess**: AirLLM itself
  raises `ImportError` for this. Catch it and surface a clean 4xx with a clear message, the same
  pattern this project already uses elsewhere for "refuse before the slow work starts" cases —
  do not let this propagate as a raw subprocess crash. (In practice `bitsandbytes` is already
  installed in the Studio venv, so this is a defensive catch, not an expected path.)
- **`airllm.NotEnoughSpaceException`** (AirLLM's own exception, confirmed exported from the
  package, raised during layer-splitting if disk space is insufficient): same clean-4xx
  treatment, not a raw crash.
- **A requested context length the free-VRAM snapshot genuinely can't support**: refuse before
  starting the load, matching this project's existing "refuse an explicit precision this host
  can never honor BEFORE the load starts" pattern (already used in both existing chat/diffusion
  paths).
- **Mid-generation failures** (OOM, a corrupted shard file, etc.): no new handling needed beyond
  what already exists for subprocess-level failures in the other two paths, since this runs
  inside the same subprocess and the same crash/restart machinery already covers it.

## 7. Testing

Per this project's standing test discipline (no real model, GPU, or subprocess in tests):

- The AirLLM branch in `InferenceBackend.load_model()` tested with `airllm.AutoModel` mocked
  entirely (the module itself, or at minimum `from_pretrained`) — confirms the branch is reached
  only on explicit selection, confirms the right kwargs are passed through (especially that
  `max_seq_len` is NEVER left at AirLLM's own 512 default when this path is selected), confirms
  `ImportError`/`NotEnoughSpaceException` map to the clean 4xx described in §6 rather than
  propagating raw.
- §4.3's sizing function tested as a pure unit (given fake config metadata + a fake free-VRAM
  number, returns the expected `max_seq_len`) — no real model, no real VRAM query.
- The install step (§4.1) is NOT unit-testable and is not part of the automated test suite —
  it's a one-time manual environment action, verified once at implementation time by actually
  importing `airllm` inside the Studio venv.
- The real layer-splitting preprocessing step and a real generation round-trip are NOT
  unit-testable in this environment (no GPU access from the test runner) — these need a manual
  smoke test on real hardware before this piece is considered genuinely working, matching the
  same "manual verification owed" pattern already recorded for the Unified Memory core piece's
  own image→video→image round-trip.

## 8. Review Focus

Failure modes/inputs this spec implies but a task's own tests could still miss, most likely
first:

- **`max_seq_len` silently falling back to AirLLM's own 512-token default** if the sizing
  function (§4.3) isn't actually wired into the `from_pretrained(...)` call, or returns `None`
  on some code path. A context of 512 tokens is barely usable for a real conversation — this
  would make the whole feature technically "working" (loads, generates) while being useless in
  practice. Needs an explicit test that the computed value, not the library default, is what
  actually reaches `AutoModel.from_pretrained`.
- **The `torch` dependency getting silently touched during the `airllm` install**, even with
  `--no-deps` specified correctly in the plan, if a future re-install or dependency resolution
  step (e.g. an unrelated `pip install -U` elsewhere in the project's setup) re-triggers a
  generic dependency resolution that picks up `airllm`'s undeclared-but-real `torch`
  requirement. Worth a one-line note in whatever environment-setup documentation this project
  already maintains, not necessarily a test.
- **Compression (`4bit`/`8bit`) silently producing garbage output** rather than failing loudly,
  if `bitsandbytes`'s actual installed version (`0.50.2`) has any incompatibility with
  `airllm==4.0.0`'s expected bitsandbytes API — this can only really be caught by the manual
  hardware smoke test (§7), not a unit test, but the smoke test should explicitly exercise
  compression at least once, not just the uncompressed default path.
- **A model whose config implies a layer count/hidden size this machine genuinely cannot fit
  even ONE layer of** (e.g. an absurdly wide model) — the sizing function should recognize this
  case and refuse cleanly (§6's "refuse before the slow work starts" pattern) rather than
  returning a nonsensical negative or zero `max_seq_len` that only fails deep inside AirLLM's
  own loading code.
- **Re-loading the same model after an eviction** should read the ALREADY-SPLIT shard files
  (from `layer_shards_saving_path`) rather than re-running the slow one-time split — this is
  AirLLM's own claimed caching behavior, but worth an explicit task-level check (e.g. timing a
  second load, or checking AirLLM's own internals for how it detects "already split") rather
  than assuming the caching "just works."
