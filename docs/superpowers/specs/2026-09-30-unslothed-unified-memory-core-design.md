# Unified Memory Core (park/restore) — Design

**Status:** Approved by conversational design review, 2026-09-30. This is piece 2 of 3 from the
user's pasted "unified AI memory" master build prompt: Task Continuity Engine (shipped,
`unslothed-release` @ `549eae1b`) → **this piece** → AirLLM integration (not started).

**Repo:** `C:\Users\Admin\unsloth`, `studio/backend`. Base branch: `unslothed-release`.

**Platform constraint (overrides anything in the master prompt that assumes Linux):** this is
built for exactly one machine — Windows 11, NVIDIA CUDA 13.0, sm_120 (RTX 5080 Laptop, 16GB
VRAM, 64GB system RAM). Do not design for portability, other accelerators, or other OSes.

## 1. Problem

`core/inference/gpu_arbiter.py` gives Studio's three heavy GPU consumers (`CHAT`, `DIFFUSION`,
`VIDEO`) exclusive, mutually-evicting ownership of the one GPU: `acquire_for(owner)` fully
**unloads** whichever owner currently holds the GPU before handing it to the new one. For
diffusion and video, "unload" (`DiffusionBackend.unload()` / the video backend's `unload()`)
means `self._state = None` — the entire pipeline object (UNet/transformer, VAE, text encoders,
everything) is dropped and garbage-collected. The next time that owner is needed, it rebuilds
from scratch: re-read weights from disk, reconstruct the diffusers pipeline, re-run offload-policy
selection.

This makes switching back and forth between diffusion and video (a common workflow — generate an
image, then animate it, then generate another image) pay a full cold-load penalty on every switch,
even when nothing about the model changed and the switch-back happens seconds later.

## 2. What already exists (do not duplicate)

Investigated before designing this, per this project's standing reconciliation discipline:

- **`core/inference/diffusion_memory.py`** (1256 lines) already implements per-model VRAM↔RAM
  tiering **during active use**: from a footprint estimate and a free-memory snapshot, it picks
  an offload policy (`none` / `model` / `group` / `streaming` / `sequential`) and applies it via
  diffusers' own `enable_model_cpu_offload()` / `apply_group_offloading()` /
  `enable_sequential_cpu_offload()`. A single model larger than VRAM already streams its own
  weights between GPU and CPU per forward pass. **This piece does not touch that system** — it
  already solves "one oversized model spans VRAM+RAM," and building a parallel mechanism for the
  same job would fight it for placement decisions.
- **`core/inference/media_keepwarm.py`** already does opt-in idle-timeout full unload for
  diffusion/video — a separate, simpler TTL mechanism, unrelated to arbiter-triggered eviction.
  Unaffected by this piece.
- **`core/inference/video.py`** explicitly reuses diffusion's offload/memory-planning layers
  ("hardware/optimisation layers are IMPORTED from the image stack unchanged"), so this piece's
  `park`/`restore` pair is added to both backends in parallel, not duplicated per-backend logic.

The actual gap — confirmed by reading `DiffusionBackend._unload_locked()`
(`core/inference/diffusion.py:5998`) — is that there is **no state between "fully resident" and
"fully destroyed."** `unload()` is the only teardown path that exists today, and it is called for
every arbiter eviction regardless of whether the caller actually wants the memory freed forever or
just wants the GPU back for something else temporarily.

## 3. Non-goals

- **Chat (`llama-server`) eviction is unchanged.** It runs as a separate OS subprocess, not
  in-process torch tensors Studio can move between devices; `_evict_chat()` keeps killing the
  subprocess exactly as today. Revisiting chat's story is AirLLM's job (a future piece), not
  this one.
- **The native `sd_cpp` engine is unchanged, for both diffusion and video.** For diffusion, on
  this CUDA machine it is never auto-selected (`diffusion_engine_router.py`'s policy: `backend ==
  "cpu" or (backend == "mps" and mps_enabled) or prefer_native` — a GPU backend is never
  native-eligible unless explicitly forced via `UNSLOTH_DIFFUSION_ENGINE=sd_cpp`). **Correction
  from the original design conversation:** video is NOT single-backend the way that conversation
  assumed — `_VideoLoadState.engine` (`core/inference/video.py:507,518`) is `"diffusers"` or
  `"sd_cpp"`, and at least the MiniMax-H3 family always loads via the native `sd_cpp` runtime
  (`core/inference/video.py:1912-1953`, `MiniMaxH3NativeRuntime`) **regardless of GPU/CPU** — its
  own comments describe normal CUDA-resident VRAM use ("20.5 GB DiT, 17 GB encoder"), so this is
  not an edge case on this machine, unlike diffusion's CPU/MPS-gated native path. Both diffusion's
  and video's evictors must branch on the active engine the same way; if either is `sd_cpp` at
  eviction time, eviction stays a direct `unload()`, exactly as today.
- **`park()` only engages when the loaded pipeline's offload policy is `OFFLOAD_NONE` (fully
  GPU-resident); any other policy falls back to a direct `unload()`.** Verified directly against
  the installed `diffusers==0.40.0.dev0`'s actual `DiffusionPipeline.to()` source: moving a
  **sequentially**-offloaded pipeline back to `cuda` (what `restore()` needs to do) raises
  `ValueError` outright ("not compatible with offloading"); a **group**-offloaded pipeline's
  offloaded modules are silently *skipped* by `.to()` entirely (never actually moved, so parking
  one would free little VRAM anyway, since those modules already live mostly off-GPU via
  diffusers' own streaming hooks); only **model**-offload (`enable_model_cpu_offload`) tolerates
  `.to()` both directions without raising, but restoring via `.to(device)` re-pins everything to
  GPU and silently discards that policy's own memory savings (diffusers itself warns: "memory
  gains from offloading are likely to be lost"). Rather than special-casing each of the four
  non-`NONE` policies' different failure/degradation shape, this piece treats "was offloaded at
  all" as "park is not the right tool here" and falls back to the existing `unload()` behavior —
  consistent with, and an extension of, the already-stated principle that a park that cannot
  cleanly help must degrade to today's safe behavior (§6). This is the common case on this
  machine anyway: 16GB VRAM comfortably fits most single diffusion/video loads at `OFFLOAD_NONE`,
  which is exactly where park/restore's fast-switch benefit is both available and safe.
- **No per-layer / per-block streaming within a single model.** That is the AirLLM piece's job.
  This piece's "block" granularity is coarse — a whole backend's pipeline is one park/restore
  unit, not individual transformer blocks. The `park`/`restore` **contract**, however, is designed
  so AirLLM can later reuse it at finer granularity without a redesign (see §7).
- **No NVMe tier.** Only VRAM and system RAM, per the user's explicit build-order decision.
- **No new UI.** RAM-budget is a setting (see §4.2), not a new panel.

## 4. Architecture

Three additive pieces:

### 4.1 `park()` / `restore()` on the backends

New methods on `DiffusionBackend` (`core/inference/diffusion.py`) and the video backend
(`core/inference/video.py`), sibling to their existing `unload()` / `status()`, using the same
locking discipline `unload()` already has (`self._lock` / `self._generate_lock`, cancel any
in-flight generation first — copy `unload()`'s existing cancellation preamble verbatim rather than
re-deriving it).

```python
def park(self) -> bool:
    """If state.offload_policy != OFFLOAD_NONE, returns False immediately (caller falls back to
    unload() -- see the Non-goals entry on why park only engages for a fully-resident pipeline).
    Otherwise moves the pipeline's modules to CPU (state.pipe.to("cpu")) and clears the CUDA
    cache, WITHOUT dropping self._state -- unlike unload(), the pipeline object survives, ready
    for a fast restore(). Sets a parked flag on state. Returns True on success."""

def restore(self) -> None:
    """Moves a parked pipeline back to its recorded device (state.pipe.to(state.device)) and
    clears the parked flag. Since park() only ever engages at OFFLOAD_NONE, there is no offload
    policy to reapply -- restore is a plain device move, nothing more. Raises if called when
    nothing is parked, or if the move fails (e.g. CUDA OOM because something else grew VRAM
    usage while this was parked)."""
```

Both reuse `unload()`'s existing in-flight-generation cancellation preamble verbatim (lock,
`_cancel_event.set()`, `_teardown_waiters` fence, `_generate_lock` barrier) — a park that doesn't
wait out an active generation the way `unload()` does would move tensors a running forward pass
still reads.

### 4.2 `core/inference/memory_residency.py` (new module)

Owns the RAM-tier budget and LRU eviction policy. Deliberately ignorant of what a "pipeline" is —
it only ever calls callables the caller hands it, the same decoupling `gpu_arbiter._EVICTORS`
already uses:

```python
def park_owner(owner: str, park: Callable[[], bool], unload: Callable[[], None],
                footprint_mib: Optional[int]) -> None:
    """If footprint_mib is None (the backend could not size this load -- see the footprint-source
    note below), calls unload(owner) directly and returns; parking without knowing the cost is
    exactly what the RAM budget exists to prevent. Otherwise: if the new total parked footprint
    would exceed the configured RAM budget, first calls unload() (a full teardown, not a park) on
    whichever OTHER currently-parked owner has been parked longest, freeing its footprint. If that
    forced eviction also fails, calls unload(owner) directly and returns (§6). Then calls park(owner).
    If park() returns False (not applicable -- e.g. the pipeline wasn't at OFFLOAD_NONE) or raises,
    calls unload(owner) directly. This function's own contract is therefore: on return, owner is
    EITHER parked (tracked here) OR fully unloaded -- never left ambiguous."""

def restore_owner(owner: str, restore: Callable[[], None]) -> None:
    """Calls restore() and clears owner's parked bookkeeping. Raises if owner is not parked."""

def is_parked(owner: str) -> bool: ...
def parked_footprint_mib() -> int: ...
```

RAM budget: a new setting, default 28 GiB (chosen: comfortably under 64 GiB total, leaving
headroom for Studio's own process, the OS, and whatever else is running; concrete default is an
implementation-time judgment call, not a hard requirement of this spec — the plan may pick a
different default in the 24-32 GiB range the design conversation settled on, provided it's
justified against actual model sizes). Surfaced through whatever Settings mechanism already hosts
`memory_mode` and similar per-project knobs — the plan should follow that existing pattern rather
than inventing a new settings surface.

### 4.3 `gpu_arbiter.py` — evictor branch

```python
def _evict_diffusion() -> None:
    from core.inference.diffusion_engine_router import active_engine_name, get_active_diffusion_engine
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    engine = get_active_diffusion_engine()
    if active_engine_name() == ENGINE_SD_CPP:
        engine.unload()   # unchanged: subprocess-based, park/restore not applicable
        return
    from core.inference import memory_residency
    memory_residency.park_owner(DIFFUSION, engine.park, engine.unload, engine.resident_footprint_mib())
```

`_evict_video` mirrors this WITH the same kind of engine branch (corrected from the original
design conversation, which assumed video has only one backend — it doesn't; see §3): check
`get_video_backend().status()["engine"]` (already exposed, `core/inference/video.py:6042`) and
route to `park_owner` only when it's `"diffusers"`, direct `unload()` when it's `"sd_cpp"`.

**Footprint source:** `resident_footprint_mib()` is a new small method on each backend, not a new
estimator. Both backends already compute this at load time via the SAME shared function,
`plan_diffusion_memory()` (`core/inference/diffusion_memory.py`, imported directly into
`video.py` too — confirmed, not duplicated): its returned `MemoryPlan.estimates` dict carries
`"resident_required_mib"`, the total resident footprint the planner budgeted this load against
(`core/inference/diffusion_memory.py:605-615`). The plan should persist that one number onto
`_LoadState`/`_VideoLoadState` as a new `resident_mib` field, populated at each construction site
(`diffusion.py:4374`; `video.py:1941`, `4041`, `4790` — video constructs `_VideoLoadState` at three
sites, one per load path, all need the new field) from `plan.estimates.get("resident_required_mib")`.
A `None` value (the planner's own "unknown, stay resident" signal) means `resident_footprint_mib()`
returns `None` too; `park_owner`'s own contract (§4.2) already treats that as "do not attempt to
park," so the evictor itself needs no separate `None` check.

`park_owner`'s contract (§4.2) guarantees `owner` ends up either parked or fully unloaded on
return — it does not raise for any of the fallback cases (unsizeable footprint, over budget with
a failed forced eviction, park() inapplicable or failing). A genuinely unexpected exception (e.g.
a bug, not one of the anticipated fallback paths) still propagates normally; `acquire_for` runs
evictors under its own lock and does not swallow evictor exceptions today, and this piece does not
change that.

### 4.4 Load routes check parked state first

**Correction from the original design conversation:** the check does NOT belong in the route.
`routes/inference.py`'s diffusion-load handler is a single, already very large (~24000-line file)
function with extensive existing preflight/precision/arbiter-ordering logic; `routes/video.py`'s
mirrors it. Neither route currently has, or needs, an "is this identical to what's already
resident" concept — that decision belongs to the backend that owns `_LoadState`. The check goes
into `DiffusionBackend.begin_load()` (`core/inference/diffusion.py:1837`) and the video backend's
equivalent entry point, BEFORE either spawns its slow background load thread (`self._run_load`,
`core/inference/diffusion.py:1940`) — restoring skips that thread entirely, which is the whole
point. **This means routes/inference.py and routes/video.py need NO changes at all** — a smaller
footprint than the original design assumed, and the intricate route logic is untouched.

```python
# Inside begin_load(), after validate_load_request()/assert_precision_available(), before the
# "with self._lock: ... threading.Thread(target=self._run_load, ...)" block:
from core.inference import memory_residency

if memory_residency.is_parked(DIFFUSION):
    if self._matches_parked_identity(
        repo_id = repo_id, gguf_filename = gguf_filename, base_repo = base_repo,
        model_kind = model_kind, transformer_quant = transformer_quant,
        text_encoder_quant = text_encoder_quant, cpu_offload = cpu_offload,
        memory_mode = memory_mode, gpu_ordinal = gpu_ordinal, loras = loras,
    ):
        try:
            memory_residency.restore_owner(DIFFUSION, self.restore)
            return self.status()
        except Exception:
            memory_residency.forget_parked(DIFFUSION)  # clears stale bookkeeping; falls through below
    else:
        memory_residency.forget_parked(DIFFUSION)  # a different model was requested; the parked one is moot -- drop it
        self.unload()
# ...existing cold-load path (spawn self._run_load) continues unchanged...
```

**Identity check (`_matches_parked_identity`, a new small private method):** compares the
requested load's parameters against the values already stored on `self._state` (most already
exist as `_LoadState` fields: `repo_id`, `gguf_filename`, `base_repo`, `kind`, `transformer_quant`,
`text_encoder_quant`, `cpu_offload`, `memory_mode`, `gpu_ordinal`) plus a LoRA check
(`_active_lora_pairs(self._state.pipe) == loras` — LoRAs are tracked as an attribute on the
pipeline object itself, `core/inference/diffusion.py:5924`, so they travel with `state.pipe`
through park/restore automatically; the check only needs to confirm the request didn't ask for
DIFFERENT ones). **Conservative by design: any field that doesn't match, or can't be compared,
means "not the same model" — restore only happens on a confirmed exact match.** Runtime-only knobs
that don't change which weights are resident (`attention_backend`, `speed_mode`,
`transformer_cache`, `gpu_ids` beyond the resolved `gpu_ordinal`) are deliberately excluded from
the identity check — a restore keeps the parked pipeline's OWN settings for these rather than
picking up newly requested ones, a documented v1 limitation, not a correctness bug (the weights
themselves are still guaranteed correct).

**`memory_residency.forget_parked(owner)`** (a small addition to the module's interface from §4.2,
alongside `park_owner`/`restore_owner`/`is_parked`/`parked_footprint_mib`): clears an owner's
parked bookkeeping without calling `restore()` or `unload()` itself — the caller has already
decided what to do with the actual pipeline (call `self.unload()` for a moot identity mismatch, or
nothing further for a failed restore that's about to fall through to a fresh cold load, which will
overwrite `self._state` itself).

This same shape (check `is_parked`, compare identity, restore-or-unload-and-cold-load) applies to
the video backend's own load entry point, using the video backend's own equivalent fields.

## 5. Data flow (worked example)

1. Diffusion generates → route calls `acquire_for(DIFFUSION, register=_begin_load)` exactly as
   today; `_begin_load` → `DiffusionBackend.begin_load(...)`. No current owner to evict;
   `is_parked(DIFFUSION)` is false; normal cold load (spawns `_run_load`).
2. User switches to video → `acquire_for(VIDEO, register=...)`. Arbiter evicts DIFFUSION →
   `_evict_diffusion()` → diffusers engine, `OFFLOAD_NONE` → `memory_residency.park_owner(DIFFUSION,
   ...)` → diffusion's pipeline moves to CPU RAM, object stays alive, tracked with footprint +
   timestamp. Ownership transfers to VIDEO; video's `begin_load` checks `is_parked(VIDEO)` → false
   (first use) → cold load.
3. User switches back to diffusion, requesting the SAME model/config as before → `acquire_for
   (DIFFUSION, ...)`. Arbiter evicts VIDEO → parked the same way. Ownership transfers to DIFFUSION;
   `begin_load` checks `is_parked(DIFFUSION)` → true → `_matches_parked_identity(...)` → true →
   `restore_owner` → pipeline moves CPU→GPU (no offload policy to reapply — park only ever engaged
   because this load was at `OFFLOAD_NONE`, §3), no disk read, no pipeline reconstruction, `_run_load`
   never spawned. (Had the user requested a DIFFERENT model instead, the identity check would fail,
   `forget_parked` + `unload()` would drop the stale parked pipeline, and a normal cold load would
   proceed — never serving the wrong weights.)
4. If step 2 or 3's park would push total parked RAM over the configured budget,
   `memory_residency` first fully `unload()`s whichever *other* parked owner has been parked
   longest before parking the new one — budget pressure degrades gracefully to today's full-reload
   behavior for the oldest parked owner, never to an OOM.

## 6. Error handling

All of the following are `memory_residency.park_owner`'s job (§4.2) — the evictor (§4.3) calls it
unconditionally and does not itself implement any of this fallback logic:

- `park()` returning `False` (not applicable — offload policy wasn't `OFFLOAD_NONE`) or raising
  both fall back to a full `unload()` of the same owner — parking is an optimization over today's
  behavior; a park that cannot cleanly help must degrade to today's safe behavior (fully freed),
  never leave a backend half-moved between devices.
- An unsizeable footprint (`footprint_mib is None`) skips parking entirely and unloads directly —
  parking without knowing the cost is exactly what the RAM budget exists to prevent.
- The forced LRU eviction of an older parked owner (freeing room for a new park) failing aborts
  parking the NEW owner too, falling back to a direct `unload()` of the new owner, rather than
  exceeding the RAM budget or leaving inconsistent tracked state (an owner marked parked whose
  forced neighbor eviction actually failed).
- `restore()` raising (e.g. CUDA OOM because something else grew VRAM usage while this was parked)
  propagates out of `restore_owner` to the caller (`begin_load`, §4.4), which calls
  `forget_parked` and falls through to the normal cold-load path — never a bare error surfaced to
  the user when a cold load would have worked. This is the one failure mode NOT absorbed inside
  `memory_residency` itself, because recovering from it means re-entering the load path, which
  `memory_residency` has no access to.

## 7. Forward compatibility (for the future AirLLM piece, not built here)

The `park(owner) -> restore(owner)` contract in `memory_residency.py` operates on whole-backend
granularity in this piece, but nothing about its interface assumes that — `park_owner`/
`restore_owner` take arbitrary callables and an opaque footprint number. A later piece that wants
finer-than-whole-pipeline granularity (e.g. individual transformer blocks within one model) can
register each block as its own "owner" against the same budget/LRU tracker, or build a parallel
tracker reusing the same pattern, without this piece's code needing to change. This spec does not
commit to that reuse happening — it only avoids closing the door on it.

## 8. Testing

Per this project's standing test discipline (no real model, GPU, or subprocess in tests; every
guard needs a mutation-demonstrated negative control):

- `memory_residency` tested in isolation with fake owners and injected `park`/`unload`/`restore`
  callables (matching `test_gpu_arbiter.py`'s existing pattern of recorder lambdas, not real
  backends). Covers: budget-exceeded triggers LRU eviction of the oldest parked owner;
  `is_parked` reflects park/restore transitions; a failed forced-eviction aborts the new park.
- `gpu_arbiter`'s evictor branch tested the same way `test_gpu_arbiter.py` already tests eviction
  sequencing today — monkeypatch `_EVICTORS`/the engine-name resolver, assert `park_owner` is
  called for the diffusers engine and the existing direct-`unload()` path is unchanged for
  `sd_cpp`.
- `DiffusionBackend.park()`/`.restore()` tested against a fake pipeline object (a stub `.to(device)`
  that records calls, matching the existing `backend._state = _LoadState(object(), ...)` pattern
  already used in `test_diffusion_backend.py`) — at `OFFLOAD_NONE`, `park()` calls `.to("cpu")` and
  leaves `_state` non-`None`, and `restore()` calls `.to(device)`; at any other offload policy,
  `park()` returns `False` without calling `.to()` at all; a `.to()` raising during `park()` falls
  back to real `unload()` (state becomes `None`).
- Video backend: same three tests, mirroring diffusion's.
- `begin_load`'s parked-state handling tested directly on each backend (mocked, no real model):
  `is_parked(owner) == True` plus a matching identity causes `restore()` to be called and
  `_run_load`/the background load thread to never spawn; a mismatched identity (different
  `repo_id`, or different `loras`) causes `forget_parked` + `unload()` + a normal cold load instead
  of a restore; a `restore()` failure falls through to the cold-load path rather than raising to
  the caller.

## 9. Review Focus

Failure modes/inputs this spec implies but a task's own tests could still miss, most likely first:

- **A pipeline loaded under a non-`OFFLOAD_NONE` policy must fall through to a normal `unload()`,
  never attempt a `.to()` move.** Verified directly against `diffusers==0.40.0.dev0`: `.to(cuda)`
  on a sequentially-offloaded pipeline raises `ValueError` outright, and group-offloaded modules
  are silently skipped by `.to()` (never actually moved). Without this gate, `park()` built
  against a simplistic fake-pipeline test stub (whose `.to()` never raises) would look correct in
  tests while being broken — or, worse, silently no-op-ing — against every real oversized-model
  load. This is the single most likely place a task's own tests pass while the real behavior is
  wrong, precisely because the common test double doesn't reproduce diffusers' actual constraint.
- **Restoring a parked pipeline when the user actually requested a DIFFERENT model.** The highest-
  cost mistake this piece could make: `is_parked(owner) == True` is not sufficient on its own to
  restore — `_matches_parked_identity` (§4.4) must genuinely compare the request against what's
  parked, and any uncertainty must fall back to a cold load, never a restore. A task whose test
  only exercises "same model, restore happens" and never "different model requested while one is
  parked, cold load happens instead (not the stale one served)" has not actually verified the
  identity check does anything — an inert-control risk this project has hit 17+ times before, and
  the exact shape of it here: a check that always passes regardless of whether it compares the
  right thing looks identical to a correct one in the "happy path only" test.
- **A park/restore race with a concurrent `acquire_for` for the SAME owner.** `gpu_arbiter`'s
  eviction runs under its own lock, but `park()`/`restore()` on the backend run under the
  backend's OWN lock, not the arbiter's — a slow park racing a fast re-acquire of the same owner
  needs to not corrupt `_state` or leave it in a device-inconsistent place. Needs an explicit test,
  not just "the arbiter serializes eviction" (it serializes *arbiter* calls, not backend-internal
  state transitions).
- **Parking an owner mid-generation.** `unload()`'s cancellation preamble handles this today;
  `park()` must reuse it exactly, not a simplified version — a park that doesn't wait out an
  active generation the way `unload()` does would move tensors a running forward pass still reads.
- **Restoring into a VRAM budget that's shrunk since parking** (another process, or a larger
  concurrent load, ate the headroom). `restore()` raising and falling back to cold `load()` (§6)
  is the specified behavior — a task's tests must actually exercise "restore fails because VRAM
  is now insufficient," not just "restore fails because of an injected generic exception."
- **The RAM budget being smaller than one model's footprint.** `park_owner`'s forced-eviction loop
  (§4.2) must terminate — if the new owner alone exceeds the whole budget, no amount of evicting
  other parked owners makes room, and this must fall back to a direct `unload()` of the new owner
  rather than looping or evicting everything for no benefit. (Also worth a quick check alongside
  this: `memory_residency`'s tracker is in-memory only, matching `gpu_arbiter`'s own `_owner`, so a
  fresh module state should report nothing parked — cheap to fold into the same task's tests.)
