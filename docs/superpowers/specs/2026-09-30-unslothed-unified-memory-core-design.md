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
- **The native `sd_cpp` / `sd-server` diffusion engine is unchanged.** On this CUDA machine it is
  never auto-selected (`diffusion_engine_router.py`'s policy: `backend == "cpu" or (backend ==
  "mps" and mps_enabled) or prefer_native` — a GPU backend is never native-eligible unless
  explicitly forced via `UNSLOTH_DIFFUSION_ENGINE=sd_cpp`). It is also subprocess-based like chat.
  If diffusion's active engine is `sd_cpp` at eviction time, eviction stays a direct `unload()`.
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
def park(self) -> None:
    """Move the resident pipeline's modules to CPU and clear the CUDA cache, WITHOUT dropping
    self._state -- unlike unload(), the pipeline object survives, ready for a fast restore()."""

def restore(self) -> None:
    """Move a parked pipeline back to its device and reapply the offload policy it had before
    parking. Raises if called when nothing is parked, or if the move fails (e.g. CUDA OOM
    because something else grew VRAM usage while this was parked)."""
```

`park()` must record enough of the pre-park state (device target, offload policy) for `restore()`
to reapply it — this is data already computed once at load time (the same policy
`diffusion_memory.py` selected), not re-derived.

### 4.2 `core/inference/memory_residency.py` (new module)

Owns the RAM-tier budget and LRU eviction policy. Deliberately ignorant of what a "pipeline" is —
it only ever calls callables the caller hands it, the same decoupling `gpu_arbiter._EVICTORS`
already uses:

```python
def park_owner(owner: str, park: Callable[[], None], unload: Callable[[], None],
                footprint_mib: int) -> None:
    """Calls park(). If the new total parked footprint would exceed the configured RAM budget,
    first calls unload() (a full teardown, not a park) on whichever OTHER currently-parked
    owner has been parked longest, freeing its footprint, before parking this one. If park()
    itself raises, propagates -- callers fall back to a direct unload() of THIS owner (see
    gpu_arbiter's evictor, §4.3)."""

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

(`_evict_video` mirrors this without the engine branch, since video has only one backend.)

**Footprint source:** `resident_footprint_mib()` is a new small method on each backend, not a new
estimator — it returns whatever `diffusion_memory.py`'s existing `estimate_gguf_resident_mib()` /
`estimate_safetensors_dense_mib()` already computed for THIS load's offload-policy selection
(`core/inference/diffusion.py:4875-4912` calls these today, as local variables — the plan needs to
persist that already-computed number onto `state` at load time, or recompute it from the same
already-resolved inputs `state` holds, whichever the implementer finds cleaner; either way, no new
footprint-estimation logic is written, only reuse of what load time already derives). The video
backend needs the equivalent, from whichever of its own load-time calculations already sizes the
resident model — the plan should locate and reuse that the same way, not introduce a second
estimation method for video.

If `memory_residency.park_owner` itself raises (e.g. the forced eviction of an older parked owner
also fails), the evictor's caller — `acquire_for`, which runs evictors under its own lock — must
not swallow that silently; let it propagate as `acquire_for` already does for any evictor
exception today (unchanged behavior, just confirming this new path doesn't need special handling).

### 4.4 Load routes check parked state first

The routes that currently call `gpu_arbiter.acquire_for(DIFFUSION, register=...)` /
`acquire_for(VIDEO, register=...)` (in `routes/inference.py`, `routes/video.py`) gain one check
in their `register` callback, before doing a cold load:

```python
from core.inference import memory_residency

if memory_residency.is_parked(DIFFUSION):
    memory_residency.restore_owner(DIFFUSION, engine.restore)
else:
    ...existing cold-load path...
```

If `restore_owner`/`engine.restore()` raises, the route catches it, clears the stale parked
bookkeeping (so a retry doesn't loop trying to restore something broken), and falls through to the
existing cold-load path — the same recovery a first-time load already has, entered from a
different place. This fallback is REQUIRED, not optional: a route that lets a restore failure
propagate as a hard error, when a cold load would have succeeded, is a regression this piece must
not introduce.

## 5. Data flow (worked example)

1. Diffusion generates → `acquire_for(DIFFUSION, register=load_cb)`. No current owner; `load_cb`
   runs a normal cold `load()`.
2. User switches to video → `acquire_for(VIDEO, register=video_load_cb)`. Arbiter evicts
   DIFFUSION → `_evict_diffusion()` → diffusers engine → `memory_residency.park_owner(DIFFUSION,
   ...)` → diffusion's pipeline moves to CPU RAM, object stays alive, tracked with footprint +
   timestamp. Ownership transfers to VIDEO; `video_load_cb` checks `is_parked(VIDEO)` → false
   (first use) → cold `load()`.
3. User switches back to diffusion → `acquire_for(DIFFUSION, ...)`. Arbiter evicts VIDEO → parked
   the same way. Ownership transfers to DIFFUSION; `load_cb` checks `is_parked(DIFFUSION)` → true
   → `restore_owner` → pipeline moves CPU→GPU, offload policy reapplied, no disk read, no
   pipeline reconstruction.
4. If step 2 or 3's park would push total parked RAM over the configured budget,
   `memory_residency` first fully `unload()`s whichever *other* parked owner has been parked
   longest before parking the new one — budget pressure degrades gracefully to today's full-reload
   behavior for the oldest parked owner, never to an OOM.

## 6. Error handling

- `park()` raising falls back to a full `unload()` of the same owner — parking is an optimization
  over today's behavior; a failed park must degrade to today's safe behavior (fully freed), never
  leave a backend half-moved between devices.
- `restore()` raising propagates to the caller (the load route), which clears the stale parked
  bookkeeping and falls back to a normal cold `load()` (§4.4) — never a bare error surfaced to the
  user when a cold load would have worked.
- `memory_residency`'s forced LRU eviction (freeing room for a new park) failing aborts parking the
  NEW owner too, falling back to a direct `unload()` of the new owner, rather than exceeding the
  RAM budget or leaving inconsistent tracked state (an owner marked parked whose forced neighbor
  eviction actually failed).

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
  already used in `test_diffusion_backend.py`) — `park()` calls `.to("cpu")` and leaves `_state`
  non-`None`; `restore()` calls `.to(device)` and reapplies the recorded offload policy; a `.to()`
  raising during `park()` falls back to real `unload()` (state becomes `None`).
- Video backend: same three tests, mirroring diffusion's.
- One test per load route confirming `is_parked(owner) == True` causes `restore()` to be called
  and the cold-load path to be skipped entirely (mocked, no real model); one confirming a
  `restore()` failure falls through to the cold-load path rather than raising to the caller.

## 9. Review Focus

Failure modes/inputs this spec implies but a task's own tests could still miss, most likely first:

- **A park/restore race with a concurrent `acquire_for` for the SAME owner.** `gpu_arbiter`'s
  eviction runs under its own lock, but `park()`/`restore()` on the backend run under the
  backend's OWN lock, not the arbiter's — a slow park racing a fast re-acquire of the same owner
  needs to not corrupt `_state` or leave it in a device-inconsistent place. Needs an explicit test,
  not just "the arbiter serializes eviction" (it serializes *arbiter* calls, not backend-internal
  state transitions).
- **Parking an owner mid-generation.** `unload()`'s cancellation preamble handles this today;
  `park()` must reuse it exactly, not a simplified version — a park that doesn't wait out an
  active generation the way `unload()` does would tear down tensors a running forward pass still
  reads.
- **Restoring into a VRAM budget that's shrunk since parking** (another process, or a larger
  concurrent load, ate the headroom). `restore()` raising and falling back to cold `load()` (§6)
  is the specified behavior — a task's tests must actually exercise "restore fails because VRAM
  is now insufficient," not just "restore fails because of an injected generic exception."
- **The RAM budget being smaller than one model's footprint.** `park_owner`'s forced-eviction loop
  (§4.2) must terminate — if the new owner alone exceeds the whole budget, no amount of evicting
  other parked owners makes room, and this must fall back to a direct `unload()` of the new owner
  rather than looping or evicting everything for no benefit.
- **`is_parked`/bookkeeping drift after a process restart.** `memory_residency`'s tracker is
  in-memory only (matches `gpu_arbiter`'s own `_owner` being in-memory); a restart must not leave
  stale "parked" state that `restore()` then fails against — worth an explicit test that a fresh
  `memory_residency` module state reports nothing parked.
