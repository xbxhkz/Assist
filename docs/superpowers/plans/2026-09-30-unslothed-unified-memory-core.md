# Unified Memory Core (park/restore) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `gpu_arbiter`'s diffusion/video owner switches park a resident, fully-GPU-loaded
pipeline to system RAM instead of destroying it, so switching back restores it in seconds instead
of a full cold reload from disk.

**Architecture:** Two backends (`DiffusionBackend`, the video backend) gain `park()`/`restore()`/
`resident_footprint_mib()` methods sibling to their existing `unload()`/`status()`. A new module,
`core/inference/memory_residency.py`, owns a RAM budget and LRU-evicts the longest-parked owner
when a new park would exceed it. `gpu_arbiter.py`'s two evictors route through it instead of
calling `unload()` directly, except when the active engine is the native `sd_cpp` runtime
(subprocess-based, unchanged). Each backend's own `begin_load()` checks whether it's currently
parked and whether the new request matches what's parked closely enough to restore instead of a
cold load — conservatively: any mismatch falls back to a cold load, never a restore.

**Tech Stack:** Python, PyTorch, diffusers 0.40.0.dev0 (installed version, verified directly
against its source for this design), pytest.

**Spec:** `C:\Users\Admin\odysseus\docs\superpowers\specs\2026-09-30-unslothed-unified-memory-core-design.md`

## Global Constraints

- Platform: Windows 11, NVIDIA CUDA 13.0, sm_120 (RTX 5080 Laptop, 16GB VRAM), 64GB system RAM.
  No portability/other-accelerator design.
- `park()` only engages when the loaded pipeline's `offload_policy == OFFLOAD_NONE`. Verified
  against the installed diffusers: `.to(cuda)` on a sequentially-offloaded pipeline raises
  `ValueError`; a group-offloaded pipeline's modules are silently skipped by `.to()` (never
  moved). Any other policy: `park()` returns `False`, caller falls back to `unload()`.
- `core/inference/diffusion_memory.py`'s existing per-model offload-tier system (`none`/`model`/
  `group`/`streaming`/`sequential`) is NOT touched by this plan. It already solves single-
  oversized-model VRAM/RAM tiering during active use; this plan is strictly about
  arbiter-triggered eviction between owners.
- Chat (`llama-server`) eviction in `gpu_arbiter.py` is UNCHANGED — `_evict_chat()` stays a
  direct subprocess kill.
- The native `sd_cpp` engine is UNCHANGED for both diffusion and video — subprocess-based,
  `park`/`restore` do not apply. Diffusion's evictor branches on
  `diffusion_engine_router.active_engine_name()`; video's branches on
  `get_video_backend().status()["engine"]`.
- `routes/inference.py` and `routes/video.py` are NOT modified by this plan — the restore
  decision lives inside each backend's own `begin_load()`, before it spawns its background load
  thread.
- Any uncertainty in matching a parked pipeline's identity against a new load request means
  "not a match" — restore only happens on a confirmed exact match. Serving the wrong model's
  weights is the worst failure mode this plan could introduce.
- Test discipline: no real model, GPU, or subprocess in any test. Match
  `tests/test_gpu_arbiter.py`'s pattern (monkeypatched `_EVICTORS` with recorder lambdas) and
  `tests/test_diffusion_backend.py`'s pattern (`backend._state = _LoadState(object(), ...)` fake
  pipeline stub). Every guard needs a negative control demonstrated to fail by mutation.
- NEVER run the full backend test suite — upstream GGUF fixtures fabricate files up to 40GB and
  have filled the disk before. Run only named test files, via
  `C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest <files> -q -p no:cacheprovider`.
- Commit before running any mutation control — never mutate uncommitted work.
- Keyword arguments use spaces around `=` (`foo(bar = 1)`), matching every file already touched
  in this plan.

## Review Focus

1. **A pipeline loaded under a non-`OFFLOAD_NONE` policy must fall through to `unload()`, never
   attempt `.to()`.** The single most likely place a task's own tests pass while the real
   behavior is wrong — a fake pipeline stub's `.to()` never raises the way a real
   sequentially-offloaded diffusers pipeline does. Task 2/3's tests must exercise both
   `OFFLOAD_NONE` (park proceeds) and at least one other policy (park refuses) explicitly.
2. **Restoring a parked pipeline when the user requested a DIFFERENT model.** The highest-cost
   mistake this plan could make. Task 5/6's tests must include a genuine mismatch case (different
   `repo_id`, and for diffusion, different `loras`) that results in a cold load, not just a
   matching-identity happy path — an identity check whose test only ever exercises "same model"
   is exactly the inert-control shape this project has shipped 17+ times before.
3. **A park/restore race with a concurrent `acquire_for` for the same owner.** `gpu_arbiter`'s
   eviction lock does not extend into the backend's own `_lock`/`_generate_lock` — a slow park
   racing a fast re-acquire of the same owner must not corrupt `_state`.
4. **The RAM budget being smaller than one model's footprint alone.** `park_owner`'s
   forced-eviction loop must terminate and fall back to a direct `unload()` of the new owner,
   never loop or evict everything for zero benefit.
5. **`restore()` failing after a successful park** (e.g. VRAM shrank in the meantime) must fall
   back to a cold load, not a bare error surfaced to the user.

---

### Task 1: `memory_residency.py` — RAM-tier budget and LRU coordinator

**Files:**
- Create: `studio/backend/core/inference/memory_residency.py`
- Create: `studio/backend/utils/memory_park_settings.py`
- Test: `studio/backend/tests/test_memory_residency.py`
- Test: `studio/backend/tests/test_memory_park_settings.py`

**Interfaces:**
- Produces: `park_owner(owner: str, park: Callable[[], bool], unload: Callable[[], None], footprint_mib: Optional[int]) -> None`, `restore_owner(owner: str, restore: Callable[[], None]) -> None`, `forget_parked(owner: str) -> None`, `is_parked(owner: str) -> bool`, `parked_footprint_mib() -> int`. `get_ram_park_budget_mib() -> int`, `set_ram_park_budget_mib(value) -> int`.

- [ ] **Step 1: Write the settings module first (it's the simpler piece), mirroring `utils/vram_budget_settings.py`'s exact pattern**

```python
# studio/backend/utils/memory_park_settings.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted RAM budget for parked (evicted-to-RAM, not destroyed) diffusion/video pipelines.

Same precedence and caching shape as utils/vram_budget_settings.py: a stored value wins, the
environment is a standalone startup default, the constant is the last resort.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

RAM_PARK_BUDGET_SETTING_KEY = "diffusion_ram_park_budget_mib"
RAM_PARK_BUDGET_ENV_VAR = "UNSLOTH_RAM_PARK_BUDGET_MIB"

# 64 GiB total on this machine; 28 GiB leaves headroom for Studio's own process, the OS, and
# whatever else is running. See spec section 4.2 for the reasoning.
RAM_PARK_BUDGET_MIN_MIB = 0
RAM_PARK_BUDGET_MAX_MIB = 61440  # 60 GiB -- never the whole 64GB, always some floor reserved
RAM_PARK_BUDGET_DEFAULT_MIB = 28672  # 28 GiB

_CACHE_TTL_S = 2.0
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, Any]] = {}
_generation: dict[str, int] = {}
_MAX_REREADS = 3


def _cached_setting(key: str) -> Any:
    stored = None
    for _attempt in range(_MAX_REREADS):
        with _cache_lock:
            hit = _cache.get(key)
            if hit is not None and time.monotonic() - hit[0] < _CACHE_TTL_S:
                return hit[1]
            generation = _generation.get(key, 0)
        try:
            from storage.studio_db import get_app_setting
            stored = get_app_setting(key, None)
        except Exception:
            return None
        with _cache_lock:
            if _generation.get(key, 0) == generation:
                _cache[key] = (time.monotonic(), stored)
                return stored
    return stored


def _invalidate(key: str) -> None:
    with _cache_lock:
        _cache.pop(key, None)
        _generation[key] = _generation.get(key, 0) + 1


def coerce_ram_budget_mib(value: Any) -> Optional[int]:
    """A RAM park budget in MiB, in [RAM_PARK_BUDGET_MIN_MIB, RAM_PARK_BUDGET_MAX_MIB], else None."""
    if isinstance(value, bool):
        return None
    try:
        mib = int(value)
    except (TypeError, ValueError):
        return None
    if not RAM_PARK_BUDGET_MIN_MIB <= mib <= RAM_PARK_BUDGET_MAX_MIB:
        return None
    return mib


def _env_budget_mib() -> Optional[int]:
    return coerce_ram_budget_mib(os.environ.get(RAM_PARK_BUDGET_ENV_VAR))


def get_ram_park_budget_mib() -> int:
    """Never raises and never returns a value outside the supported range."""
    stored = coerce_ram_budget_mib(_cached_setting(RAM_PARK_BUDGET_SETTING_KEY))
    if stored is not None:
        return stored
    from_env = _env_budget_mib()
    if from_env is not None:
        return from_env
    return RAM_PARK_BUDGET_DEFAULT_MIB


def set_ram_park_budget_mib(value: Any = None) -> int:
    """Store a budget, or clear it with None so env/default applies again."""
    from storage.studio_db import upsert_app_settings

    if value is None:
        upsert_app_settings({RAM_PARK_BUDGET_SETTING_KEY: None})
        _invalidate(RAM_PARK_BUDGET_SETTING_KEY)
        return get_ram_park_budget_mib()
    coerced = coerce_ram_budget_mib(value)
    if coerced is None:
        raise ValueError(
            f"RAM park budget must be an integer MiB in "
            f"[{RAM_PARK_BUDGET_MIN_MIB}, {RAM_PARK_BUDGET_MAX_MIB}], got {value!r}"
        )
    upsert_app_settings({RAM_PARK_BUDGET_SETTING_KEY: coerced})
    _invalidate(RAM_PARK_BUDGET_SETTING_KEY)
    return coerced
```

- [ ] **Step 2: Write the settings tests, mirroring `tests/test_vram_budget_settings.py`'s exact pattern**

```python
# studio/backend/tests/test_memory_park_settings.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import pytest

import utils.memory_park_settings as mp


@pytest.fixture(autouse = True)
def _isolate(monkeypatch):
    monkeypatch.delenv(mp.RAM_PARK_BUDGET_ENV_VAR, raising = False)
    monkeypatch.setattr(mp, "_cached_setting", lambda _key: None)


class TestCoerceRamBudgetMib:
    @pytest.mark.parametrize("value", [0, 1024, 28672, 61440, "16384"])
    def test_accepts_in_range(self, value):
        assert mp.coerce_ram_budget_mib(value) == int(value)

    @pytest.mark.parametrize(
        "value", [-1, 61441, "nan", float("nan"), "inf", float("inf"), "", None, True, "abc"],
    )
    def test_rejects_out_of_range_or_unusable(self, value):
        assert mp.coerce_ram_budget_mib(value) is None


def test_get_budget_falls_back_to_default_with_nothing_set():
    assert mp.get_ram_park_budget_mib() == mp.RAM_PARK_BUDGET_DEFAULT_MIB


def test_get_budget_reads_the_environment_when_nothing_stored(monkeypatch):
    monkeypatch.setenv(mp.RAM_PARK_BUDGET_ENV_VAR, "16384")
    assert mp.get_ram_park_budget_mib() == 16384


def test_stored_value_wins_over_environment(monkeypatch):
    monkeypatch.setenv(mp.RAM_PARK_BUDGET_ENV_VAR, "16384")
    monkeypatch.setattr(mp, "_cached_setting", lambda _key: 8192)
    assert mp.get_ram_park_budget_mib() == 8192


def test_set_then_clear_round_trips(monkeypatch, tmp_path):
    calls = {"upserted": None}

    def fake_upsert(settings):
        calls["upserted"] = settings
        return settings

    monkeypatch.setattr("storage.studio_db.upsert_app_settings", fake_upsert)
    result = mp.set_ram_park_budget_mib(4096)
    assert result == 4096
    assert calls["upserted"] == {mp.RAM_PARK_BUDGET_SETTING_KEY: 4096}

    mp.set_ram_park_budget_mib(None)
    assert calls["upserted"] == {mp.RAM_PARK_BUDGET_SETTING_KEY: None}


def test_set_rejects_an_out_of_range_value():
    with pytest.raises(ValueError):
        mp.set_ram_park_budget_mib(99999999)
```

- [ ] **Step 3: Run the settings tests to verify they pass**

Run: `C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_memory_park_settings.py -q -p no:cacheprovider`
Expected: 10 passed (2 test classes/functions collapse into the parametrized counts: 5 + 8 + 1 + 1 + 1 + 1 = check actual collected count, all green)

- [ ] **Step 4: Write `memory_residency.py`**

```python
# studio/backend/core/inference/memory_residency.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAM-tier budget and LRU eviction for parked (not destroyed) diffusion/video pipelines.

Deliberately ignorant of what a "pipeline" is -- every operation takes callables the caller
hands it, the same decoupling core.inference.gpu_arbiter._EVICTORS already uses. This module
only ever tracks owner -> (footprint_mib, parked_at, unload_callable); it never imports
DiffusionBackend or the video backend.

The tracked tuple stores each owner's UNLOAD callable, not a restore one: forced eviction of an
older parked owner (to make RAM room for a new one) always means a full teardown of that OTHER
owner, which only ITS OWN unload() can do -- this module has no other way to reach it. restore()
is never stored; it is supplied fresh by restore_owner's own caller, since only that caller
(the backend whose pipeline it is) ever has a reason to call it.

Contract: park_owner's caller (gpu_arbiter's evictors) never needs its own fallback logic --
on return, the named owner is EITHER parked (tracked here) OR fully unloaded. It never raises
for any of the anticipated fallback paths (unsizeable footprint, over budget with a failed
forced eviction, park() inapplicable or failing); a genuinely unexpected exception from park()
or unload() still propagates, since that is a bug, not a known degrade path.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
# owner -> (footprint_mib, parked_at_monotonic, unload_callable)
_parked: dict[str, tuple[int, float, Callable[[], None]]] = {}


def is_parked(owner: str) -> bool:
    with _lock:
        return owner in _parked


def parked_footprint_mib() -> int:
    with _lock:
        return sum(footprint for footprint, _, _ in _parked.values())


def forget_parked(owner: str) -> None:
    """Clears owner's parked bookkeeping without calling restore() or unload() -- the caller has
    already decided what to do with the actual pipeline."""
    with _lock:
        _parked.pop(owner, None)


def _oldest_parked_other_than(owner: str) -> Optional[str]:
    with _lock:
        candidates = [(parked_at, o) for o, (_, parked_at, _) in _parked.items() if o != owner]
    if not candidates:
        return None
    candidates.sort(key = lambda pair: pair[0])
    return candidates[0][1]


def park_owner(
    owner: str,
    park: Callable[[], bool],
    unload: Callable[[], None],
    footprint_mib: Optional[int],
) -> None:
    if footprint_mib is None:
        logger.info("memory_residency: %s has no resident footprint estimate, unloading directly", owner)
        unload()
        forget_parked(owner)
        return

    from utils.memory_park_settings import get_ram_park_budget_mib

    budget = get_ram_park_budget_mib()
    while parked_footprint_mib() + footprint_mib > budget:
        victim = _oldest_parked_other_than(owner)
        if victim is None:
            logger.info(
                "memory_residency: parking %s (%d MiB) would exceed the %d MiB budget with "
                "nothing left to evict; unloading directly", owner, footprint_mib, budget,
            )
            unload()
            forget_parked(owner)
            return
        with _lock:
            _, _, victim_unload = _parked.get(victim, (0, 0.0, None))
        try:
            if victim_unload is not None:
                victim_unload()
        except Exception:
            logger.exception(
                "memory_residency: forced eviction of %s failed; unloading %s directly instead "
                "of parking it", victim, owner,
            )
            unload()
            forget_parked(owner)
            return
        forget_parked(victim)

    try:
        park_ok = park()
    except Exception:
        logger.exception("memory_residency: park() raised for %s, unloading directly", owner)
        park_ok = False
    if not park_ok:
        unload()
        forget_parked(owner)
        return
    with _lock:
        _parked[owner] = (footprint_mib, time.monotonic(), unload)


def restore_owner(owner: str, restore: Callable[[], None]) -> None:
    with _lock:
        if owner not in _parked:
            raise KeyError(f"{owner!r} is not parked")
    restore()
    forget_parked(owner)
```

- [ ] **Step 5: Write the module tests**

```python
# studio/backend/tests/test_memory_residency.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the RAM-tier park/restore coordinator. No real backend, GPU, or model --
every owner is a fake with recorder callables, matching tests/test_gpu_arbiter.py's pattern."""

from __future__ import annotations

import pytest

import core.inference.memory_residency as mr


@pytest.fixture(autouse = True)
def _reset(monkeypatch):
    monkeypatch.setattr(mr, "_parked", {})
    monkeypatch.setattr("utils.memory_park_settings.get_ram_park_budget_mib", lambda: 10000)


def _fake_owner(name, calls, park_ok = True, footprint = 1000):
    def park():
        calls.append(f"park-{name}")
        return park_ok

    def unload():
        calls.append(f"unload-{name}")

    def restore():
        calls.append(f"restore-{name}")

    return park, unload, restore, footprint


def test_park_then_is_parked_true():
    calls = []
    park, unload, restore, footprint = _fake_owner("a", calls)
    mr.park_owner("a", park, unload, footprint)
    assert mr.is_parked("a")
    assert calls == ["park-a"]


def test_restore_clears_parked_state():
    calls = []
    park, unload, restore, footprint = _fake_owner("a", calls)
    mr.park_owner("a", park, unload, footprint)
    mr.restore_owner("a", restore)
    assert not mr.is_parked("a")
    assert calls == ["park-a", "restore-a"]


def test_restore_raises_if_not_parked():
    with pytest.raises(KeyError):
        mr.restore_owner("nope", lambda: None)


def test_none_footprint_unloads_directly_never_parks():
    calls = []
    park, unload, restore, _ = _fake_owner("a", calls)
    mr.park_owner("a", park, unload, None)
    assert not mr.is_parked("a")
    assert calls == ["unload-a"]  # park() never called


def test_park_returning_false_falls_back_to_unload():
    calls = []
    park, unload, restore, footprint = _fake_owner("a", calls, park_ok = False)
    mr.park_owner("a", park, unload, footprint)
    assert not mr.is_parked("a")
    assert calls == ["park-a", "unload-a"]


def test_park_raising_falls_back_to_unload():
    calls = []
    def raising_park():
        calls.append("park-a")
        raise RuntimeError("boom")
    def unload():
        calls.append("unload-a")
    mr.park_owner("a", raising_park, unload, 1000)
    assert not mr.is_parked("a")
    assert calls == ["park-a", "unload-a"]


def test_exceeding_budget_evicts_the_oldest_parked_other_owner(monkeypatch):
    monkeypatch.setattr("utils.memory_park_settings.get_ram_park_budget_mib", lambda: 1500)
    calls = []
    park_a, unload_a, _, _ = _fake_owner("a", calls, footprint = 1000)
    mr.park_owner("a", park_a, unload_a, 1000)
    park_b, unload_b, _, _ = _fake_owner("b", calls, footprint = 1000)
    mr.park_owner("b", park_b, unload_b, 1000)  # 1000 + 1000 > 1500 -> evict a first
    assert calls == ["park-a", "unload-a", "park-b"]
    assert not mr.is_parked("a")
    assert mr.is_parked("b")


def test_failed_forced_eviction_falls_back_to_unloading_the_new_owner(monkeypatch):
    monkeypatch.setattr("utils.memory_park_settings.get_ram_park_budget_mib", lambda: 1500)
    calls = []
    def raising_unload_a():
        calls.append("unload-a-fails")
        raise RuntimeError("boom")
    park_a, _, _, _ = _fake_owner("a", calls, footprint = 1000)
    mr.park_owner("a", park_a, raising_unload_a, 1000)
    park_b, unload_b, _, _ = _fake_owner("b", calls, footprint = 1000)
    mr.park_owner("b", park_b, unload_b, 1000)
    assert calls == ["park-a", "unload-b"]  # b's own park() never even attempted
    assert not mr.is_parked("a")
    assert not mr.is_parked("b")


def test_new_owner_alone_exceeds_budget_falls_back_to_direct_unload(monkeypatch):
    monkeypatch.setattr("utils.memory_park_settings.get_ram_park_budget_mib", lambda: 500)
    calls = []
    park, unload, _, _ = _fake_owner("a", calls, footprint = 1000)
    mr.park_owner("a", park, unload, 1000)  # nothing parked to evict, still over budget alone
    assert calls == ["unload-a"]  # park() never even attempted
    assert not mr.is_parked("a")


def test_parked_footprint_mib_sums_all_parked_owners():
    calls = []
    park_a, unload_a, _, _ = _fake_owner("a", calls, footprint = 1000)
    mr.park_owner("a", park_a, unload_a, 1000)
    park_b, unload_b, _, _ = _fake_owner("b", calls, footprint = 2000)
    mr.park_owner("b", park_b, unload_b, 2000)
    assert mr.parked_footprint_mib() == 3000


def test_a_fresh_module_state_reports_nothing_parked():
    assert mr.parked_footprint_mib() == 0
    assert not mr.is_parked("anything")
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_memory_residency.py tests/test_memory_park_settings.py -q -p no:cacheprovider`
Expected: all passed, 0 failed.

- [ ] **Step 7: Commit**

```bash
git add studio/backend/core/inference/memory_residency.py studio/backend/utils/memory_park_settings.py studio/backend/tests/test_memory_residency.py studio/backend/tests/test_memory_park_settings.py
git commit -m "feat(memory): RAM-tier park/restore coordinator with budget + LRU eviction"
```

- [ ] **Step 8: Mutation control — budget check**

Temporarily change `while parked_footprint_mib() + footprint_mib > budget:` to `while False:` in
`memory_residency.py`. Run `test_exceeding_budget_evicts_the_oldest_parked_other_owner` — expect
FAIL (owner `a` never evicted, `b` parks anyway, assertion on `calls` mismatches). Revert:
`git checkout -- studio/backend/core/inference/memory_residency.py`.

---

### Task 2: `DiffusionBackend.park()` / `.restore()` / `.resident_footprint_mib()`

**Files:**
- Modify: `studio/backend/core/inference/diffusion.py` (`_LoadState` at line 761; new methods
  added near `unload()` at line 5973; `_LoadState(...)` construction at line 4374)
- Test: `studio/backend/tests/test_diffusion_backend.py`

**Interfaces:**
- Consumes: nothing from Task 1 directly (this task's methods are called BY `memory_residency`
  via callables, they don't import it).
- Produces: `DiffusionBackend.park() -> bool`, `.restore() -> None`, `.resident_footprint_mib() -> Optional[int]`. Task 4/5 consume these.

- [ ] **Step 1: Add two new fields to `_LoadState`** (`core/inference/diffusion.py:761`)

```python
    # New fields for park()/restore() (Unified Memory core). Defaulted so every existing
    # construction (including every test fixture already in this file) keeps working unchanged.
    parked: bool = False
    resident_mib: Optional[int] = None
```

Add these two lines inside the `_LoadState` dataclass body, after the existing `gpu_ordinal`/
`placed_ordinal` fields (anywhere among the defaulted fields is fine; dataclass field order
after the first defaulted field doesn't matter for keyword construction, which is how every
existing call site already builds `_LoadState`).

- [ ] **Step 2: Populate `resident_mib` at the construction site** (`core/inference/diffusion.py:4374-4413`)

Find the `MemoryPlan` this load already computed (it is the `plan` local variable already in
scope at this point in `_run_load` — confirm by reading the ~100 lines before line 4374 for
where `plan = plan_diffusion_memory(...)` or similar is assigned; it is the same `plan` whose
`.requested_mode` already feeds `memory_mode = plan.requested_mode` two lines into the existing
`_LoadState(...)` call). Add one line to that same call:

```python
                        resident_mib = plan.estimates.get("resident_required_mib"),
```

- [ ] **Step 3: Write the failing tests first**

Find `test_diffusion_backend.py`'s existing `_LoadState(object(), ...)` fixture pattern (search
for `backend._state = _LoadState(` — several examples already exist in this file) and match its
exact construction style. Write:

```python
def test_park_moves_a_fully_resident_pipeline_to_cpu_and_keeps_state():
    backend = DiffusionBackend()
    fake_pipe = _RecordingPipe()
    backend._state = _LoadState(
        fake_pipe, None, "repo", "base", "cuda", "float16", False,
        offload_policy = OFFLOAD_NONE,
    )
    result = backend.park()
    assert result is True
    assert fake_pipe.to_calls == ["cpu"]
    assert backend._state is not None
    assert backend._state.parked is True


def test_park_refuses_a_non_none_offload_policy():
    backend = DiffusionBackend()
    fake_pipe = _RecordingPipe()
    backend._state = _LoadState(
        fake_pipe, None, "repo", "base", "cuda", "float16", False,
        offload_policy = OFFLOAD_GROUP,
    )
    result = backend.park()
    assert result is False
    assert fake_pipe.to_calls == []  # never even attempted -- the whole point of the gate
    assert backend._state is not None  # unchanged, still resident


def test_park_failure_falls_back_to_a_real_unload():
    backend = DiffusionBackend()
    fake_pipe = _RecordingPipe(raise_on_to = True)
    backend._state = _LoadState(
        fake_pipe, None, "repo", "base", "cuda", "float16", False,
        offload_policy = OFFLOAD_NONE,
    )
    result = backend.park()
    assert result is False
    assert backend._state is None  # real unload happened


def test_restore_moves_a_parked_pipeline_back_to_its_device():
    backend = DiffusionBackend()
    fake_pipe = _RecordingPipe()
    backend._state = _LoadState(
        fake_pipe, None, "repo", "base", "cuda", "float16", False,
        offload_policy = OFFLOAD_NONE, parked = True,
    )
    backend.restore()
    assert fake_pipe.to_calls == ["cuda"]
    assert backend._state.parked is False


def test_restore_raises_when_nothing_is_parked():
    backend = DiffusionBackend()
    with pytest.raises(Exception):
        backend.restore()


def test_resident_footprint_mib_reads_the_recorded_value():
    backend = DiffusionBackend()
    backend._state = _LoadState(
        object(), None, "repo", "base", "cuda", "float16", False, resident_mib = 4096,
    )
    assert backend.resident_footprint_mib() == 4096


def test_resident_footprint_mib_is_none_when_unrecorded():
    backend = DiffusionBackend()
    backend._state = _LoadState(object(), None, "repo", "base", "cuda", "float16", False)
    assert backend.resident_footprint_mib() is None
```

Add a small `_RecordingPipe` test helper near the top of `test_diffusion_backend.py` (or reuse
one if a similar stub already exists in the file — check first):

```python
class _RecordingPipe:
    def __init__(self, raise_on_to = False):
        self.to_calls = []
        self._raise_on_to = raise_on_to

    def to(self, device):
        if self._raise_on_to:
            raise RuntimeError("simulated .to() failure")
        self.to_calls.append(device)
        return self
```

Check the exact positional argument order `_LoadState`'s constructor expects (`pipe, family,
repo_id, base_repo, device, dtype, cpu_offload`, then keyword-only for everything defaulted) by
reading one of the file's EXISTING `_LoadState(object(), ...)` constructions — match it exactly;
the order above is inferred from the dataclass field declarations read during planning and must
be verified against the actual file before use.

- [ ] **Step 4: Run tests to verify they fail**

Run: `C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_diffusion_backend.py -k "park or restore or resident_footprint" -v`
Expected: FAIL — `AttributeError: 'DiffusionBackend' object has no attribute 'park'` (and similar).

- [ ] **Step 5: Implement `park()`, `restore()`, `resident_footprint_mib()`**

Add near `unload()` (`core/inference/diffusion.py:5973`), reusing its exact cancellation
preamble. Read `unload()`'s full body first (lines 5973-5996) and copy its locking shape
verbatim into `park()`:

```python
    def park(self) -> bool:
        """Move a fully-resident (OFFLOAD_NONE) pipeline's modules to CPU and clear the CUDA
        cache, WITHOUT dropping self._state -- unlike unload(), the pipeline object survives for
        a fast restore(). Returns False (does nothing) if the loaded pipeline isn't at
        OFFLOAD_NONE, or if the move itself fails (falls back to a real unload() in that case)."""
        with self._lock:
            state = self._state
            if state is None or state.offload_policy != OFFLOAD_NONE:
                return False
            self._cancel_event.set()
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
            self._teardown_waiters += 1
            self._load_token += 1
            self._loading = None
        with self._generate_lock:
            with self._lock:
                try:
                    state = self._state
                    if state is None:
                        return False
                    try:
                        state.pipe.to("cpu")
                    except Exception:
                        logger.exception("diffusion.park: .to('cpu') failed, falling back to unload")
                        self._unload_locked()
                        return False
                    clear_gpu_cache()
                    self._state = replace(state, parked = True)
                    return True
                finally:
                    self._teardown_waiters -= 1

    def restore(self) -> None:
        """Move a parked pipeline back to its recorded device. Raises if nothing is parked, or
        if the move fails (e.g. CUDA OOM because something else grew VRAM usage while parked)."""
        with self._lock:
            state = self._state
            if state is None or not state.parked:
                raise RuntimeError("diffusion.restore: nothing is parked")
            state.pipe.to(state.device)
            self._state = replace(state, parked = False)

    def resident_footprint_mib(self) -> Optional[int]:
        state = self._state
        return state.resident_mib if state is not None else None
```

Add `from dataclasses import replace` to the top of `diffusion.py` if not already imported
(check first — `diffusion_memory.py` already imports it, `diffusion.py` may not).

- [ ] **Step 6: Run tests to verify they pass**

Run: `C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_diffusion_backend.py -k "park or restore or resident_footprint" -v`
Expected: PASS, all 7.

- [ ] **Step 7: Run the full diffusion backend test file to confirm no collateral breakage**

Run: `C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_diffusion_backend.py -q -p no:cacheprovider`
Expected: all passed (baseline count plus 7 new).

- [ ] **Step 8: Commit**

```bash
git add studio/backend/core/inference/diffusion.py studio/backend/tests/test_diffusion_backend.py
git commit -m "feat(diffusion): park()/restore()/resident_footprint_mib(), gated to OFFLOAD_NONE"
```

- [ ] **Step 9: Mutation control — offload-policy gate**

Temporarily remove the `state.offload_policy != OFFLOAD_NONE` check from `park()` (always
proceed to `.to("cpu")`). Run `test_park_refuses_a_non_none_offload_policy` — expect FAIL (the
fake pipe's `.to()` gets called, `fake_pipe.to_calls == []` assertion fails). Revert:
`git checkout -- studio/backend/core/inference/diffusion.py`.

---

### Task 3: Video backend `park()` / `.restore()` / `.resident_footprint_mib()`

**Files:**
- Modify: `studio/backend/core/inference/video.py` (`_VideoLoadState` at line 497; new methods
  near `unload()` at line 5974; `_VideoLoadState(...)` constructed at lines 1941, 4041, 4790)
- Test: `studio/backend/tests/test_video_backend.py`

**Interfaces:**
- Consumes: nothing from earlier tasks directly.
- Produces: same three methods as Task 2, on the video backend class. Task 4/6 consume these.

- [ ] **Step 1: Add the same two fields to `_VideoLoadState`** (`core/inference/video.py:497`)

```python
    # New fields for park()/restore() (Unified Memory core).
    parked: bool = False
    resident_mib: Optional[int] = None
```

- [ ] **Step 2: Populate `resident_mib` at ALL THREE construction sites** (lines 1941, 4041, 4790)

At each of the three `_VideoLoadState(...)` calls, add one line reading from that call's own
`plan` (the `plan_diffusion_memory(...)` result already in scope near each site — confirmed
present via `plan_diffusion_memory` import at the top of `video.py`; read the ~80 lines before
each construction site to find that site's own `plan` variable name, which may differ between
the three call sites since they're different load paths):

```python
                        resident_mib = plan.estimates.get("resident_required_mib"),
```

Skip this for the site at line 1941 specifically if that load path is the native `sd_cpp`
engine's own construction (confirmed earlier during spec-writing: `engine = "sd_cpp"` at that
exact site) — a native-engine load has no resident-in-Studio-process footprint to estimate the
same way (verify by checking whether `plan_diffusion_memory` is even called on that path; if
not, leave `resident_mib` at its `None` default there, which is correct: `park()`/`restore()`
are gated to `engine == "diffusers"` anyway per Task 4's evictor branch, so this field is simply
unused on the native path).

- [ ] **Step 3: Write the failing tests**, mirroring Task 2's exactly (same test names, same
  `_RecordingPipe` helper — check whether `test_video_backend.py` already has an equivalent
  stub before adding a duplicate one), against `_VideoLoadState` instead of `_LoadState`. Confirm
  the video backend's exact `unload()` cancellation preamble first (`core/inference/video.py:5974-5993`,
  already read during planning) and reuse it exactly the same way Task 2's `park()` reused
  diffusion's.

- [ ] **Step 4: Run tests to verify they fail**

Run: `C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_video_backend.py -k "park or restore or resident_footprint" -v`
Expected: FAIL.

- [ ] **Step 5: Implement `park()`, `restore()`, `resident_footprint_mib()`** on the video
  backend class, mirroring Task 2's diffusion implementation exactly, but calling
  `self._teardown_state_locked()` in place of `self._unload_locked()` on a failed park (video's
  own teardown method, confirmed at `core/inference/video.py:5960`).

- [ ] **Step 6: Run tests to verify they pass**

Run: `C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_video_backend.py -k "park or restore or resident_footprint" -v`
Expected: PASS.

- [ ] **Step 7: Run the full video backend test file to confirm no collateral breakage**

Run: `C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_video_backend.py -q -p no:cacheprovider`
Expected: all passed.

- [ ] **Step 8: Commit**

```bash
git add studio/backend/core/inference/video.py studio/backend/tests/test_video_backend.py
git commit -m "feat(video): park()/restore()/resident_footprint_mib(), gated to OFFLOAD_NONE"
```

- [ ] **Step 9: Mutation control — offload-policy gate**, same technique as Task 2 Step 9,
  against video's `park()`. Revert after confirming the failure.

---

### Task 4: `gpu_arbiter.py` evictor branch

**Files:**
- Modify: `studio/backend/core/inference/gpu_arbiter.py` (`_evict_diffusion` at line 55,
  `_evict_video` at line 61)
- Test: `studio/backend/tests/test_gpu_arbiter.py`

**Interfaces:**
- Consumes: `memory_residency.park_owner` (Task 1), `DiffusionBackend.park/.unload/.resident_footprint_mib` and the video backend's equivalents (Tasks 2, 3).
- Produces: nothing new consumed by later tasks — this is the wiring task.

- [ ] **Step 1: Write the failing tests first**, extending `test_gpu_arbiter.py`'s existing
  `calls` fixture pattern (read the file's current content first — it's short, already fully
  read during planning):

```python
def test_diffusion_eviction_routes_through_memory_residency_when_diffusers(monkeypatch):
    recorded = []
    monkeypatch.setattr(arb, "_owner", None)
    monkeypatch.setattr(
        "core.inference.diffusion_engine_router.active_engine_name", lambda: "diffusers",
    )
    fake_engine = types.SimpleNamespace(
        park = lambda: True, unload = lambda: None, resident_footprint_mib = lambda: 4096,
    )
    monkeypatch.setattr(
        "core.inference.diffusion_engine_router.get_active_diffusion_engine", lambda: fake_engine,
    )
    monkeypatch.setattr(
        "core.inference.memory_residency.park_owner",
        lambda owner, park, unload, footprint: recorded.append((owner, footprint)),
    )
    arb.acquire_for(arb.DIFFUSION)
    arb.acquire_for(arb.CHAT)  # evicts diffusion
    assert recorded == [(arb.DIFFUSION, 4096)]


def test_diffusion_eviction_stays_direct_unload_for_sd_cpp(monkeypatch):
    recorded = []
    monkeypatch.setattr(arb, "_owner", None)
    monkeypatch.setattr(
        "core.inference.diffusion_engine_router.active_engine_name", lambda: "sd_cpp",
    )
    fake_engine = types.SimpleNamespace(unload = lambda: recorded.append("unloaded"))
    monkeypatch.setattr(
        "core.inference.diffusion_engine_router.get_active_diffusion_engine", lambda: fake_engine,
    )
    arb.acquire_for(arb.DIFFUSION)
    arb.acquire_for(arb.CHAT)
    assert recorded == ["unloaded"]


def test_video_eviction_routes_through_memory_residency_when_diffusers(monkeypatch):
    recorded = []
    monkeypatch.setattr(arb, "_owner", None)
    fake_backend = types.SimpleNamespace(
        status = lambda: {"engine": "diffusers"},
        park = lambda: True, unload = lambda: None, resident_footprint_mib = lambda: 8192,
    )
    monkeypatch.setattr("core.inference.video.get_video_backend", lambda: fake_backend)
    monkeypatch.setattr(
        "core.inference.memory_residency.park_owner",
        lambda owner, park, unload, footprint: recorded.append((owner, footprint)),
    )
    arb.acquire_for(arb.VIDEO)
    arb.acquire_for(arb.CHAT)
    assert recorded == [(arb.VIDEO, 8192)]


def test_video_eviction_stays_direct_unload_for_sd_cpp(monkeypatch):
    recorded = []
    monkeypatch.setattr(arb, "_owner", None)
    fake_backend = types.SimpleNamespace(
        status = lambda: {"engine": "sd_cpp"}, unload = lambda: recorded.append("unloaded"),
    )
    monkeypatch.setattr("core.inference.video.get_video_backend", lambda: fake_backend)
    arb.acquire_for(arb.VIDEO)
    arb.acquire_for(arb.CHAT)
    assert recorded == ["unloaded"]
```

Add `import types` to the test file's imports if not already present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_gpu_arbiter.py -v`
Expected: the 4 new tests FAIL (the old `_EVICTORS`-monkeypatch-based tests still pass, since
they replace the whole evictor function and never reach the branch this task adds).

- [ ] **Step 3: Implement the evictor branch**

Replace `_evict_diffusion`/`_evict_video` in `core/inference/gpu_arbiter.py` (currently lines
55-63):

```python
def _evict_diffusion() -> None:
    from core.inference.diffusion_engine_router import (
        active_engine_name, get_active_diffusion_engine,
    )

    engine = get_active_diffusion_engine()
    if active_engine_name() == "sd_cpp":
        engine.unload()  # unchanged: subprocess-based, park/restore not applicable
        return
    from core.inference import memory_residency
    memory_residency.park_owner(
        DIFFUSION, engine.park, engine.unload, engine.resident_footprint_mib(),
    )


def _evict_video() -> None:
    from core.inference.video import get_video_backend

    backend = get_video_backend()
    if backend.status()["engine"] == "sd_cpp":
        backend.unload()  # unchanged: subprocess-based, park/restore not applicable
        return
    from core.inference import memory_residency
    memory_residency.park_owner(
        VIDEO, backend.park, backend.unload, backend.resident_footprint_mib(),
    )
```

Import the exact string constant (`"sd_cpp"`) rather than `ENGINE_SD_CPP` from
`sd_cpp_engine.py` if that introduces a heavier import chain into `gpu_arbiter.py` — check
whether `diffusion_engine_router.py`'s `ENGINE_SD_CPP` is a cheap, side-effect-free import first
(it looked like a plain string constant during planning); prefer importing the named constant
over the bare string literal if it stays cheap, for the usual reason (a rename elsewhere breaks
this loudly instead of silently).

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_gpu_arbiter.py -q -p no:cacheprovider`
Expected: all passed (existing tests + 4 new).

- [ ] **Step 5: Commit**

```bash
git add studio/backend/core/inference/gpu_arbiter.py studio/backend/tests/test_gpu_arbiter.py
git commit -m "feat(gpu_arbiter): route diffusion/video eviction through memory_residency for the diffusers engine"
```

- [ ] **Step 6: Mutation control**

Temporarily hardcode `_evict_diffusion` to always call `engine.unload()` directly (bypass the
`memory_residency` branch entirely, as if this task were never done). Run
`test_diffusion_eviction_routes_through_memory_residency_when_diffusers` — expect FAIL (`recorded`
stays empty). Revert: `git checkout -- studio/backend/core/inference/gpu_arbiter.py`.

---

### Task 5: `DiffusionBackend.begin_load()` — restore-or-cold-load decision

**Files:**
- Modify: `studio/backend/core/inference/diffusion.py` (`begin_load` at line 1837)
- Test: `studio/backend/tests/test_diffusion_backend.py`

**Interfaces:**
- Consumes: `memory_residency.is_parked/.restore_owner/.forget_parked` (Task 1),
  `DiffusionBackend.restore/.unload` (Task 2).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing tests first**

```python
def test_begin_load_restores_instead_of_cold_loading_on_a_matching_identity(monkeypatch):
    backend = DiffusionBackend()
    fake_pipe = _RecordingPipe()
    backend._state = _LoadState(
        fake_pipe, None, "unsloth/same-repo", "base", "cuda", "float16", False,
        offload_policy = OFFLOAD_NONE, parked = True, kind = "gguf",
        gguf_filename = "model.gguf", transformer_quant = None, text_encoder_quant = None,
        memory_mode = "auto", gpu_ordinal = None,
    )
    monkeypatch.setattr("core.inference.memory_residency.is_parked", lambda owner: True)
    restored = []
    monkeypatch.setattr(
        "core.inference.memory_residency.restore_owner",
        lambda owner, restore: (restore(), restored.append(owner)),
    )
    run_load_calls = []
    monkeypatch.setattr(backend, "_run_load", lambda **kw: run_load_calls.append(kw))
    monkeypatch.setattr(backend, "validate_load_request", lambda *a, **k: types.SimpleNamespace(base_repo = "base"))
    monkeypatch.setattr(backend, "assert_precision_available", lambda *a, **k: None)

    backend.begin_load(
        "unsloth/same-repo", gguf_filename = "model.gguf", model_kind = "gguf",
    )
    assert restored == [arb.DIFFUSION]
    assert run_load_calls == []  # the cold-load thread must never spawn


def test_begin_load_cold_loads_when_the_requested_model_differs_from_whats_parked(monkeypatch):
    backend = DiffusionBackend()
    fake_pipe = _RecordingPipe()
    backend._state = _LoadState(
        fake_pipe, None, "unsloth/parked-repo", "base", "cuda", "float16", False,
        offload_policy = OFFLOAD_NONE, parked = True, kind = "gguf",
    )
    monkeypatch.setattr("core.inference.memory_residency.is_parked", lambda owner: True)
    forgotten = []
    monkeypatch.setattr(
        "core.inference.memory_residency.forget_parked", lambda owner: forgotten.append(owner),
    )
    restore_called = []
    monkeypatch.setattr(
        "core.inference.memory_residency.restore_owner",
        lambda owner, restore: restore_called.append(owner),
    )
    run_load_calls = []
    monkeypatch.setattr(backend, "_run_load", lambda **kw: run_load_calls.append(kw))
    monkeypatch.setattr(backend, "validate_load_request", lambda *a, **k: types.SimpleNamespace(base_repo = "base"))
    monkeypatch.setattr(backend, "assert_precision_available", lambda *a, **k: None)

    backend.begin_load("unsloth/a-totally-different-repo", model_kind = "gguf")
    assert restore_called == []  # never attempted -- identity mismatch
    assert len(run_load_calls) == 1  # cold load proceeded


def test_begin_load_falls_back_to_cold_load_when_restore_raises(monkeypatch):
    backend = DiffusionBackend()
    fake_pipe = _RecordingPipe()
    backend._state = _LoadState(
        fake_pipe, None, "unsloth/same-repo", "base", "cuda", "float16", False,
        offload_policy = OFFLOAD_NONE, parked = True, kind = "gguf",
    )
    monkeypatch.setattr("core.inference.memory_residency.is_parked", lambda owner: True)
    def raising_restore(owner, restore):
        raise RuntimeError("CUDA OOM")
    monkeypatch.setattr("core.inference.memory_residency.restore_owner", raising_restore)
    forgotten = []
    monkeypatch.setattr(
        "core.inference.memory_residency.forget_parked", lambda owner: forgotten.append(owner),
    )
    run_load_calls = []
    monkeypatch.setattr(backend, "_run_load", lambda **kw: run_load_calls.append(kw))
    monkeypatch.setattr(backend, "validate_load_request", lambda *a, **k: types.SimpleNamespace(base_repo = "base"))
    monkeypatch.setattr(backend, "assert_precision_available", lambda *a, **k: None)

    backend.begin_load("unsloth/same-repo", model_kind = "gguf")
    assert forgotten == [arb.DIFFUSION]
    assert len(run_load_calls) == 1  # fell through to cold load, did not raise to the caller
```

These three tests use `arb.DIFFUSION` (the owner-name constant `begin_load`'s new code will use
internally, written in Step 3) — add `import core.inference.gpu_arbiter as arb` and `import types`
to `test_diffusion_backend.py`'s imports if not already present (check first; `types` may already
be imported for other fixtures in this large file).

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_diffusion_backend.py -k "begin_load_restores or begin_load_cold_loads or begin_load_falls_back" -v`
Expected: FAIL (no such branch exists yet in `begin_load`).

- [ ] **Step 3: Implement `_matches_parked_identity` and the `begin_load` branch**

Add a new private method near `begin_load` (`core/inference/diffusion.py`, before line 1837):

```python
    def _matches_parked_identity(
        self, *, repo_id, gguf_filename, base_repo, model_kind, transformer_quant,
        text_encoder_quant, cpu_offload, memory_mode, gpu_ordinal, loras,
    ) -> bool:
        """Conservative by construction: any field that doesn't match, or can't be compared,
        means "not the same model" -- restore only happens on a confirmed exact match. Serving
        the wrong model's weights is the worst failure mode this whole piece could introduce."""
        state = self._state
        if state is None or not state.parked:
            return False
        if (
            state.repo_id != repo_id
            or state.gguf_filename != gguf_filename
            or state.base_repo != base_repo
            or state.kind != model_kind
            or state.transformer_quant != transformer_quant
            or state.text_encoder_quant != text_encoder_quant
            or state.cpu_offload != cpu_offload
            or state.memory_mode != memory_mode
            or state.gpu_ordinal != gpu_ordinal
        ):
            return False
        active_loras = _active_lora_pairs(state.pipe)
        requested_loras = loras or []
        return list(active_loras) == list(requested_loras)
```

(`_active_lora_pairs` is already imported/used in this file at line 5924, confirmed during
planning — no new import needed if it's already module-level; verify.)

Then in `begin_load`, immediately after the existing `assert_precision_available(...)` call and
before the `with self._lock:` block that spawns `_run_load` (around line 1897):

```python
        from core.inference import gpu_arbiter, memory_residency

        if memory_residency.is_parked(gpu_arbiter.DIFFUSION):
            if self._matches_parked_identity(
                repo_id = repo_id, gguf_filename = gguf_filename, base_repo = base_repo,
                model_kind = model_kind, transformer_quant = transformer_quant,
                text_encoder_quant = text_encoder_quant, cpu_offload = cpu_offload,
                memory_mode = memory_mode, gpu_ordinal = gpu_ordinal, loras = loras,
            ):
                try:
                    memory_residency.restore_owner(gpu_arbiter.DIFFUSION, self.restore)
                    return self.status()
                except Exception:
                    logger.exception("diffusion.begin_load: restore failed, falling back to a cold load")
                    memory_residency.forget_parked(gpu_arbiter.DIFFUSION)
            else:
                memory_residency.forget_parked(gpu_arbiter.DIFFUSION)
                self.unload()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_diffusion_backend.py -k "begin_load_restores or begin_load_cold_loads or begin_load_falls_back" -v`
Expected: PASS, all 3.

- [ ] **Step 5: Run the full diffusion backend test file to confirm no collateral breakage**

Run: `C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_diffusion_backend.py -q -p no:cacheprovider`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add studio/backend/core/inference/diffusion.py studio/backend/tests/test_diffusion_backend.py
git commit -m "feat(diffusion): begin_load restores a matching parked pipeline instead of cold-loading"
```

- [ ] **Step 7: Mutation control — identity check is the highest-priority guard in this whole plan**

Temporarily hardcode `_matches_parked_identity` to `return True` unconditionally. Run
`test_begin_load_cold_loads_when_the_requested_model_differs_from_whats_parked` — expect FAIL
(`restore_called` is no longer empty; the wrong model would have been "restored"). Revert:
`git checkout -- studio/backend/core/inference/diffusion.py`.

---

### Task 6: Video backend `begin_load()` — restore-or-cold-load decision

**Files:**
- Modify: `studio/backend/core/inference/video.py` (`begin_load` at line 1272)
- Test: `studio/backend/tests/test_video_backend.py`

**Interfaces:**
- Consumes: same as Task 5, video's own `park`/`restore`/`unload` (Task 3).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1-7: Mirror Task 5's Steps 1-7 exactly**, adapted for video's `begin_load` signature
  (no `loras` parameter — video has no LoRA support at all, confirmed during planning:
  "none of the image module's img2img/inpaint/ControlNet/LoRA surface applies" — so
  `_matches_parked_identity` for video omits the LoRA check entirely) and its own field names
  (`_VideoLoadState` uses the same `repo_id`/`base_repo`/`gguf_filename`/`kind`/
  `transformer_quant`/`text_encoder_quant`/`memory_mode`/`gpu_ordinal` names as `_LoadState`,
  confirmed during planning — no `cpu_offload` field on `_VideoLoadState`, so omit that
  comparison too; video's own `offload_policy` field is the closest equivalent and is already
  covered by the `park()`/`restore()` gate itself, not the identity check). This includes Task
  5's Step 7 mutation control (hardcode video's `_matches_parked_identity` to `return True`,
  confirm the mismatched-identity test fails, revert).

---

### Task 7: Whole-branch review and finish

Once Tasks 1-6 are complete and each individually reviewed (if using subagent-driven-development)
or self-reviewed (if executing inline):

- [ ] **Step 1:** Run the full scoped test command:

```bash
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_memory_residency.py tests/test_memory_park_settings.py tests/test_diffusion_backend.py tests/test_video_backend.py tests/test_gpu_arbiter.py -q -p no:cacheprovider
```

Expected: all passed, 0 failed.

- [ ] **Step 2:** Dispatch (or perform) a whole-branch review against this plan's spec
  (`docs/superpowers/specs/2026-09-30-unslothed-unified-memory-core-design.md`), on the most
  capable available model, per `superpowers:subagent-driven-development`'s process. Pay
  particular attention to this plan's own Review Focus items (above) and the spec's §9 — they are
  the same list, since this plan was written directly from that spec section.

- [ ] **Step 3:** Address any findings via the skill's single permitted fix wave, then re-review.

- [ ] **Step 4:** Once clean, use `superpowers:finishing-a-development-branch` to merge/PR/keep
  as the user directs.
