# Serve-time Context Auto-Fit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded 16384 serve context with one auto-fitted to the chosen GGUF's trained context and the detected VRAM, wired into the native Local Models serve path.

**Architecture:** Extend the GGUF header sniffer to read numeric fields; add a pure `recommend_context` function (VRAM/trained-ctx/CPU heuristics over injected dicts); have `LocalModelManager.start` compute the context from the model file + detected hardware and pass it to the existing `build_serve_argv(ctx_size=…)` parameter. GPU-layer auto-fitting is untouched (no `-ngl`).

**Tech Stack:** Python 3.14, existing `src/gguf_meta.py`, `services/hwfit/hardware.detect_system`, llama.cpp serving. No new dependencies.

## Global Constraints

- **No new dependencies.** No change to `src/localmodels/runtime.py` (it already accepts `ctx_size`).
- **Never set `-ngl` / `--n-gpu-layers`.** The bundled Vulkan build auto-fits GPU layers to VRAM; an explicit `-ngl` disables that fitter and OOMs. This is a hard safety constraint.
- **VRAM/RAM units are GiB.** `detect_system()` returns `gpu_vram_gb` and `available_ram_gb` in **binary GiB** (Windows path: `AdapterRAM / 1073741824`). Convert to bytes with `* 1024**3`, never `* 1e9`.
- `detect_system()` fields used: `has_gpu` (bool), `gpu_vram_gb` (float GiB), `available_ram_gb` (float GiB).
- **Pure recommender never raises.** `recommend_context` always returns a valid positive int; unreadable metadata / detect failure degrades to a safe default. `read_gguf_metadata` never raises (returns `{}`), matching `gguf_meta.py`'s contract.
- Context policy constants (`FLOOR=2048`, `HARD_CEILING=32768`, `CPU_CEILING=8192`, `DEFAULT_TRAINED=8192`, `KV_VRAM_FRACTION=0.5`, `CTX_LADDER=[2048,4096,8192,16384,32768]`) live at the top of `serve_tuning.py`.
- pytest `--import-mode=importlib`. Commit directly to `dev`. Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Stage only the files each task names — never `git add -A`; never stage `installer/Output/Assist-Setup.exe`, `assistappicon.png`, `assistlogo.png`.
- ~220 unrelated pre-existing test failures exist elsewhere — run only the test files each task names.

---

### Task 1: `read_gguf_metadata` — numeric GGUF fields

**Files:**
- Modify: `src/gguf_meta.py` (add a function + two module constants; leave `read_gguf_architecture` unchanged)
- Test: `tests/test_gguf_metadata.py`

**Interfaces:**
- Consumes: the existing module constants `_SCALAR_SIZES`, `_STRING`, `_ARRAY`, `_MAX_KEYS`, `_MAX_ARRAY_ITEMS` in `src/gguf_meta.py`.
- Produces: `read_gguf_metadata(path: str) -> dict` returning any of `{"architecture": str, "context_length": int, "block_count": int, "head_count": int, "head_count_kv": int, "embedding_length": int}` (only keys found); `{}` for unreadable/non-GGUF. Never raises.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gguf_metadata.py`:

```python
import struct

from src.gguf_meta import read_gguf_metadata

_STRING_T, _UINT32_T, _UINT64_T = 8, 4, 10


def _kv_string(key: bytes, val: bytes) -> bytes:
    return (struct.pack("<Q", len(key)) + key
            + struct.pack("<I", _STRING_T)
            + struct.pack("<Q", len(val)) + val)


def _kv_uint32(key: bytes, n: int) -> bytes:
    return struct.pack("<Q", len(key)) + key + struct.pack("<I", _UINT32_T) + struct.pack("<I", n)


def _kv_uint64(key: bytes, n: int) -> bytes:
    return struct.pack("<Q", len(key)) + key + struct.pack("<I", _UINT64_T) + struct.pack("<Q", n)


def _build_gguf(entries: list) -> bytes:
    body = b"".join(entries)
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(entries)) + body


def test_reads_arch_and_numeric_fields(tmp_path):
    entries = [
        _kv_string(b"general.architecture", b"llama"),
        _kv_uint32(b"llama.context_length", 8192),
        _kv_uint32(b"llama.block_count", 32),
        _kv_uint32(b"llama.attention.head_count", 32),
        _kv_uint32(b"llama.attention.head_count_kv", 8),
        _kv_uint64(b"llama.embedding_length", 4096),
    ]
    p = tmp_path / "m.gguf"
    p.write_bytes(_build_gguf(entries))
    meta = read_gguf_metadata(str(p))
    assert meta == {
        "architecture": "llama", "context_length": 8192, "block_count": 32,
        "head_count": 32, "head_count_kv": 8, "embedding_length": 4096,
    }


def test_arch_after_numeric_keys_still_resolves(tmp_path):
    # order-independence: numeric keys before the arch string
    entries = [
        _kv_uint32(b"llama.context_length", 4096),
        _kv_string(b"general.architecture", b"llama"),
    ]
    p = tmp_path / "m2.gguf"
    p.write_bytes(_build_gguf(entries))
    assert read_gguf_metadata(str(p))["context_length"] == 4096


def test_non_gguf_returns_empty(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"NOTGGUF" + b"\x00" * 40)
    assert read_gguf_metadata(str(p)) == {}


def test_truncated_returns_empty(tmp_path):
    p = tmp_path / "t.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<IQ", 3, 0))  # header cut short
    assert read_gguf_metadata(str(p)) == {}


def test_missing_file_returns_empty():
    assert read_gguf_metadata("C:/no/such/file.gguf") == {}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_gguf_metadata.py --import-mode=importlib -q`
Expected: FAIL (`ImportError: cannot import name 'read_gguf_metadata'`).

- [ ] **Step 3: Implement `read_gguf_metadata`**

Add to `src/gguf_meta.py` (below `read_gguf_architecture`; do not modify that function). The module already defines `_SCALAR_SIZES`, `_STRING = 8`, `_ARRAY = 9`, `_MAX_KEYS`, `_MAX_ARRAY_ITEMS`, and imports `struct`:

```python
# GGUF integer scalar types → struct format. Sizes match _SCALAR_SIZES, so
# reading the value advances the file exactly as the seek-past path would.
_INT_FORMATS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}

# result field -> GGUF metadata key suffix (prefixed by the architecture)
_WANTED_SUFFIXES = {
    "context_length": "context_length",
    "block_count": "block_count",
    "head_count": "attention.head_count",
    "head_count_kv": "attention.head_count_kv",
    "embedding_length": "embedding_length",
}


def read_gguf_metadata(path: str) -> dict:
    """Best-effort numeric + architecture metadata from a GGUF header.

    Walks the KV block, capturing `general.architecture` and every integer
    scalar, then resolves the `<arch>.*` fields this app needs to fit a serve
    context. Returns only the keys it found; {} for a non-GGUF / unreadable /
    truncated file. Never raises. `read_gguf_architecture` is left untouched;
    this file already keeps a separate walk per reader.
    """
    strings = {}
    ints = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return {}
            head = f.read(20)
            if len(head) != 20:
                return {}
            version, _tensors, kv_count = struct.unpack("<IQQ", head)
            if version not in (2, 3):
                return {}
            for _ in range(min(kv_count, _MAX_KEYS)):
                raw = f.read(8)
                if len(raw) != 8:
                    break
                klen = struct.unpack("<Q", raw)[0]
                key = f.read(klen)
                if len(key) != klen:
                    break
                raw = f.read(4)
                if len(raw) != 4:
                    break
                vtype = struct.unpack("<I", raw)[0]
                if vtype in _INT_FORMATS:
                    size = _SCALAR_SIZES[vtype]
                    raw = f.read(size)
                    if len(raw) != size:
                        break
                    ints[key.decode("utf-8", "replace")] = struct.unpack(_INT_FORMATS[vtype], raw)[0]
                elif vtype == _STRING:
                    raw = f.read(8)
                    if len(raw) != 8:
                        break
                    slen = struct.unpack("<Q", raw)[0]
                    val = f.read(slen)
                    if len(val) != slen:
                        break
                    strings[key.decode("utf-8", "replace")] = val.decode("utf-8", "replace")
                elif vtype == _ARRAY:
                    raw = f.read(12)
                    if len(raw) != 12:
                        break
                    etype, count = struct.unpack("<IQ", raw)
                    if etype in _SCALAR_SIZES:
                        f.seek(count * _SCALAR_SIZES[etype], 1)
                    elif etype == _STRING and count <= _MAX_ARRAY_ITEMS:
                        ok = True
                        for _ in range(count):
                            raw = f.read(8)
                            if len(raw) != 8:
                                ok = False
                                break
                            f.seek(struct.unpack("<Q", raw)[0], 1)
                        if not ok:
                            break
                    else:
                        break  # unknown/huge array element — stop; resolve what we have
                elif vtype in _SCALAR_SIZES:
                    f.seek(_SCALAR_SIZES[vtype], 1)  # float/bool: skip
                else:
                    break  # unknown value type — misalignment risk, stop
    except OSError:
        return {}

    arch = strings.get("general.architecture")
    if not arch:
        return {}
    result = {"architecture": arch}
    for field, suffix in _WANTED_SUFFIXES.items():
        key = f"{arch}.{suffix}"
        if key in ints:
            result[field] = ints[key]
    return result
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest tests/test_gguf_metadata.py --import-mode=importlib -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/gguf_meta.py tests/test_gguf_metadata.py
git commit -m "feat(localmodels): read_gguf_metadata — numeric GGUF header fields

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `recommend_context` — the pure serve-context recommender

**Files:**
- Create: `src/localmodels/serve_tuning.py`
- Test: `tests/test_serve_tuning.py`

**Interfaces:**
- Consumes: nothing (pure; operates on injected `meta` and `hardware` dicts).
- Produces:
  - `estimate_kv_bytes_per_token(meta: dict) -> int | None`
  - `recommend_context(meta: dict, hardware: dict, *, requested=None) -> int`
  - Module constants `FLOOR, HARD_CEILING, CPU_CEILING, DEFAULT_TRAINED, KV_VRAM_FRACTION, CTX_LADDER`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_serve_tuning.py`:

```python
from src.localmodels.serve_tuning import (
    recommend_context, estimate_kv_bytes_per_token,
    FLOOR, HARD_CEILING, CPU_CEILING, CTX_LADDER,
)

# A GQA-free 7B-ish config so KV/token is large enough to make VRAM bite.
_HEAVY = {"context_length": 131072, "block_count": 32, "head_count": 32,
          "head_count_kv": 32, "embedding_length": 4096}


def test_estimate_kv_bytes_per_token():
    # 2 (K,V) * 32 layers * 32 kv_heads * (4096/32=128 head_dim) * 2 bytes
    assert estimate_kv_bytes_per_token(_HEAVY) == 2 * 32 * 32 * 128 * 2
    assert estimate_kv_bytes_per_token({"block_count": 32}) is None       # missing fields
    assert estimate_kv_bytes_per_token({**_HEAVY, "head_count": 0}) is None  # zero
    assert estimate_kv_bytes_per_token("nope") is None


def test_vram_sensitivity_small_vs_large():
    small = recommend_context(_HEAVY, {"has_gpu": True, "gpu_vram_gb": 4})
    large = recommend_context(_HEAVY, {"has_gpu": True, "gpu_vram_gb": 24})
    assert small == 4096
    assert large == 16384
    assert small < large


def test_auto_never_exceeds_trained():
    meta = {**_HEAVY, "context_length": 4096}
    ctx = recommend_context(meta, {"has_gpu": True, "gpu_vram_gb": 48})
    assert ctx == 4096  # capped at trained, not the big-VRAM 32768


def test_cpu_only_bounded_by_ceiling():
    ctx = recommend_context(_HEAVY, {"has_gpu": False, "available_ram_gb": 64})
    assert ctx <= CPU_CEILING and ctx in CTX_LADDER


def test_missing_kv_fields_uses_vram_tier():
    meta = {"architecture": "llama", "context_length": 32768}  # no head/layer fields
    assert recommend_context(meta, {"has_gpu": True, "gpu_vram_gb": 6}) == 8192   # <12 tier
    assert recommend_context(meta, {"has_gpu": True, "gpu_vram_gb": 4}) == 4096   # <6 tier


def test_requested_override_honored_and_clamped():
    assert recommend_context(_HEAVY, {"has_gpu": True, "gpu_vram_gb": 4}, requested=100000) == HARD_CEILING
    assert recommend_context(_HEAVY, {"has_gpu": True, "gpu_vram_gb": 4}, requested=1000) == FLOOR
    # override may intentionally exceed the trained ceiling
    low = {**_HEAVY, "context_length": 4096}
    assert recommend_context(low, {"has_gpu": True, "gpu_vram_gb": 4}, requested=16384) == 16384
    assert recommend_context(_HEAVY, {}, requested=True) != True  # bool is not a valid int override


def test_result_invariants_never_raise():
    for vram in (0, 2, 6, 8, 12, 24, 48):
        for hw in ({"has_gpu": True, "gpu_vram_gb": vram}, {"has_gpu": False, "available_ram_gb": vram}):
            ctx = recommend_context(_HEAVY, hw)
            assert isinstance(ctx, int) and ctx in CTX_LADDER
            assert ctx <= min(_HEAVY["context_length"], HARD_CEILING)
    assert recommend_context({}, {}) in CTX_LADDER  # empty inputs → safe default
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_serve_tuning.py --import-mode=importlib -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.localmodels.serve_tuning'`).

- [ ] **Step 3: Implement `serve_tuning.py`**

Create `src/localmodels/serve_tuning.py`:

```python
"""Pure serve-parameter recommender (no I/O). Given GGUF metadata + detected
hardware, pick a llama-server context size that fits the model's trained limit
and the machine's VRAM — so a small card isn't crippled by an oversized KV
cache and a big box isn't needlessly capped. GPU-layer fitting is llama.cpp's
job (never set -ngl); this only chooses --ctx-size. Never raises."""

FLOOR = 2048
HARD_CEILING = 32768
CPU_CEILING = 8192          # huge context on CPU is painfully slow
DEFAULT_TRAINED = 8192      # assumed trained context when metadata is missing
KV_VRAM_FRACTION = 0.5      # target: KV cache uses at most this share of VRAM/RAM
CTX_LADDER = [2048, 4096, 8192, 16384, 32768]
_BYTES_PER_GIB = 1024 ** 3


def _is_pos_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _snap_down(x):
    """Largest ladder value <= x (falls back to the ladder floor)."""
    best = CTX_LADDER[0]
    for c in CTX_LADDER:
        if c <= x:
            best = c
    return best


def _vram_tier_fallback(gib):
    if gib < 6:
        return 4096
    if gib < 12:
        return 8192
    if gib < 24:
        return 16384
    return 32768


def _largest_ctx_within(kv_per_token, budget_bytes, max_ctx):
    """Largest ladder value c (<= max_ctx) whose KV cache fits budget_bytes."""
    best = CTX_LADDER[0]
    for c in CTX_LADDER:
        if c <= max_ctx and kv_per_token * c <= budget_bytes:
            best = c
    return best


def estimate_kv_bytes_per_token(meta):
    """f16 KV-cache bytes per token: 2 (K,V) * layers * kv_heads * head_dim * 2.
    None when required fields are absent/non-positive."""
    if not isinstance(meta, dict):
        return None
    block_count = meta.get("block_count")
    head_count = meta.get("head_count")
    head_count_kv = meta.get("head_count_kv")
    embedding_length = meta.get("embedding_length")
    if not all(_is_pos_int(v) for v in (block_count, head_count, head_count_kv, embedding_length)):
        return None
    head_dim = embedding_length // head_count
    if head_dim <= 0:
        return None
    return 2 * block_count * head_count_kv * head_dim * 2


def recommend_context(meta, hardware, *, requested=None):
    """Pick a serve --ctx-size. Explicit `requested` wins (clamped to
    [FLOOR, HARD_CEILING], may exceed trained). Otherwise fit to VRAM (or RAM
    for CPU-only), capped at the model's trained context. Always returns a
    valid positive int; never raises."""
    meta = meta if isinstance(meta, dict) else {}
    hardware = hardware if isinstance(hardware, dict) else {}

    if _is_pos_int(requested):
        return max(FLOOR, min(requested, HARD_CEILING))

    trained = meta.get("context_length")
    if not _is_pos_int(trained):
        trained = DEFAULT_TRAINED
    ceiling = min(trained, HARD_CEILING)

    kv = estimate_kv_bytes_per_token(meta)
    vram_gib = hardware.get("gpu_vram_gb") or 0
    has_gpu = bool(hardware.get("has_gpu")) and vram_gib > 0

    if has_gpu:
        if kv:
            budget = KV_VRAM_FRACTION * vram_gib * _BYTES_PER_GIB
            candidate = _largest_ctx_within(kv, budget, HARD_CEILING)
        else:
            candidate = _vram_tier_fallback(vram_gib)
    else:
        ram_gib = hardware.get("available_ram_gb") or 0
        if kv and ram_gib > 0:
            budget = KV_VRAM_FRACTION * ram_gib * _BYTES_PER_GIB
            candidate = _largest_ctx_within(kv, budget, CPU_CEILING)
        else:
            candidate = CPU_CEILING
        candidate = min(candidate, CPU_CEILING)

    # Snap to the ladder, then hard-cap at the trained ceiling (covers the rare
    # trained < FLOOR case, where snap_down would otherwise round up to 2048).
    return min(_snap_down(min(candidate, ceiling)), ceiling)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest tests/test_serve_tuning.py --import-mode=importlib -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/localmodels/serve_tuning.py tests/test_serve_tuning.py
git commit -m "feat(localmodels): recommend_context — pure serve-context fit

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Wire the recommender into `LocalModelManager.start`

**Files:**
- Modify: `src/localmodels/manager.py` (imports; `__init__` two injectable deps; `start` computes + passes `ctx_size`; store `self._served_ctx`)
- Modify: `tests/test_localmodels_manager.py` (inject deterministic deps into the `make_manager` helper + the one standalone `LocalModelManager(...)` so existing `start()` tests stay hermetic — they must not shell out to the real `detect_system`)
- Test: `tests/test_manager_ctx_autofit.py`

**Interfaces:**
- Consumes (Tasks 1-2): `src.gguf_meta.read_gguf_metadata`; `src.localmodels.serve_tuning.recommend_context`; and the existing `build_serve_argv(binary, model_path, port, ctx_size=…, host=…, device=…, mmproj=…)` (in `src/localmodels/runtime.py` — DO NOT modify it; it already takes `ctx_size` and still adds no `-ngl`).
- Produces: `LocalModelManager` gains `metadata_reader` + `hardware_detect` constructor params (defaulting to the real ones); `start()` serves with the fitted context and stores it in `self._served_ctx`. `status()` is NOT changed (an existing test asserts its exact shape; the chosen context is visible via the argv, which llama-server echoes to the captured serve log).

- [ ] **Step 1: Write the failing test**

Create `tests/test_manager_ctx_autofit.py`:

```python
from src.localmodels.manager import LocalModelManager


class _FakeProc:
    pid = 4321
    def poll(self):
        return None  # still running


def _mgr(**kw):
    captured = {}
    def spy_spawn(argv):
        captured["argv"] = argv
        return _FakeProc()
    mgr = LocalModelManager(
        spawn=spy_spawn,
        port_chooser=lambda: 9000,
        resolve_binary=lambda device: "llama-server",
        probe=lambda url: True,          # ready on first poll
        register_endpoint=None,
        **kw,
    )
    return mgr, captured


def test_start_passes_recommended_ctx_and_no_ngl():
    # heavy 24GB config → recommend_context returns 16384 (see serve_tuning tests)
    mgr, captured = _mgr(
        metadata_reader=lambda p: {"context_length": 131072, "block_count": 32,
                                   "head_count": 32, "head_count_kv": 32,
                                   "embedding_length": 4096},
        hardware_detect=lambda: {"has_gpu": True, "gpu_vram_gb": 24, "available_ram_gb": 64},
    )
    mgr.start("model.gguf", device="gpu")
    argv = captured["argv"]
    assert "--ctx-size" in argv
    assert argv[argv.index("--ctx-size") + 1] == "16384"
    assert "-ngl" not in argv and "--n-gpu-layers" not in argv
    assert mgr._served_ctx == 16384


def test_start_degrades_when_detect_and_read_raise():
    def boom(*a, **k):
        raise RuntimeError("probe failed")
    mgr, captured = _mgr(metadata_reader=boom, hardware_detect=boom)
    mgr.start("model.gguf", device="cpu")
    argv = captured["argv"]
    # both sources failed → recommend_context still returns a safe ladder default
    assert "--ctx-size" in argv
    ctx = int(argv[argv.index("--ctx-size") + 1])
    assert ctx in (2048, 4096, 8192, 16384, 32768)
```

Note: this task does NOT change `status()`. An existing test (`test_localmodels_manager.py:117`) asserts `status()`'s exact dict with `==`; adding a key there would break it. The chosen context is exposed as `self._served_ctx` (asserted above) and is visible in the argv (which llama-server echoes to the serve log). `_FakeProc.poll()` returns `None` so the manager's liveness check treats it as running.

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_manager_ctx_autofit.py --import-mode=importlib -q`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'metadata_reader'`).

- [ ] **Step 3: Wire the manager**

In `src/localmodels/manager.py`:

(a) Add imports near the existing `from src.localmodels.runtime import (...)` block:

```python
from src.gguf_meta import read_gguf_metadata
from src.localmodels.serve_tuning import recommend_context


def _default_hardware_detect():
    """Detect hardware for context fitting. Lazy import keeps hwfit off the
    module-load path (it shells out to detect GPUs)."""
    from services.hwfit.hardware import detect_system
    return detect_system()
```

(b) Extend `__init__` (the signature at line 86-90 and the body at 91-107) with two injectable deps, mirroring the existing injection style:

```python
    def __init__(self, spawn=None, port_chooser=None, probe=None,
                 register_endpoint=None, unregister_endpoint=None,
                 resolve_binary=resolve_llama_binary, log_path=None,
                 sleep=None, now=None, ready_timeout=45.0, probe_interval=0.5,
                 sec_per_gb=12.0, force_kill=None,
                 metadata_reader=None, hardware_detect=None):
        # ... existing assignments unchanged ...
        self._metadata_reader = metadata_reader or read_gguf_metadata
        self._hardware_detect = hardware_detect or _default_hardware_detect
        self._served_ctx = None
```

(c) Add a helper method that never raises:

```python
    def _fit_context(self, model_path: str) -> int:
        try:
            meta = self._metadata_reader(model_path)
        except Exception:
            meta = {}
        try:
            hardware = self._hardware_detect()
        except Exception:
            hardware = {}
        return recommend_context(meta if isinstance(meta, dict) else {},
                                 hardware if isinstance(hardware, dict) else {})
```

(d) In `start()` (lines 160-162), compute the context and pass it to `build_serve_argv` (this is the only change to the spawn call — still no `-ngl`):

```python
            mmproj = find_mmproj(model_path)
            ctx = self._fit_context(model_path)
            self._served_ctx = ctx
            proc = self._spawn(build_serve_argv(binary, model_path, port,
                                                ctx_size=ctx, device=device,
                                                mmproj=mmproj))
```

Do NOT modify `status()` or the `_state` dict — an existing test asserts `status()`'s exact shape, and the context is already visible via the argv (llama-server echoes `n_ctx` to the captured serve log).

(e) Keep the existing manager tests hermetic. In `tests/test_localmodels_manager.py`, the `make_manager` helper (and the one standalone `LocalModelManager(...)` around line 97-108, the "stubborn proc" stop test) construct a manager that now calls `detect_system` during `start()`. Inject deterministic deps into EVERY `LocalModelManager(...)` construction in that file so no test shells out to real hardware detection:

```python
        metadata_reader=lambda p: {},
        hardware_detect=lambda: {"has_gpu": False, "available_ram_gb": 16},
```

With `meta={}` and CPU hardware, `recommend_context` returns `8192` (the CPU default) — deterministic. No existing assertion changes: `status()` is unchanged, and no existing test asserts the spawned argv's contents (grep the file to confirm before running).

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest tests/test_manager_ctx_autofit.py --import-mode=importlib -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the manager + feature regression tests (no regression)**

Run: `python -m pytest tests/test_localmodels_manager.py tests/test_localmodels_runtime.py tests/test_gguf_metadata.py tests/test_serve_tuning.py tests/test_manager_ctx_autofit.py --import-mode=importlib -q`
Expected: PASS (the existing manager/runtime tests stay green with the injected deterministic deps + the three feature files green).

Then confirm the app still imports:
Run: `python -c "import app"`
Expected: no error.

- [ ] **Step 6: Commit**

```bash
git add src/localmodels/manager.py tests/test_manager_ctx_autofit.py tests/test_localmodels_manager.py
git commit -m "feat(localmodels): auto-fit serve context to model + hardware

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **All TDD.** Task 1 builds tiny GGUF headers in-test (no real model file). Task 2 is a pure function over injected dicts. Task 3 injects a fake metadata reader + fake `detect_system` + a spy spawn (the manager already supports injection).
- **The `-ngl` prohibition is load-bearing.** Never add `-ngl`/`--n-gpu-layers`; `build_serve_argv` must stay unmodified. Task 3's test asserts the argv still has no `-ngl`.
- **Units are GiB.** Convert `gpu_vram_gb`/`available_ram_gb` to bytes with `* 1024**3`. A `* 1e9` bug would over-estimate the budget by ~7% and is easy to miss — the VRAM-sensitivity test pins the expected ladder values, so a unit error will fail it.
- **No new dependencies, no `Assist.spec` change** (`gguf_meta`, `serve_tuning`, `manager` are all already bundled). A frozen boot-check after a rebuild confirms serving still starts; expected clean.
- **Owed by the user (manual):** serve a real model on the 6GB card and confirm the logged context + observed speed are sensible — the automated tests prove the math and the wiring, not real-hardware tok/s.
- **Scope:** context only. Do NOT add `-ngl`, flash-attn, batch, tensor-split, benchmarking, a Cookbook-path change, or a user-facing override UI (all explicit non-goals).
