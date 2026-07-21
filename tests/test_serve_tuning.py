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


def test_sub_floor_trained_clamped_to_floor():
    # A degenerate model claiming a trained context below FLOOR must still yield
    # a valid ladder value >= FLOOR (never an off-ladder sub-FLOOR number).
    meta = {**_HEAVY, "context_length": 1000}
    ctx = recommend_context(meta, {"has_gpu": True, "gpu_vram_gb": 48})
    assert ctx == FLOOR
    assert ctx in CTX_LADDER
