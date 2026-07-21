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
