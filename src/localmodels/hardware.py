"""Hardware detection + model-fit analysis for native local models (Phase 3c).

Uses services/hwfit READ-ONLY. Detection is cached (it is slow); fit is a
conservative hybrid of a size-based estimate and hwfit's param-based
estimate_memory_gb, taking the max so the app never over-promises a fit.
"""
import re
import threading

_PARAM_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[bB](?![\w])")

_hw_cache = None
_hw_lock = threading.Lock()


def _default_detect():
    from services.hwfit.hardware import detect_system
    return detect_system()


def get_hardware_system(detect=None, refresh=False) -> dict:
    """Raw hwfit system dict, cached process-wide. Safe-defaults to {} on error."""
    global _hw_cache
    with _hw_lock:
        if _hw_cache is None or refresh:
            det = detect or _default_detect
            try:
                _hw_cache = det() or {}
            except Exception:
                _hw_cache = {}
        return _hw_cache


def get_hardware(detect=None, refresh=False) -> dict:
    """Normalized hardware summary for the UI."""
    s = get_hardware_system(detect=detect, refresh=refresh) or {}
    return {
        "ram_gb": float(s.get("available_ram_gb") or 0.0),
        "has_gpu": bool(s.get("has_gpu")),
        "gpu_name": s.get("gpu_name"),
        "vram_gb": float(s.get("gpu_vram_gb") or 0.0),
    }


def _infer_params_b(name):
    m = _PARAM_RE.search(name or "")
    return float(m.group(1)) if m else None


def _kv_overhead(ctx):
    # Rough runtime/KV headroom on top of the weights, scaled by context length.
    return 0.5 * (ctx / 4096.0)


def _verdict(needed_gb, hardware):
    if hardware.get("has_gpu") and needed_gb <= (hardware.get("vram_gb") or 0):
        return "gpu"
    if needed_gb <= (hardware.get("ram_gb") or 0):
        return "ram"
    return "too_big"


def fit_for_file(file, hardware, ctx=4096, estimate=None):
    """Hybrid fit verdict for a downloadable GGUF file (size + hwfit estimate)."""
    size_gb = float(file.get("size") or 0) / 1e9
    size_needed = size_gb + _kv_overhead(ctx)
    param_needed = None
    params = _infer_params_b(file.get("filename") or "")
    if params:
        try:
            from services.hwfit.models import (
                infer_quantization_from_name, estimate_memory_gb,
            )
            est = estimate or estimate_memory_gb
            quant = infer_quantization_from_name(file.get("filename") or "")
            param_needed = float(est({"parameter_count": f"{params}B"}, quant, ctx))
        except Exception:
            param_needed = None
    needed_gb = max(size_needed, param_needed or 0.0)
    return {
        "verdict": _verdict(needed_gb, hardware),
        "needed_gb": round(needed_gb, 2),
        "size_gb": round(size_gb, 2),
        "param_estimate_gb": round(param_needed, 2) if param_needed is not None else None,
    }


def recommend_models(limit=8, rank=None, detect=None):
    """hwfit-ranked model families for the detected machine."""
    system = get_hardware_system(detect=detect)
    try:
        from services.hwfit.fit import rank_models
        r = rank or rank_models
        out = r(system, limit=limit) or []
    except Exception:
        return []
    return [{"name": m.get("name"), "score": m.get("score")}
            for m in out if m.get("name")]
