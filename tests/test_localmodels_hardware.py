"""Unit tests for hardware detection + hybrid model-fit (hwfit injected)."""
import pytest

import src.localmodels.hardware as hw


@pytest.fixture(autouse=True)
def _reset_cache():
    hw._hw_cache = None
    yield
    hw._hw_cache = None


def _fake_detect():
    return {"available_ram_gb": 16.0, "has_gpu": True,
            "gpu_name": "RTX 3060", "gpu_vram_gb": 8.0}


def test_get_hardware_normalizes():
    got = hw.get_hardware(detect=_fake_detect, refresh=True)
    assert got == {"ram_gb": 16.0, "has_gpu": True,
                   "gpu_name": "RTX 3060", "vram_gb": 8.0}


def test_get_hardware_caches_detection():
    calls = {"n": 0}
    def det():
        calls["n"] += 1
        return _fake_detect()
    hw.get_hardware(detect=det, refresh=True)
    hw.get_hardware(detect=det)   # cached — no second detect call
    assert calls["n"] == 1


def test_get_hardware_safe_default_on_error():
    def boom():
        raise RuntimeError("no hw")
    got = hw.get_hardware(detect=boom, refresh=True)
    assert got == {"ram_gb": 0.0, "has_gpu": False, "gpu_name": None, "vram_gb": 0.0}


def test_infer_params_b():
    assert hw._infer_params_b("Qwen2.5-7B-Instruct-Q4_K_M.gguf") == 7.0
    assert hw._infer_params_b("tiny-1.5b-chat.gguf") == 1.5
    assert hw._infer_params_b("model-Q4.gguf") is None


def test_verdict_thresholds():
    gpu_hw = {"has_gpu": True, "vram_gb": 8.0, "ram_gb": 16.0}
    assert hw._verdict(6.0, gpu_hw) == "gpu"
    assert hw._verdict(12.0, gpu_hw) == "ram"      # exceeds vram, fits ram
    assert hw._verdict(20.0, gpu_hw) == "too_big"
    cpu_hw = {"has_gpu": False, "vram_gb": 0.0, "ram_gb": 16.0}
    assert hw._verdict(6.0, cpu_hw) == "ram"       # no gpu → ram path


def test_fit_takes_conservative_max_param_over_size():
    # size ~5GB; injected param estimate 9GB → needed=9 → too big for 8GB vram, fits 16 ram.
    hardware = {"has_gpu": True, "vram_gb": 8.0, "ram_gb": 16.0}
    f = {"filename": "big-7B-Q8.gguf", "size": 5_000_000_000}
    out = hw.fit_for_file(f, hardware, estimate=lambda m, q, c: 9.0)
    assert out["param_estimate_gb"] == 9.0
    assert out["needed_gb"] == 9.0
    assert out["verdict"] == "ram"


def test_fit_uses_size_when_param_larger_is_absent():
    # No param token → size-only; 5GB + kv(4096)=0.5 → 5.5 → fits 8GB gpu.
    hardware = {"has_gpu": True, "vram_gb": 8.0, "ram_gb": 16.0}
    f = {"filename": "model-Q4.gguf", "size": 5_000_000_000}
    out = hw.fit_for_file(f, hardware)
    assert out["param_estimate_gb"] is None
    assert out["needed_gb"] == 5.5
    assert out["verdict"] == "gpu"


def test_recommend_models_injected_rank():
    def fake_rank(system, limit=8):
        return [{"name": "Qwen2.5-7B", "score": 0.9}, {"name": "Phi-3.5", "score": 0.8},
                {"score": 0.1}]  # no name → skipped
    out = hw.recommend_models(limit=8, rank=fake_rank, detect=_fake_detect)
    assert out == [{"name": "Qwen2.5-7B", "score": 0.9}, {"name": "Phi-3.5", "score": 0.8}]


def test_recommend_models_empty_on_error():
    def boom(system, limit=8):
        raise RuntimeError("rank down")
    assert hw.recommend_models(rank=boom, detect=_fake_detect) == []
