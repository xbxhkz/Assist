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
