from src.localmodels.runtime import build_serve_argv


def test_lora_and_alias_appended():
    argv = build_serve_argv("llama-server", "C:/m/base.gguf", 8100,
                            lora="C:/m/adapter.gguf", alias="tuned-x")
    assert "--lora" in argv and argv[argv.index("--lora") + 1] == "C:/m/adapter.gguf"
    assert argv[argv.index("--alias") + 1] == "tuned-x"


def test_no_lora_by_default_and_alias_falls_back_to_basename():
    argv = build_serve_argv("llama-server", "C:/m/base.gguf", 8100)
    assert "--lora" not in argv
    assert argv[argv.index("--alias") + 1] == "base.gguf"


from src.localmodels.manager import LocalModelManager


class FakeProc:
    def poll(self): return None
    def terminate(self): pass
    def kill(self): pass
    def wait(self, timeout=None): return 0
    @property
    def pid(self): return 1234


def test_manager_start_threads_lora_and_registers_alias():
    captured, reg = {}, {}
    mgr = LocalModelManager(
        spawn=lambda argv: (captured.__setitem__("argv", argv), FakeProc())[1],
        port_chooser=lambda: 8100,
        probe=lambda url: True,
        register_endpoint=lambda name, base_url: (reg.__setitem__("name", name), "eid-1")[1],
        unregister_endpoint=lambda e: None,
        resolve_binary=lambda device="cpu": "/bin/llama-server",
        metadata_reader=lambda p: {},
        hardware_detect=lambda: {"has_gpu": False, "available_ram_gb": 16},
    )
    mgr.start("/models/base.gguf", device="cpu",
              lora="/models/adapter.gguf", alias="qwen · run-1 (LoRA)")
    argv = captured["argv"]
    assert "--lora" in argv and argv[argv.index("--lora") + 1] == "/models/adapter.gguf"
    assert reg["name"] == "qwen · run-1 (LoRA)"
