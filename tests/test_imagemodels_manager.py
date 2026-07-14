"""State-machine tests for ImageModelManager using injected fakes."""
import itertools

from src.imagemodels.manager import ImageModelManager

FILES = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
         "clip_l": "/m/clip.safetensors", "vae": "/m/ae.safetensors"}


class FakeProc:
    def __init__(self, pid=4321, exit_code=None):
        self.pid = pid
        self._exit_code = exit_code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._exit_code

    def terminate(self):
        self.terminated = True
        if self._exit_code is None:
            self._exit_code = -15

    def wait(self, timeout=None):
        pass

    def kill(self):
        self.killed = True
        if self._exit_code is None:
            self._exit_code = -9


def make_manager(ready=True, spawned=None, registered=None, unregistered=None,
                 proc_exit_code=None, vram=6.0, ready_after=None):
    spawned = spawned if spawned is not None else []
    registered = registered if registered is not None else []
    unregistered = unregistered if unregistered is not None else []
    if ready_after is None:
        ready_after = 0 if ready else 999  # 0 = first attempt ready; 999 = never

    def spawn(argv):
        p = FakeProc(exit_code=proc_exit_code)
        spawned.append((argv, p))
        return p

    def register(name, base_url):
        eid = f"img-local-{len(registered)}"
        registered.append({"name": name, "base_url": base_url, "id": eid})
        return eid

    def probe(url):
        # The latest spawn is the current attempt (0-based index).
        return (len(spawned) - 1) >= ready_after

    clock = itertools.count(0, 10)
    mgr = ImageModelManager(
        spawn=spawn, port_chooser=lambda: 8200, probe=probe,
        register_endpoint=register, unregister_endpoint=lambda eid: unregistered.append(eid),
        resolve_binary=lambda device: f"/bin/sd-{device}",
        log_path="/nonexistent/sd-server.log",
        sleep=lambda _s: None, now=lambda: next(clock), ready_timeout=45.0,
        vram_probe=lambda: vram)
    return mgr, spawned, registered, unregistered


def test_start_launches_and_registers():
    mgr, spawned, registered, _ = make_manager()
    st = mgr.start(FILES, device="cpu")
    assert st["running"] is True and st["model"] == "flux.gguf"
    assert st["port"] == 8200 and st["endpoint_id"] == "img-local-0"
    assert registered[0]["base_url"] == "http://127.0.0.1:8200/v1"


def test_start_with_full_checkpoint_dict():
    """An all-in-one SD/SDXL checkpoint dict carries 'checkpoint' and no
    'diffusion_model' key — start() must resolve the model path from either,
    not KeyError (which surfaced as a 500 when serving SDXL)."""
    mgr, spawned, registered, _ = make_manager()
    st = mgr.start({"checkpoint": "/m/juggernaut-xl-v9-Q8_0.gguf"}, device="gpu")
    assert st["running"] is True and st["model"] == "juggernaut-xl-v9-Q8_0.gguf"
    assert registered[0]["name"] == "juggernaut-xl-v9-Q8_0.gguf"
    assert "-m" in spawned[0][0]  # served via -m (checkpoint), not --diffusion-model


def test_start_readiness_failure_kills_and_raises():
    import pytest
    mgr, spawned, registered, _ = make_manager(ready=False)
    with pytest.raises(RuntimeError, match="did not become ready"):
        mgr.start(FILES)
    assert spawned[0][1].terminated is True
    assert registered == []
    assert mgr.status()["running"] is False


def test_start_fails_fast_on_early_exit():
    import pytest
    mgr, spawned, registered, _ = make_manager(ready=True, proc_exit_code=1)
    with pytest.raises(RuntimeError, match="exited on startup"):
        mgr.start(FILES)
    assert registered == []


def test_start_gpu_tier1_fills_vram():
    mgr, spawned, registered, _ = make_manager(vram=6.0)
    st = mgr.start(FILES, device="gpu")
    assert st["running"] is True and st["device"] == "gpu"
    argv = spawned[0][0]
    assert argv[argv.index("--max-vram") + 1] == "5"  # 6.0 - 1.0 margin
    # offload-to-cpu is required alongside --max-vram for residency streaming
    assert "--stream-layers" in argv and "--offload-to-cpu" in argv
    assert len(spawned) == 1


def test_start_gpu_falls_back_to_offload():
    mgr, spawned, registered, _ = make_manager(vram=6.0, ready_after=1)
    st = mgr.start(FILES, device="gpu")
    assert st["running"] is True and st["device"] == "gpu"
    assert "--max-vram" in spawned[0][0]           # Tier 1 attempted
    assert "--offload-to-cpu" in spawned[1][0]     # Tier 2 succeeded
    assert spawned[0][0][0] == "/bin/sd-gpu" and spawned[1][0][0] == "/bin/sd-gpu"
    assert len(spawned) == 2


def test_start_gpu_falls_back_to_cpu():
    mgr, spawned, registered, _ = make_manager(vram=6.0, ready_after=2)
    st = mgr.start(FILES, device="gpu")
    assert st["running"] is True and st["device"] == "cpu"
    assert spawned[2][0][0] == "/bin/sd-cpu"       # Tier 3 uses the CPU binary
    assert "--max-vram" not in spawned[2][0] and "--offload-to-cpu" not in spawned[2][0]
    assert len(spawned) == 3


def test_start_gpu_no_budget_skips_tier1():
    mgr, spawned, registered, _ = make_manager(vram=None)
    st = mgr.start(FILES, device="gpu")
    assert st["running"] is True
    assert "--offload-to-cpu" in spawned[0][0] and "--max-vram" not in spawned[0][0]
    assert len(spawned) == 1


def test_start_cpu_does_not_probe_vram():
    probed = []
    clock = itertools.count(0, 10)
    spawned = []
    mgr = ImageModelManager(
        spawn=lambda argv: (spawned.append((argv, FakeProc())) or spawned[-1][1]),
        port_chooser=lambda: 8200, probe=lambda url: True,
        register_endpoint=lambda name, base_url: "eid",
        unregister_endpoint=lambda eid: None,
        resolve_binary=lambda device: f"/bin/sd-{device}",
        log_path="/nonexistent/sd-server.log", sleep=lambda _s: None,
        now=lambda: next(clock),
        vram_probe=lambda: probed.append(1))
    st = mgr.start(FILES, device="cpu")
    assert st["running"] is True and st["device"] == "cpu"
    assert probed == []            # VRAM probe not consulted for a CPU request
    assert spawned[0][0][0] == "/bin/sd-cpu"


def test_start_gpu_all_tiers_fail_raises():
    import pytest
    mgr, spawned, registered, _ = make_manager(vram=6.0, ready_after=999)
    with pytest.raises(RuntimeError, match="did not become ready"):
        mgr.start(FILES, device="gpu")
    assert registered == [] and len(spawned) == 3  # tried all three tiers
    assert mgr.status()["running"] is False


def test_stop_terminates_and_unregisters():
    mgr, spawned, registered, unregistered = make_manager()
    mgr.start(FILES)
    st = mgr.stop()
    assert st["running"] is False
    assert spawned[0][1].terminated is True
    assert unregistered == ["img-local-0"]


def test_status_when_idle():
    mgr, *_ = make_manager()
    assert mgr.status() == {"running": False, "model": None, "port": None,
                            "endpoint_id": None, "device": None}


def test_ready_timeout_scales_with_size():
    mgr, *_ = make_manager()
    assert mgr._timeout_for_bytes(2_000_000) == 45.0
    assert mgr._timeout_for_bytes(12_000_000_000) == 12.0 * 12


def test_list_models_merges_image_dir_and_downloaded_flux(tmp_path, monkeypatch):
    """The picker must show FLUX ggufs the user downloaded into the shared
    LLM models dir, not just files manually placed in image-models/."""
    import struct
    import src.imagemodels.manager as mod

    def gguf(arch):
        kb, vb = b"general.architecture", arch.encode()
        kv = struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb
        return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1) + kv

    img_dir = tmp_path / "image-models"; img_dir.mkdir()
    dl_dir = tmp_path / "models"; dl_dir.mkdir()
    (img_dir / "manual.gguf").write_bytes(b"x")            # placed by hand: listed as-is
    (dl_dir / "flux1-dev.gguf").write_bytes(gguf("flux"))  # downloaded image model
    (dl_dir / "chat.gguf").write_bytes(gguf("llama"))      # LLM: excluded
    monkeypatch.setattr(mod, "IMAGE_MODELS_DIR", str(img_dir))
    monkeypatch.setattr(mod, "MODELS_DIR", str(dl_dir))

    mgr, *_ = make_manager()
    names = [m["name"] for m in mgr.list_models()]
    assert names == ["manual.gguf", "flux1-dev.gguf"]


def test_terminate_escalates_to_force_kill():
    killed = []

    class StubbornProc:
        pid = 9999
        def poll(self): return None
        def terminate(self): pass
        def wait(self, timeout=None): pass
        def kill(self): pass

    mgr = ImageModelManager(
        spawn=lambda argv: StubbornProc(), port_chooser=lambda: 8200,
        probe=lambda url: True, register_endpoint=lambda name, base_url: "e",
        unregister_endpoint=lambda eid: None, resolve_binary=lambda device: "/b",
        log_path="/nonexistent/sd-server.log", sleep=lambda _s: None,
        now=lambda: 0.0, force_kill=lambda pid: killed.append(pid))
    mgr.start(FILES)
    mgr.stop()
    assert 9999 in killed
