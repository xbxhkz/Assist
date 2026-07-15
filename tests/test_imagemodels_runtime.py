import os
import src.imagemodels.runtime as rt


def test_endpoint_url():
    assert rt.local_image_endpoint_url(8200) == "http://127.0.0.1:8200/v1"


def test_build_argv_flux_four_files_cpu():
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/clip.safetensors", "vae": "/m/ae.safetensors"}
    argv = rt.build_serve_argv("/x/sd-server", files, 8200, device="cpu", threads=8)
    for f in ("--diffusion-model", "/m/flux.gguf", "--t5xxl", "/m/t5.gguf",
              "--clip_l", "/m/clip.safetensors", "--vae", "/m/ae.safetensors",
              "--listen-ip", "127.0.0.1", "--listen-port", "8200", "-t", "8"):
        assert f in argv
    assert "--auto-fit" not in argv


def test_build_argv_flux_guidance_distilled_cfg():
    """FLUX models are guidance-distilled: sd-server's default cfg 7.0 doubles
    inference and washes out output, so the serve bakes in cfg 1.0."""
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/clip.safetensors", "vae": "/m/ae.safetensors"}
    argv = rt.build_serve_argv("/x/sd-server", files, 8200)
    assert argv[argv.index("--cfg-scale") + 1] == "1.0"
    assert "--steps" not in argv  # sd-server default (20) is right for FLUX.1


def test_build_argv_gpu_low_vram_layout():
    """GPU serves use sd.cpp's offload-to-cpu recipe (weights in RAM, streamed
    to VRAM per use) so models larger than the card still run at GPU speed.
    --auto-fit's VAE fallback and a hard --backend pin both failed live."""
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/clip.safetensors", "vae": "/m/ae.safetensors"}
    argv = rt.build_serve_argv("/x/sd-server", files, 8200, device="gpu")
    assert "--offload-to-cpu" in argv and "--diffusion-fa" in argv
    assert "--auto-fit" not in argv and "--backend" not in argv


def test_build_argv_always_tiles_vae():
    """1024x1024 VAE decode wants an 8.5GB compute buffer — tiling keeps it
    inside both 6GB VRAM and small-RAM machines, on every device."""
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/clip.safetensors", "vae": "/m/ae.safetensors"}
    assert "--vae-tiling" in rt.build_serve_argv("/x/sd", files, 8200, device="cpu")
    assert "--vae-tiling" in rt.build_serve_argv("/x/sd", files, 8200, device="gpu")


def test_build_argv_flux2_uses_llm_encoder():
    """FLUX.2 (klein) file set: --llm replaces --t5xxl/--clip_l, and the
    distilled klein wants 4 steps at cfg 1.0 as server defaults."""
    files = {"diffusion_model": "/m/flux2-klein.gguf", "llm": "/m/qwen3-4b.gguf",
             "vae": "/m/flux2_ae.safetensors"}
    argv = rt.build_serve_argv("/x/sd-server", files, 8200, device="cpu", threads=8)
    for f in ("--diffusion-model", "/m/flux2-klein.gguf", "--llm", "/m/qwen3-4b.gguf",
              "--vae", "/m/flux2_ae.safetensors", "-t", "8"):
        assert f in argv
    assert "--t5xxl" not in argv and "--clip_l" not in argv
    assert argv[argv.index("--cfg-scale") + 1] == "1.0"
    assert argv[argv.index("--steps") + 1] == "4"


def test_build_argv_flux2_gpu_low_vram_layout():
    files = {"diffusion_model": "/m/flux2-klein.gguf", "llm": "/m/qwen3-4b.gguf",
             "vae": "/m/flux2_ae.safetensors"}
    argv = rt.build_serve_argv("/x/sd-server", files, 8200, device="gpu")
    assert "--offload-to-cpu" in argv and "--diffusion-fa" in argv
    assert "--auto-fit" not in argv and "--backend" not in argv


def test_resolve_prefers_path_binary():
    got = rt.resolve_sd_binary(device="cpu",
        path_lookup=lambda n: "/usr/bin/sd-server" if n == "sd-server" else None)
    assert got == "/usr/bin/sd-server"


def test_resolve_uses_bundled_gpu(tmp_path):
    b = tmp_path / "sd" / "vulkan"; b.mkdir(parents=True)
    name = "sd-server.exe" if os.name == "nt" else "sd-server"
    (b / name).write_text("x")
    got = rt.resolve_sd_binary(device="gpu", path_lookup=lambda n: None, frozen_base=str(tmp_path))
    assert got == str(b / name)


def test_list_filters_gguf(tmp_path):
    (tmp_path / "flux.gguf").write_bytes(b"x")
    (tmp_path / "note.txt").write_text("n")
    got = rt.list_gguf_image_models(str(tmp_path))
    assert [m["name"] for m in got] == ["flux.gguf"]


def _gguf_with_arch(arch: str) -> bytes:
    import struct
    kb, vb = b"general.architecture", arch.encode()
    kv = struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1) + kv


def test_list_image_arch_ggufs_filters_by_header(tmp_path):
    """Only diffusion-architecture ggufs qualify — LLMs and unparseable files
    in the shared download dir must not appear in the image picker."""
    (tmp_path / "flux1-dev.gguf").write_bytes(_gguf_with_arch("flux"))
    (tmp_path / "chat.gguf").write_bytes(_gguf_with_arch("llama"))
    (tmp_path / "junk.gguf").write_bytes(b"x")
    got = rt.list_image_arch_ggufs(str(tmp_path))
    assert [m["name"] for m in got] == ["flux1-dev.gguf"]


def test_list_image_arch_ggufs_missing_dir_is_empty():
    assert rt.list_image_arch_ggufs("/no/such/dir") == []


def test_build_argv_taesd_flag():
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/c.safetensors", "vae": "/m/v.safetensors",
             "taesd": "/m/taef1.safetensors"}
    argv = rt.build_serve_argv("/x/sd", files, 8200)
    assert argv[argv.index("--taesd") + 1] == "/m/taef1.safetensors"


def test_build_argv_no_taesd_by_default():
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/c.safetensors", "vae": "/m/v.safetensors"}
    assert "--taesd" not in rt.build_serve_argv("/x/sd", files, 8200)


def test_build_argv_zimage_turbo_default_steps():
    """Z-Image turbo is distilled to ~8 steps (sd.cpp docs/z_image.md)."""
    files = {"diffusion_model": "/m/z-image-turbo-Q5_0.gguf",
             "llm": "/m/Qwen3-4B-Q4_K_M.gguf", "vae": "/m/ae.safetensors"}
    argv = rt.build_serve_argv("/x/sd", files, 8200)
    assert argv[argv.index("--steps") + 1] == "8"
    assert "--llm" in argv and "--t5xxl" not in argv


def test_build_argv_explicit_steps_override():
    files = {"diffusion_model": "/m/flux2-klein.gguf", "llm": "/m/q.gguf",
             "vae": "/m/v.safetensors"}
    argv = rt.build_serve_argv("/x/sd", files, 8200, steps=8)
    assert argv[argv.index("--steps") + 1] == "8"


def test_build_argv_flux1_explicit_steps():
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/c.safetensors", "vae": "/m/v.safetensors"}
    argv = rt.build_serve_argv("/x/sd", files, 8200, steps=12)
    assert argv[argv.index("--steps") + 1] == "12"


def test_looks_like_flux2():
    assert rt.looks_like_flux2("flux-2-klein-4b-Q8_0.gguf")
    assert rt.looks_like_flux2("FLUX.2-dev-Q4_K.gguf")
    assert rt.looks_like_flux2("flux2_whatever.gguf")
    assert not rt.looks_like_flux2("flux1-dev-Q3_K_S.gguf")
    assert not rt.looks_like_flux2("flux-dev.gguf")
    assert not rt.looks_like_flux2("")


def test_looks_like_chroma():
    assert rt.looks_like_chroma("Chroma1-HD-Q4_K_M.gguf")
    assert rt.looks_like_chroma("chroma-unlocked-v37.gguf")
    assert not rt.looks_like_chroma("flux1-dev.gguf")
    assert not rt.looks_like_chroma("")


def test_build_argv_chroma_t5_vae_no_clip_real_cfg():
    """Chroma drops CLIP (T5 only) and is NOT guidance-distilled: no --clip_l,
    real cfg (~4), extra steps, and the DiT mask disabled per the sd.cpp recipe."""
    files = {"diffusion_model": "/m/Chroma1-HD-Q4_K_M.gguf",
             "t5xxl": "/m/t5.gguf", "vae": "/m/ae.safetensors"}
    argv = rt.build_serve_argv("/x/sd", files, 8200, device="gpu")
    for f in ("--diffusion-model", "/m/Chroma1-HD-Q4_K_M.gguf",
              "--t5xxl", "/m/t5.gguf", "--vae", "/m/ae.safetensors"):
        assert f in argv
    assert "--clip_l" not in argv
    assert argv[argv.index("--cfg-scale") + 1] == "4.0"
    assert "chroma_use_dit_mask=0" in argv
    assert int(argv[argv.index("--steps") + 1]) >= 20
    assert "--offload-to-cpu" in argv and "--vae-tiling" in argv


def test_build_argv_chroma_steps_override():
    files = {"diffusion_model": "/m/chroma.gguf", "t5xxl": "/m/t5.gguf",
             "vae": "/m/ae.safetensors"}
    argv = rt.build_serve_argv("/x/sd", files, 8200, steps=30)
    assert argv[argv.index("--steps") + 1] == "30"


def test_build_argv_checkpoint_uses_dash_m_no_flash_attn():
    """An all-in-one SD/SDXL checkpoint serves via -m (embedded encoders+VAE),
    at real cfg, and WITHOUT flash attention — --diffusion-fa crashes SDXL on
    the bundled Vulkan build (verified live)."""
    files = {"checkpoint": "/m/juggernaut-xl-v9-Q8_0.gguf"}
    argv = rt.build_serve_argv("/x/sd", files, 8200, device="gpu")
    assert "-m" in argv and argv[argv.index("-m") + 1] == "/m/juggernaut-xl-v9-Q8_0.gguf"
    assert "--diffusion-model" not in argv
    assert "--t5xxl" not in argv and "--clip_l" not in argv and "--llm" not in argv
    assert argv[argv.index("--cfg-scale") + 1] == "7.0"
    assert "--offload-to-cpu" in argv and "--vae-tiling" in argv
    assert "--diffusion-fa" not in argv


def test_build_argv_checkpoint_cpu_threads_no_flash_attn():
    files = {"checkpoint": "/m/sdxl.gguf"}
    argv = rt.build_serve_argv("/x/sd", files, 8200, device="cpu", threads=8)
    assert argv[argv.index("-t") + 1] == "8"
    assert "--diffusion-fa" not in argv


def test_build_argv_checkpoint_lightning_low_cfg_few_steps():
    """Distilled few-step SDXL (Lightning/Turbo/Hyper) burns out at cfg 7 —
    detect it by name and drop to low cfg + few steps."""
    files = {"checkpoint": "/m/RealVisXL_V5.0_Lightning_q5_1.gguf"}
    argv = rt.build_serve_argv("/x/sd", files, 8200, device="gpu")
    assert argv[argv.index("--cfg-scale") + 1] == "2.0"
    assert int(argv[argv.index("--steps") + 1]) <= 8
    assert "--diffusion-fa" not in argv


def test_list_image_arch_ggufs_includes_arch_less_full_checkpoint(tmp_path):
    """An SDXL all-in-one checkpoint has NO general.architecture tag, but its
    embedded text encoder marks it as an image checkpoint — it must still list."""
    import struct
    def tinfo(name):
        nb = name.encode()
        return (struct.pack("<Q", len(nb)) + nb + struct.pack("<I", 1) +
                struct.pack("<Q", 16) + struct.pack("<I", 0) + struct.pack("<Q", 0))
    names = ["conditioner.embedders.0.transformer.text_model.x.weight",
             "model.diffusion_model.y.weight"]
    blob = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(names)) +
            struct.pack("<Q", 0) + b"".join(tinfo(n) for n in names))
    (tmp_path / "juggernaut-xl.gguf").write_bytes(blob)
    (tmp_path / "chat.gguf").write_bytes(_gguf_with_arch("llama"))
    got = rt.list_image_arch_ggufs(str(tmp_path))
    assert [m["name"] for m in got] == ["juggernaut-xl.gguf"]


def test_fmt_gb_integer_and_fractional():
    assert rt._fmt_gb(5) == "5"
    assert rt._fmt_gb(5.0) == "5"
    assert rt._fmt_gb(4.5) == "4.5"


def test_build_argv_gpu_max_vram_fills_and_streams():
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/c.safetensors", "vae": "/m/v.safetensors"}
    argv = rt.build_serve_argv("/x/sd", files, 8200, device="gpu", max_vram_gb=5)
    assert argv[argv.index("--max-vram") + 1] == "5"
    assert "--stream-layers" in argv
    # --offload-to-cpu (weights in the CPU params backend) is REQUIRED alongside
    # --max-vram: residency streaming only works when params live in RAM. Without
    # it, sd.cpp loads every weight to VRAM and OOMs (verified live).
    assert "--offload-to-cpu" in argv
    assert "--diffusion-fa" in argv  # FLUX.1 keeps flash attention


def test_build_argv_gpu_no_budget_keeps_offload():
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/c.safetensors", "vae": "/m/v.safetensors"}
    argv = rt.build_serve_argv("/x/sd", files, 8200, device="gpu", max_vram_gb=None)
    assert "--offload-to-cpu" in argv and "--max-vram" not in argv


def test_build_argv_checkpoint_max_vram_no_flash_attn():
    files = {"checkpoint": "/m/juggernaut-xl-v9-Q8_0.gguf"}
    argv = rt.build_serve_argv("/x/sd", files, 8200, device="gpu", max_vram_gb=5)
    assert "--max-vram" in argv and "--stream-layers" in argv
    assert "--diffusion-fa" not in argv


def test_build_argv_cpu_ignores_max_vram():
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/c.safetensors", "vae": "/m/v.safetensors"}
    argv = rt.build_serve_argv("/x/sd", files, 8200, device="cpu", threads=8, max_vram_gb=5)
    assert "--max-vram" not in argv and "--offload-to-cpu" not in argv
    assert argv[argv.index("-t") + 1] == "8"


def test_build_argv_sets_lora_model_dir(monkeypatch, tmp_path):
    import src.imagemodels.loras as loras
    monkeypatch.setattr(loras, "IMAGE_MODELS_DIR", str(tmp_path))
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/c.safetensors", "vae": "/m/v.safetensors"}
    for dev in ("cpu", "gpu"):
        argv = rt.build_serve_argv("/x/sd", files, 8200, device=dev)
        assert argv[argv.index("--lora-model-dir") + 1] == loras.loras_dir()
        assert "--vae-tiling" in argv     # existing flag still present
