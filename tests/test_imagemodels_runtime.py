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
    """GPU serves pin diffusion to the GPU and text encoder + VAE to CPU.
    --auto-fit is NOT used: it ignores explicit --backend assignments and its
    VAE-OOM tiling fallback fails on 6GB cards at 1024x1024 (observed live)."""
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/clip.safetensors", "vae": "/m/ae.safetensors"}
    argv = rt.build_serve_argv("/x/sd-server", files, 8200, device="gpu")
    assert argv[argv.index("--backend") + 1] == "diffusion=vulkan0,te=cpu,vae=cpu"
    assert "--diffusion-fa" in argv and "--auto-fit" not in argv


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
    assert argv[argv.index("--backend") + 1] == "diffusion=vulkan0,te=cpu,vae=cpu"
    assert "--diffusion-fa" in argv and "--auto-fit" not in argv


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
