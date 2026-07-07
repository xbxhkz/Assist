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


def test_build_argv_gpu_autofit():
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/clip.safetensors", "vae": "/m/ae.safetensors"}
    argv = rt.build_serve_argv("/x/sd-server", files, 8200, device="gpu")
    assert "--auto-fit" in argv and "--diffusion-fa" in argv


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


def test_looks_like_flux2():
    assert rt.looks_like_flux2("flux-2-klein-4b-Q8_0.gguf")
    assert rt.looks_like_flux2("FLUX.2-dev-Q4_K.gguf")
    assert rt.looks_like_flux2("flux2_whatever.gguf")
    assert not rt.looks_like_flux2("flux1-dev-Q3_K_S.gguf")
    assert not rt.looks_like_flux2("flux-dev.gguf")
    assert not rt.looks_like_flux2("")
