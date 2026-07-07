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
