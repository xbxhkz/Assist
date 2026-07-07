"""Pure helpers for native image-model serving via stable-diffusion.cpp's
sd-server. No process launching or DB access here — just binary resolution,
command construction, and filesystem listing, so this module is unit-testable.
"""
import os
import shutil
import sys


def local_image_endpoint_url(port: int) -> str:
    """OpenAI-compatible base URL for a locally served image model (loopback)."""
    return f"http://127.0.0.1:{port}/v1"


def build_serve_argv(binary, files, port, device="cpu", host="127.0.0.1", threads=0):
    """sd-server argv for a FLUX GGUF. `files` = diffusion_model + the three
    shared aux files (t5xxl, clip_l, vae). On GPU the big text encoders + VAE
    stay on CPU so a small (6GB) card can host the diffusion model."""
    argv = [
        binary,
        "--diffusion-model", files["diffusion_model"],
        "--t5xxl", files["t5xxl"],
        "--clip_l", files["clip_l"],
        "--vae", files["vae"],
        "--host", host,
        "--port", str(port),
    ]
    if device == "gpu":
        argv += ["--clip-on-cpu", "--vae-on-cpu", "--diffusion-fa"]
    elif threads:
        argv += ["-t", str(threads)]
    return argv


def _bundled_name() -> str:
    return "sd-server.exe" if os.name == "nt" else "sd-server"


def resolve_sd_binary(device="cpu", path_lookup=shutil.which, frozen_base=None,
                      dev_base=None) -> str:
    """Resolve the sd-server executable.

    Preference: (1) a user-installed `sd-server` on PATH (possibly a CUDA build);
    (2) the bundled binary at `<frozen_base>/sd/<vulkan|cpu>/<name>` when frozen;
    (3) a dev fallback at `<repo>/build_assets/sd/<vulkan|cpu>/<name>`. Raises if
    none. `device="gpu"` selects the Vulkan build, else the CPU build."""
    found = path_lookup("sd-server") or path_lookup("sd-server.exe")
    if found:
        return found

    sub = "vulkan" if device == "gpu" else "cpu"
    name = _bundled_name()

    base = frozen_base
    if base is None and getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None)
    if base:
        cand = os.path.join(base, "sd", sub, name)
        if os.path.isfile(cand):
            return cand

    if dev_base is None:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dev_base = os.path.join(repo, "build_assets", "sd", sub)
    cand = os.path.join(dev_base, name)
    if os.path.isfile(cand):
        return cand

    raise RuntimeError(
        "sd-server not found: no server on PATH and no bundled binary under "
        "sd/<vulkan|cpu>/."
    )


def list_gguf_image_models(models_dir: str) -> list:
    """List `.gguf` diffusion models in `models_dir` as [{name, path, size}]."""
    out = []
    if os.path.isdir(models_dir):
        for fn in sorted(os.listdir(models_dir)):
            if fn.lower().endswith(".gguf"):
                p = os.path.join(models_dir, fn)
                out.append({"name": fn, "path": p, "size": os.path.getsize(p)})
    return out
