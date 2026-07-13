"""Pure helpers for native image-model serving via stable-diffusion.cpp's
sd-server. No process launching or DB access here — just binary resolution,
command construction, and filesystem listing, so this module is unit-testable.
"""
import os
import re
import shutil
import sys

from src.gguf_meta import read_gguf_architecture, is_image_architecture


def local_image_endpoint_url(port: int) -> str:
    """OpenAI-compatible base URL for a locally served image model (loopback)."""
    return f"http://127.0.0.1:{port}/v1"


def build_serve_argv(binary, files, port, device="cpu", host="127.0.0.1",
                     threads=0, steps=None):
    """sd-server argv for a FLUX GGUF. `files` = diffusion_model + the family's
    aux files: FLUX.1 uses t5xxl/clip_l/vae, FLUX.2 (klein) uses llm/vae (the
    "llm" key selects the FLUX.2 layout). FLUX is guidance-distilled, so the
    server default cfg 7.0 is baked down to 1.0; klein additionally wants 4
    sampling steps instead of the default 20. An explicit `steps` wins over
    the family default."""
    argv = [binary, "--diffusion-model", files["diffusion_model"]]
    base = os.path.basename(files["diffusion_model"]).lower()
    if "llm" in files:
        argv += ["--llm", files["llm"], "--vae", files["vae"],
                 "--cfg-scale", "1.0"]
        if steps:
            eff_steps = steps
        elif "klein" in base:
            eff_steps = 4       # step-distilled (flux2-dev keeps the default 20)
        elif ("z-image" in base or "z_image" in base) and "turbo" in base:
            eff_steps = 8       # Z-Image turbo default per sd.cpp docs
        else:
            eff_steps = None
    elif "clip_l" in files:
        argv += ["--t5xxl", files["t5xxl"], "--clip_l", files["clip_l"],
                 "--vae", files["vae"], "--cfg-scale", "1.0"]
        eff_steps = steps
    else:
        # Chroma: an uncensored FLUX finetune that drops CLIP (conditions on T5
        # only, so NO --clip_l) and is NOT guidance-distilled — it needs a real
        # cfg (~4) and more sampling steps than distilled FLUX. Disable the DiT
        # attention mask per the sd.cpp Chroma recipe.
        argv += ["--t5xxl", files["t5xxl"], "--vae", files["vae"],
                 "--cfg-scale", "4.0", "--model-args", "chroma_use_dit_mask=0"]
        eff_steps = steps or 26
    if eff_steps:
        argv += ["--steps", str(int(eff_steps))]
    if files.get("taesd"):
        # Tiny AutoEncoder: near-instant decode at slightly lower detail.
        argv += ["--taesd", files["taesd"]]
    # 1024x1024 VAE decode wants a ~8.5GB compute buffer; tiling keeps the
    # peak inside 6GB VRAM / small-RAM machines on every device.
    argv += ["--vae-tiling", "--listen-ip", host, "--listen-port", str(port)]
    if device == "gpu":
        # sd.cpp's documented low-VRAM recipe: weights live in RAM and stream
        # into VRAM per use, so models larger than the card still run at GPU
        # speed (klein Q8 on a 6GB RTX 4050: 1024px in ~80s, verified live).
        # NOT --auto-fit (its VAE-OOM fallback failed live) and NOT a hard
        # --backend pin (spills to host-visible memory at ~10x slowdown when
        # weights exceed VRAM).
        argv += ["--offload-to-cpu", "--diffusion-fa"]
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


def list_image_arch_ggufs(models_dir: str) -> list:
    """Diffusion-architecture `.gguf` files in `models_dir` (by GGUF header).
    Used to surface image models the user downloaded into the shared LLM
    models dir; anything unparseable or LLM-architected is skipped."""
    return [m for m in list_gguf_image_models(models_dir)
            if is_image_architecture(read_gguf_architecture(m["path"]))]


_FLUX2_NAME = re.compile(r"flux[-_. ]?2", re.IGNORECASE)


def looks_like_flux2(filename: str) -> bool:
    """Best-effort FLUX.2 detection by filename. FLUX.2 GGUFs carry the same
    `flux` architecture tag as FLUX.1 but need a Mistral `--llm` text encoder
    instead of t5xxl/clip_l — serving one with FLUX.1 encoders fails after a
    long load, so we reject early on the naming convention."""
    return bool(_FLUX2_NAME.search(os.path.basename(filename or "")))


def looks_like_chroma(filename: str) -> bool:
    """Best-effort Chroma detection by filename. Chroma GGUFs carry the `flux`
    (or `chroma`) architecture tag but take t5xxl + vae only (no clip_l) and a
    non-distilled cfg; a filename check routes them to the Chroma encoder set."""
    return "chroma" in os.path.basename(filename or "").lower()
