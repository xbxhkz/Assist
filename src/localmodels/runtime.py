"""Pure helpers for the native local-model runtime (Phase 3a).

No process launching or DB access here — just binary resolution, command
construction, and filesystem listing, so this module is fully unit-testable.
"""
import os
import shutil
import sys


def local_endpoint_url(port: int) -> str:
    """OpenAI-compatible base URL for a locally served model (loopback only)."""
    return f"http://127.0.0.1:{port}/v1"


def find_mmproj(model_path: str, listdir=os.listdir) -> str:
    """Return a sibling multimodal-projector GGUF for `model_path`, or None.

    Vision GGUFs ship a separate `mmproj-*.gguf` that llama-server needs via
    --mmproj to load the vision encoder; without it a VL model serves
    text-only (blind). Convention (ggml-org / unsloth): the projector sits
    beside the model, named `mmproj*...gguf`. When several are present, prefer
    the one whose name shares the most family tokens with the model (so
    `Qwen2.5-VL-...` matches its own projector, not a stray llava one).
    `listdir` is injectable for tests.
    """
    d = os.path.dirname(model_path)
    try:
        cands = [f for f in listdir(d)
                 if f.lower().startswith("mmproj") and f.lower().endswith(".gguf")]
    except OSError:
        return None
    if not cands:
        return None
    # Score by shared 3+ char tokens from the model's name (minus the quant
    # suffix), so the projector for THIS model family wins over an unrelated one.
    stem = os.path.basename(model_path).lower()
    for _q in ("-q4", "-q5", "-q6", "-q8", "-q3", "-q2", "-iq", "-f16", "-bf16", "-ud"):
        cut = stem.find(_q)
        if cut > 0:
            stem = stem[:cut]
            break
    toks = [t for t in stem.replace("_", "-").replace(".", "-").split("-") if len(t) >= 3]

    def _score(fn):
        fl = fn.lower()
        return sum(1 for t in toks if t in fl)

    cands.sort(key=_score, reverse=True)
    # Only attach a projector that actually matches this model's name family.
    # A non-vision model (Ornith, Qwen3.6) sharing a folder with a VL model's
    # mmproj must get NO projector — passing a mismatched one makes
    # llama-server refuse to load ("you may have the wrong mmproj").
    if _score(cands[0]) == 0:
        return None
    return os.path.join(d, cands[0])


def build_serve_argv(binary: str, model_path: str, port: int,
                     ctx_size: int = 4096, host: str = "127.0.0.1",
                     device: str = "cpu", mmproj: str = None) -> list:
    """llama-server argv for an OpenAI-compatible loopback server.

    `--alias` sets the id llama-server advertises at /v1/models (and accepts
    for chat). Without it, the id is the full model *path*, which the model
    picker then shows verbatim (its display split can't shorten a Windows
    backslash path). Aliasing to the .gguf filename makes the served model
    show up under the same name the Local Models list uses.

    `mmproj` (a multimodal-projector GGUF path) makes the served model
    vision-capable — llama-server loads the vision encoder and accepts
    image_url content blocks. Pass the sibling projector from find_mmproj().
    """
    argv = [
        binary,
        "--model", model_path,
        "--alias", os.path.basename(model_path),
        "--host", host,
        "--port", str(port),
        "--ctx-size", str(ctx_size),
    ]
    if mmproj:
        argv += ["--mmproj", mmproj]
    # device=gpu adds NO flags on purpose: the Vulkan build auto-fits GPU
    # layers to free VRAM (common_fit_params), and an explicit -ngl DISABLES
    # that fitter ("n_gpu_layers already set by user, abort") — observed live
    # as a hard OOM on a 6GB card. Device selection is the binary choice in
    # resolve_llama_binary.
    return argv


def _bundled_binary_name() -> str:
    return "llama-server.exe" if os.name == "nt" else "llama-server"


def resolve_llama_binary(device: str = "cpu", path_lookup=shutil.which,
                         frozen_base: str = None, dev_base: str = None) -> str:
    """Resolve the llama-server executable.

    Preference: (1) a `llama-server`/`.exe` on PATH (user-installed, possibly
    GPU); (2) the bundled binary at `<frozen_base>/llama/<cpu|vulkan>/<name>`
    when frozen (falling back to the legacy flat `llama/<name>` layout);
    (3) the same layout under `<repo>/build_assets/llama`. Raises if none.
    `device="gpu"` selects the Vulkan build, else the CPU build.
    `path_lookup`/`frozen_base`/`dev_base` are injectable for tests.
    """
    found = path_lookup("llama-server") or path_lookup("llama-server.exe")
    if found:
        return found

    sub = "vulkan" if device == "gpu" else "cpu"
    name = _bundled_binary_name()

    base = frozen_base
    if base is None and getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None)
    if base:
        for rel in (os.path.join("llama", sub, name), os.path.join("llama", name)):
            cand = os.path.join(base, rel)
            if os.path.isfile(cand):
                return cand

    if dev_base is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dev_base = os.path.join(repo_root, "build_assets", "llama")
    for cand in (os.path.join(dev_base, sub, name), os.path.join(dev_base, name)):
        if os.path.isfile(cand):
            return cand

    raise RuntimeError(
        "llama-server not found: no server on PATH, no bundled binary, and no "
        "dev binary under build_assets/llama."
    )


def list_gguf_models(models_dir: str) -> list:
    """List llama-servable `.gguf` files in `models_dir` as [{name, path, size}],
    sorted by name. Diffusion/encoder-architecture files (FLUX, t5, ...) share
    this download dir but belong to sd-server — llama-server dies on them at
    load, so they're excluded here (the Image Models card lists them instead)."""
    from src.gguf_meta import read_gguf_architecture, is_llm_servable
    out = []
    if os.path.isdir(models_dir):
        for fn in sorted(os.listdir(models_dir)):
            if fn.lower().endswith(".gguf"):
                # mmproj-*.gguf is a vision projector, not a standalone model —
                # it's attached to its sibling VL model via --mmproj, never
                # served on its own. Hide it from the servable list.
                if fn.lower().startswith("mmproj"):
                    continue
                p = os.path.join(models_dir, fn)
                if not is_llm_servable(read_gguf_architecture(p)):
                    continue
                out.append({"name": fn, "path": p, "size": os.path.getsize(p)})
    return out
