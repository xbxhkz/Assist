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


def build_serve_argv(binary: str, model_path: str, port: int,
                     ctx_size: int = 4096, host: str = "127.0.0.1") -> list:
    """llama-server argv for an OpenAI-compatible loopback server.

    `--alias` sets the id llama-server advertises at /v1/models (and accepts
    for chat). Without it, the id is the full model *path*, which the model
    picker then shows verbatim (its display split can't shorten a Windows
    backslash path). Aliasing to the .gguf filename makes the served model
    show up under the same name the Local Models list uses.
    """
    return [
        binary,
        "--model", model_path,
        "--alias", os.path.basename(model_path),
        "--host", host,
        "--port", str(port),
        "--ctx-size", str(ctx_size),
    ]


def _bundled_binary_name() -> str:
    return "llama-server.exe" if os.name == "nt" else "llama-server"


def resolve_llama_binary(path_lookup=shutil.which, frozen_base: str = None,
                         dev_base: str = None) -> str:
    """Resolve the llama-server executable.

    Preference: (1) a `llama-server`/`.exe` on PATH (user-installed, possibly
    GPU); (2) the bundled CPU binary at `<frozen_base>/llama/<name>` when frozen;
    (3) a dev fallback at `<repo>/build_assets/llama/<name>`. Raises if none.
    `path_lookup`/`frozen_base`/`dev_base` are injectable for tests.
    """
    found = path_lookup("llama-server") or path_lookup("llama-server.exe")
    if found:
        return found

    name = _bundled_binary_name()

    base = frozen_base
    if base is None and getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None)
    if base:
        cand = os.path.join(base, "llama", name)
        if os.path.isfile(cand):
            return cand

    if dev_base is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dev_base = os.path.join(repo_root, "build_assets", "llama")
    cand = os.path.join(dev_base, name)
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
                p = os.path.join(models_dir, fn)
                if not is_llm_servable(read_gguf_architecture(p)):
                    continue
                out.append({"name": fn, "path": p, "size": os.path.getsize(p)})
    return out
