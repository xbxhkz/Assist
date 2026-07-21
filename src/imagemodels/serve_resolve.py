"""Resolve the sd-server `files` dict for a local image GGUF: detect the model
family and gather its encoders/VAE. Extracted from the imagemodels serve route so
BOTH the route and the on-demand auto-serve path share one implementation."""
import os

from src.gguf_meta import gguf_is_full_checkpoint, read_gguf_architecture
from src.imagemodels.runtime import looks_like_flux2, looks_like_chroma
from src.imagemodels.encoders import (
    resolve_flux_files, resolve_flux2_files, resolve_zimage_files,
    resolve_chroma_files, MissingEncoderError,
)

_MISSING_HINTS = {
    "Z-Image": ("Missing Z-Image files: {missing}. Z-Image needs a Qwen3-4B text "
                "encoder (llm, e.g. Qwen3-4B-Q4_K_M.gguf) and the FLUX.1 VAE (vae, "
                "ae.safetensors) next to the model or in the shared encoders folder."),
    "FLUX.2": ("Missing FLUX.2 files: {missing}. klein needs a Qwen3 text encoder "
               "(llm, e.g. Qwen3-4B-Q4_K_M.gguf) and the FLUX.2 VAE (vae, "
               "flux2_ae.safetensors) next to the model or in the shared encoders folder."),
    "Chroma": ("Missing Chroma files: {missing}. Chroma needs a T5-XXL text encoder "
               "(t5xxl, e.g. t5xxl_q8_0.gguf) and the FLUX.1 VAE (vae, ae.safetensors) "
               "next to the model or in the shared encoders folder."),
    "FLUX": ("Missing FLUX files: {missing}. Put t5xxl / clip_l / vae next to the "
             "model, or download the FLUX encoders."),
}


def resolve_image_files(model_path, *, llm=None, vae=None, t5xxl=None, clip_l=None) -> dict:
    """Return the sd-server `files` dict for `model_path`. Raises MissingEncoderError
    (with a `.hint` string) when a required encoder/VAE can't be found."""
    real = os.path.realpath(model_path)
    base = os.path.basename(real).lower()

    # All-in-one SD/SDXL checkpoint (embedded encoders+VAE): self-contained.
    if gguf_is_full_checkpoint(real):
        return {"checkpoint": real}

    arch = read_gguf_architecture(real)
    if arch == "lumina2" or "z-image" in base or "z_image" in base:
        kind, call = "Z-Image", lambda: resolve_zimage_files(real, llm=llm, vae=vae)
    elif looks_like_flux2(real):
        kind, call = "FLUX.2", lambda: resolve_flux2_files(real, llm=llm, vae=vae)
    elif arch == "chroma" or looks_like_chroma(real):
        kind, call = "Chroma", lambda: resolve_chroma_files(real, t5xxl=t5xxl, vae=vae)
    else:
        kind, call = "FLUX", lambda: resolve_flux_files(real, t5xxl=t5xxl, clip_l=clip_l, vae=vae)

    try:
        return call()
    except MissingEncoderError as e:
        e.hint = _MISSING_HINTS[kind].format(missing=", ".join(e.missing))
        raise
