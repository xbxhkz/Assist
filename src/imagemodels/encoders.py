"""Resolve the shared FLUX aux files (T5-XXL text encoder, CLIP-L, VAE).

sd-server needs four files to serve FLUX: the diffusion GGUF (user-supplied) plus
these three, which are the SAME across FLUX models. For each, look at: an explicit
path, then next to the diffusion GGUF, then the shared `<IMAGE_MODELS_DIR>/encoders`
dir (where the app can download them once). Missing files raise a clear error.
"""
import os

from src.constants import IMAGE_MODELS_DIR

# Candidate filenames for each aux role, in preference order.
ENCODER_FILENAMES = {
    "t5xxl": ["t5xxl.gguf", "t5xxl_q8_0.gguf", "t5xxl_fp16.safetensors",
              "t5xxl_fp16.gguf", "t5-v1_1-xxl-encoder-Q8_0.gguf"],
    "clip_l": ["clip_l.safetensors", "clip_l.gguf"],
    "vae": ["ae.safetensors", "ae.sft", "vae.safetensors", "flux_vae.safetensors"],
}


class MissingEncoderError(Exception):
    """Raised when required FLUX aux files can't be found. `.missing` lists the
    unresolved roles (e.g. ["t5xxl", "vae"])."""
    def __init__(self, missing):
        self.missing = list(missing)
        super().__init__("missing FLUX files: " + ", ".join(self.missing))


def encoders_dir() -> str:
    return os.path.join(IMAGE_MODELS_DIR, "encoders")


def _find(role: str, explicit, search_dirs):
    if explicit and os.path.isfile(explicit):
        return os.path.realpath(explicit)
    for d in search_dirs:
        for fn in ENCODER_FILENAMES[role]:
            cand = os.path.join(d, fn)
            if os.path.isfile(cand):
                return os.path.realpath(cand)
    return None


def resolve_flux_files(diffusion_model, t5xxl=None, clip_l=None, vae=None) -> dict:
    """Return {diffusion_model, t5xxl, clip_l, vae} of realpaths, or raise
    MissingEncoderError naming what couldn't be resolved."""
    diff = os.path.realpath(diffusion_model or "")
    search = [os.path.dirname(diff), encoders_dir()]
    resolved = {"diffusion_model": diff}
    missing = []
    for role, explicit in (("t5xxl", t5xxl), ("clip_l", clip_l), ("vae", vae)):
        hit = _find(role, explicit, search)
        if hit:
            resolved[role] = hit
        else:
            missing.append(role)
    if missing:
        raise MissingEncoderError(missing)
    return resolved
