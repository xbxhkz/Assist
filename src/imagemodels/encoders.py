"""Resolve the shared FLUX aux files.

sd-server needs the diffusion GGUF (user-supplied) plus aux files that are the
SAME across models of a family: FLUX.1 → t5xxl + clip_l + ae; FLUX.2 (klein) →
a Qwen3/Mistral `--llm` text encoder + the flux2 VAE. For each, look at: an
explicit path, then next to the diffusion GGUF, then the shared
`<IMAGE_MODELS_DIR>/encoders` dir (where the app can download them once).
Missing files raise a clear error.
"""
import fnmatch
import os

from src.constants import IMAGE_MODELS_DIR

# Candidate filenames for each aux role, in preference order. Entries may be
# fnmatch patterns (matched case-insensitively) — encoder ggufs ship under
# many quant-suffixed names.
ENCODER_FILENAMES = {
    "t5xxl": ["t5xxl.gguf", "t5xxl_q8_0.gguf", "t5xxl_fp16.safetensors",
              "t5xxl_fp16.gguf", "t5-v1_1-xxl-encoder-Q8_0.gguf"],
    "clip_l": ["clip_l.safetensors", "clip_l.gguf"],
    "vae": ["ae.safetensors", "ae.sft", "vae.safetensors", "flux_vae.safetensors"],
}

# FLUX.2 roles. The VAE names are deliberately flux2-specific: FLUX.1's
# ae.safetensors shares the encoders dir and MUST NOT match here.
FLUX2_ENCODER_FILENAMES = {
    "llm": ["qwen3-4b*.gguf", "qwen_3_4b*", "qwen3-8b*.gguf",
            "mistral-small*.gguf", "flux2_text_encoder*"],
    "vae": ["flux2_ae.safetensors", "flux2-ae.safetensors",
            "flux2_vae.safetensors"],
}


class MissingEncoderError(Exception):
    """Raised when required FLUX aux files can't be found. `.missing` lists the
    unresolved roles (e.g. ["t5xxl", "vae"])."""
    def __init__(self, missing):
        self.missing = list(missing)
        super().__init__("missing FLUX files: " + ", ".join(self.missing))


def encoders_dir() -> str:
    return os.path.join(IMAGE_MODELS_DIR, "encoders")


def _find(role: str, explicit, search_dirs, names=ENCODER_FILENAMES):
    if explicit and os.path.isfile(explicit):
        return os.path.realpath(explicit)
    for d in search_dirs:
        try:
            entries = sorted(os.listdir(d)) if os.path.isdir(d) else []
        except OSError:
            entries = []
        lowered = {e.lower(): e for e in entries}
        for pat in names[role]:
            pat = pat.lower()
            if any(ch in pat for ch in "*?["):
                for low, orig in lowered.items():
                    if fnmatch.fnmatchcase(low, pat):
                        cand = os.path.join(d, orig)
                        if os.path.isfile(cand):
                            return os.path.realpath(cand)
            elif pat in lowered:
                cand = os.path.join(d, lowered[pat])
                if os.path.isfile(cand):
                    return os.path.realpath(cand)
    return None


def _resolve(diffusion_model, roles, names) -> dict:
    diff = os.path.realpath(diffusion_model or "")
    search = [os.path.dirname(diff), encoders_dir()]
    resolved = {"diffusion_model": diff}
    missing = []
    for role, explicit in roles:
        hit = _find(role, explicit, search, names=names)
        if hit:
            resolved[role] = hit
        else:
            missing.append(role)
    if missing:
        raise MissingEncoderError(missing)
    return resolved


def resolve_flux_files(diffusion_model, t5xxl=None, clip_l=None, vae=None) -> dict:
    """FLUX.1: return {diffusion_model, t5xxl, clip_l, vae} of realpaths, or
    raise MissingEncoderError naming what couldn't be resolved."""
    return _resolve(diffusion_model,
                    (("t5xxl", t5xxl), ("clip_l", clip_l), ("vae", vae)),
                    ENCODER_FILENAMES)


def resolve_flux2_files(diffusion_model, llm=None, vae=None) -> dict:
    """FLUX.2 (klein): return {diffusion_model, llm, vae} of realpaths, or
    raise MissingEncoderError. `llm` is the Qwen3 (klein) / Mistral (dev)
    text encoder passed to sd-server's --llm."""
    return _resolve(diffusion_model, (("llm", llm), ("vae", vae)),
                    FLUX2_ENCODER_FILENAMES)
