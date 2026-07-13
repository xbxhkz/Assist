"""Minimal GGUF header sniffing.

Reads `general.architecture` from a .gguf so model lists can route files to
the right runtime: llama-server (LLMs) vs sd-server (diffusion models). Never
raises — returns None for anything unreadable or non-GGUF.
"""
import struct

# ggml/llama.cpp GGUF value types → fixed byte sizes (string/array handled apart)
_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_STRING, _ARRAY = 8, 9

_MAX_KEYS = 256          # general.* keys are written first in practice
_MAX_ARRAY_ITEMS = 4096  # bail instead of walking tokenizer vocab arrays

# Architectures sd-server (stable-diffusion.cpp) serves. flux covers FLUX.1
# and FLUX.2 GGUFs that keep the flux tag; flux2 is listed defensively.
# lumina2 = Z-Image (turbo).
IMAGE_ARCHITECTURES = frozenset({
    "flux", "flux2", "sd1", "sd2", "sdxl", "sd3", "wan", "wan2",
    "ltxv", "qwen_image", "chroma", "lumina2",
})

# Standalone encoder/VAE files: servable by neither llama-server nor the
# image card's diffusion picker.
_ENCODER_ARCHITECTURES = frozenset({"t5encoder", "t5", "clip", "vae"})


def read_gguf_architecture(path: str):
    """Return general.architecture from a GGUF file, or None."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            head = f.read(4 + 8 + 8)
            if len(head) != 20:
                return None
            version, _tensors, kv_count = struct.unpack("<IQQ", head)
            if version not in (2, 3):
                return None
            for _ in range(min(kv_count, _MAX_KEYS)):
                raw = f.read(8)
                if len(raw) != 8:
                    return None
                key = f.read(struct.unpack("<Q", raw)[0])
                raw = f.read(4)
                if len(raw) != 4:
                    return None
                vtype = struct.unpack("<I", raw)[0]
                if vtype == _STRING:
                    raw = f.read(8)
                    if len(raw) != 8:
                        return None
                    val = f.read(struct.unpack("<Q", raw)[0])
                    if key == b"general.architecture":
                        return val.decode("utf-8", "replace") or None
                elif vtype == _ARRAY:
                    raw = f.read(4 + 8)
                    if len(raw) != 12:
                        return None
                    etype, count = struct.unpack("<IQ", raw)
                    if etype in _SCALAR_SIZES:
                        f.seek(count * _SCALAR_SIZES[etype], 1)
                    elif etype == _STRING and count <= _MAX_ARRAY_ITEMS:
                        for _ in range(count):
                            raw = f.read(8)
                            if len(raw) != 8:
                                return None
                            f.seek(struct.unpack("<Q", raw)[0], 1)
                    else:
                        return None  # huge/nested array before arch key: bail
                elif vtype in _SCALAR_SIZES:
                    f.seek(_SCALAR_SIZES[vtype], 1)
                else:
                    return None
    except OSError:
        return None
    return None


def _skip_value(f, vtype) -> bool:
    """Advance `f` past one GGUF metadata value of `vtype`. False if unreadable."""
    if vtype == _STRING:
        raw = f.read(8)
        if len(raw) != 8:
            return False
        f.seek(struct.unpack("<Q", raw)[0], 1)
        return True
    if vtype == _ARRAY:
        raw = f.read(12)
        if len(raw) != 12:
            return False
        etype, count = struct.unpack("<IQ", raw)
        if etype in _SCALAR_SIZES:
            f.seek(count * _SCALAR_SIZES[etype], 1)
            return True
        if etype == _STRING:
            for _ in range(count):
                raw = f.read(8)
                if len(raw) != 8:
                    return False
                f.seek(struct.unpack("<Q", raw)[0], 1)
            return True
        return False
    if vtype in _SCALAR_SIZES:
        f.seek(_SCALAR_SIZES[vtype], 1)
        return True
    return False


def read_gguf_tensor_names(path: str, limit: int = 6000):
    """Return the tensor NAMES in a GGUF (or [] if unreadable). Skips the
    metadata KV block, then reads the tensor directory (names only — never the
    tensor data). Bounded by `limit`."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return []
            head = f.read(20)
            if len(head) != 20:
                return []
            version, ntensors, kv_count = struct.unpack("<IQQ", head)
            if version not in (2, 3):
                return []
            for _ in range(kv_count):
                raw = f.read(8)
                if len(raw) != 8:
                    return []
                f.seek(struct.unpack("<Q", raw)[0], 1)   # key bytes
                raw = f.read(4)
                if len(raw) != 4 or not _skip_value(f, struct.unpack("<I", raw)[0]):
                    return []
            names = []
            for _ in range(min(ntensors, limit)):
                raw = f.read(8)
                if len(raw) != 8:
                    break
                nlen = struct.unpack("<Q", raw)[0]
                if nlen > 1024:
                    break
                name = f.read(nlen)
                if len(name) != nlen:
                    break
                raw = f.read(4)
                if len(raw) != 4:
                    break
                ndim = struct.unpack("<I", raw)[0]
                if ndim > 8:
                    break
                f.seek(ndim * 8 + 4 + 8, 1)   # dims + type + offset
                names.append(name.decode("utf-8", "replace"))
            return names
    except OSError:
        return []


# Tensor-name markers that appear only in an all-in-one SD/SDXL *checkpoint*
# (embedded text encoder), never in a bare diffusion-only GGUF (FLUX/klein/chroma).
_CHECKPOINT_TE_MARKERS = ("conditioner.embedders", "cond_stage_model", "text_encoders.")


def gguf_is_full_checkpoint(path: str) -> bool:
    """True if the GGUF is an all-in-one SD/SDXL checkpoint (embedded text
    encoder + VAE) — sd-server serves these via `-m`, no external encoders."""
    for nm in read_gguf_tensor_names(path):
        low = nm.lower()
        if any(m in low for m in _CHECKPOINT_TE_MARKERS):
            return True
    return False


def is_image_architecture(arch) -> bool:
    """True when `arch` is a diffusion model sd-server can serve."""
    return bool(arch) and arch in IMAGE_ARCHITECTURES


def is_llm_servable(arch) -> bool:
    """True unless `arch` is a known diffusion/encoder architecture.
    Unknown (None) stays servable — llama-server is the status-quo default."""
    if not arch:
        return True
    return arch not in IMAGE_ARCHITECTURES and arch not in _ENCODER_ARCHITECTURES
