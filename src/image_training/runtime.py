"""Locate the image-training sidecar script at runtime. Mirrors
src/training/runtime.py's resolve_sidecar_script frozen/dev resolution."""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _frozen_base(explicit):
    if explicit is not None:
        return explicit
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", None)
    return None


def resolve_image_sidecar_script(frozen_base=None, dev_base=None) -> str:
    """Path to image_training_sidecar/train_sdxl_lora.py. Frozen:
    <_MEIPASS>/image_training_sidecar/train_sdxl_lora.py; dev:
    <repo>/image_training_sidecar/train_sdxl_lora.py. Raises RuntimeError
    if missing."""
    base = _frozen_base(frozen_base)
    if base:
        cand = os.path.join(base, "image_training_sidecar", "train_sdxl_lora.py")
        if os.path.isfile(cand):
            return cand
    if dev_base is None:
        dev_base = os.path.join(_REPO_ROOT, "image_training_sidecar")
    cand = os.path.join(dev_base, "train_sdxl_lora.py")
    if os.path.isfile(cand):
        return cand
    raise RuntimeError("image_training_sidecar/train_sdxl_lora.py not found.")
