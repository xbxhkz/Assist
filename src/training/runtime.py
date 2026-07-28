"""Locate the vendored `uv` binary and the training sidecar script at runtime.

Mirrors src/localmodels/runtime.py's resolve_llama_binary frozen/dev resolution:
frozen builds look under sys._MEIPASS, dev under the repo's build_assets / root.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _frozen_base(explicit):
    if explicit is not None:
        return explicit
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", None)
    return None


def resolve_uv_binary(frozen_base=None, dev_base=None) -> str:
    """Path to the vendored uv.exe. Frozen: <_MEIPASS>/uv/uv.exe; dev:
    <repo>/build_assets/uv/uv.exe. Raises RuntimeError if not found."""
    name = "uv.exe" if os.name == "nt" else "uv"
    base = _frozen_base(frozen_base)
    if base:
        cand = os.path.join(base, "uv", name)
        if os.path.isfile(cand):
            return cand
    if dev_base is None:
        dev_base = os.path.join(_REPO_ROOT, "build_assets", "uv")
    cand = os.path.join(dev_base, name)
    if os.path.isfile(cand):
        return cand
    raise RuntimeError(
        "uv binary not found: run scripts/fetch_uv.py to vendor it into "
        "build_assets/uv/, or ensure it's bundled in the frozen build.")


def resolve_sidecar_script(frozen_base=None, dev_base=None) -> str:
    """Path to training_sidecar/train.py. Frozen: <_MEIPASS>/training_sidecar/
    train.py; dev: <repo>/training_sidecar/train.py. Raises RuntimeError if missing."""
    base = _frozen_base(frozen_base)
    if base:
        cand = os.path.join(base, "training_sidecar", "train.py")
        if os.path.isfile(cand):
            return cand
    if dev_base is None:
        dev_base = os.path.join(_REPO_ROOT, "training_sidecar")
    cand = os.path.join(dev_base, "train.py")
    if os.path.isfile(cand):
        return cand
    raise RuntimeError("training_sidecar/train.py not found.")


def resolve_convert_script(frozen_base=None, dev_base=None) -> str:
    """Path to training_sidecar/convert.py (frozen <_MEIPASS>/training_sidecar/
    convert.py; dev <repo>/training_sidecar/convert.py). Raises RuntimeError if missing."""
    base = _frozen_base(frozen_base)
    if base:
        cand = os.path.join(base, "training_sidecar", "convert.py")
        if os.path.isfile(cand):
            return cand
    if dev_base is None:
        dev_base = os.path.join(_REPO_ROOT, "training_sidecar")
    cand = os.path.join(dev_base, "convert.py")
    if os.path.isfile(cand):
        return cand
    raise RuntimeError("training_sidecar/convert.py not found.")


def resolve_merge_script(frozen_base=None, dev_base=None) -> str:
    """Path to training_sidecar/merge.py (frozen <_MEIPASS>/training_sidecar/merge.py;
    dev <repo>/training_sidecar/merge.py). Raises RuntimeError if missing."""
    base = _frozen_base(frozen_base)
    if base:
        cand = os.path.join(base, "training_sidecar", "merge.py")
        if os.path.isfile(cand):
            return cand
    if dev_base is None:
        dev_base = os.path.join(_REPO_ROOT, "training_sidecar")
    cand = os.path.join(dev_base, "merge.py")
    if os.path.isfile(cand):
        return cand
    raise RuntimeError("training_sidecar/merge.py not found.")
