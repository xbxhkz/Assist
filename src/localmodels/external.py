"""Registry of "linked" local .gguf models — files the user browsed to and
added by reference, living anywhere on disk (not copied into MODELS_DIR).

Persisted as a JSON list of absolute realpaths at
``<DATA_DIR>/external_models.json``. Adding/removing never touches the file
itself; a registered path whose file has since disappeared is pruned on list.
All file I/O is best-effort so a corrupt/missing registry degrades to empty.
"""
import json
import os

from src.constants import MODELS_DIR


def _registry_file() -> str:
    return os.path.join(os.path.dirname(MODELS_DIR), "external_models.json")


def _load() -> list:
    try:
        with open(_registry_file(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return [p for p in data if isinstance(p, str)] if isinstance(data, list) else []
    except Exception:
        return []


def _save(paths: list) -> None:
    try:
        f = _registry_file()
        os.makedirs(os.path.dirname(f), exist_ok=True)
        with open(f, "w", encoding="utf-8") as fh:
            json.dump(paths, fh)
    except Exception:
        pass


def _entry(path: str, external: bool) -> dict:
    return {"name": os.path.basename(path), "path": path,
            "size": os.path.getsize(path), "external": external}


def _is_inside_models_dir(real: str) -> bool:
    try:
        return os.path.commonpath([real, os.path.realpath(MODELS_DIR)]) == \
            os.path.realpath(MODELS_DIR)
    except ValueError:  # different drive on Windows
        return False


def add_external_model(path: str) -> dict:
    """Register `path` as a linked model. Raises ValueError if it isn't an
    existing .gguf. A path already inside MODELS_DIR is returned as a normal
    (non-external) entry and NOT registered (it's already listed)."""
    real = os.path.realpath(path or "")
    if not real.lower().endswith(".gguf"):
        raise ValueError("must be a .gguf file")
    if not os.path.isfile(real):
        raise ValueError("file not found")
    if _is_inside_models_dir(real):
        return _entry(real, external=False)
    paths = _load()
    if real not in paths:
        paths.append(real)
        _save(paths)
    return _entry(real, external=True)


def remove_external_model(path: str) -> None:
    """Unlink `path` from the registry. The file on disk is left untouched."""
    real = os.path.realpath(path or "")
    paths = _load()
    if real in paths:
        paths.remove(real)
        _save(paths)


def list_external_models() -> list:
    """Registered linked models whose files still exist (missing ones pruned),
    sorted by name."""
    paths = _load()
    kept, out = [], []
    for p in paths:
        if os.path.isfile(p):
            kept.append(p)
            out.append(_entry(p, external=True))
    if kept != paths:
        _save(kept)
    out.sort(key=lambda e: e["name"].lower())
    return out


def is_registered_external(path: str) -> bool:
    return os.path.realpath(path or "") in _load()
