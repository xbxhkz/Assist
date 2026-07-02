"""Helpers for resolving runtime paths in source and frozen builds."""

import os
import sys


def get_app_root() -> str:
    """Return the app root directory.

    In normal source runs, this is the repository root. In a frozen Windows
    build, it is the bundle content root (PyInstaller's internal directory)
    so bundled runtime folders like `static/`, `scripts/`, and `data/` stay
    together with the executable payload.
    """
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _frozen_data_dir(home: str) -> str:
    """Resolve the frozen data dir under `home`, migrating a legacy
    `.odysseus/data` to `.assist/data` once. Falls back to the legacy path
    if a migration attempt fails but legacy data exists (never lose data)."""
    new = os.path.join(home, ".assist", "data")
    legacy = os.path.join(home, ".odysseus", "data")
    try:
        if not os.path.exists(new) and os.path.isdir(legacy):
            os.makedirs(os.path.dirname(new), exist_ok=True)
            os.rename(legacy, new)
    except Exception:
        if os.path.isdir(legacy):
            return legacy
    return new


def get_default_data_dir() -> str:
    """Return the default path to the data directory.

    In normal runs, this is a 'data' subdirectory under the app root. In frozen
    builds, it is a persistent user directory (~/.assist/data), migrated once
    from the legacy ~/.odysseus/data if present.
    """
    if getattr(sys, "frozen", False):
        return _frozen_data_dir(os.path.expanduser("~"))
    return os.path.join(get_app_root(), "data")