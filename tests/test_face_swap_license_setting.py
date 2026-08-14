"""face_swap_license_accepted is a plain, persistent DEFAULT_SETTINGS key --
unlike screen/camera/input/shell access, it must NOT be wired into any
startup reset function, since a one-time license acknowledgment should
persist across restarts. See
docs/superpowers/specs/2026-08-13-image-editing-face-swap-design.md."""
import inspect

from src.settings import DEFAULT_SETTINGS


def test_face_swap_license_accepted_defaults_false():
    assert DEFAULT_SETTINGS["face_swap_license_accepted"] is False


def test_face_swap_license_accepted_not_reset_at_startup():
    import src.settings as settings_module
    reset_fns = [
        obj for name, obj in vars(settings_module).items()
        if name.startswith("reset_") and inspect.isfunction(obj)
    ]
    assert reset_fns, "expected at least one reset_* function to exist as a baseline"
    for fn in reset_fns:
        src = inspect.getsource(fn)
        assert "face_swap_license_accepted" not in src, (
            f"{fn.__name__} must not touch face_swap_license_accepted -- "
            "it's a persistent one-time acceptance, not a per-session capability"
        )
