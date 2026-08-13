"""insightface must be a real, importable dependency (not the Cookbook
pip-install allowlist, which is a no-op in the frozen build) and must be
collected by PyInstaller in Assist.spec, mirroring onnxruntime/ultralytics.
See docs/superpowers/specs/2026-08-13-image-editing-face-swap-design.md."""
from pathlib import Path

_SPEC_FILE = Path(__file__).resolve().parent.parent / "Assist.spec"
_REQUIREMENTS_FILE = Path(__file__).resolve().parent.parent / "requirements.txt"


def test_insightface_importable():
    import insightface  # noqa: F401


def test_face_analysis_class_importable():
    from insightface.app import FaceAnalysis
    assert callable(FaceAnalysis)


def test_model_zoo_get_model_importable():
    from insightface.model_zoo import get_model
    assert callable(get_model)


def test_requirements_declares_insightface():
    text = _REQUIREMENTS_FILE.read_text(encoding="utf-8")
    assert "insightface" in text


def test_assist_spec_collects_insightface():
    text = _SPEC_FILE.read_text(encoding="utf-8")
    assert '"insightface"' in text
