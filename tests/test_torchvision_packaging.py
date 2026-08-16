"""torchvision must be a real, declared dependency (not just an implicit
transitive of ultralytics) and must be collected by PyInstaller in
Assist.spec, mirroring onnxruntime/ultralytics/insightface. See
docs/superpowers/specs/2026-08-15-shape-detection-design.md."""
from pathlib import Path

_SPEC_FILE = Path(__file__).resolve().parent.parent / "Assist.spec"
_REQUIREMENTS_FILE = Path(__file__).resolve().parent.parent / "requirements.txt"


def test_torchvision_importable():
    import torchvision  # noqa: F401


def test_maskrcnn_importable():
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    assert callable(maskrcnn_resnet50_fpn)
    assert MaskRCNN_ResNet50_FPN_Weights.DEFAULT is not None


def test_requirements_declares_torchvision():
    text = _REQUIREMENTS_FILE.read_text(encoding="utf-8")
    assert "torchvision" in text


def test_assist_spec_collects_torchvision():
    text = _SPEC_FILE.read_text(encoding="utf-8")
    assert '"torchvision"' in text
