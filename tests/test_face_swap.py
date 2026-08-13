"""face_swap.swap_face runs InsightFace's detection + InSwapper via
injectable analyzer/swapper objects, mirroring src/bg_removal.py's
session=None pattern -- tests never need the real (gated, non-bundled)
model weights or a real license acceptance. See
docs/superpowers/specs/2026-08-13-image-editing-face-swap-design.md.
"""
import io

import numpy as np
import pytest
from PIL import Image

from src import face_swap


def _make_png(size=(64, 64), color=(200, 50, 50)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeFace:
    pass


class _FakeAnalyzer:
    def __init__(self, faces_by_call):
        self._faces_by_call = list(faces_by_call)

    def get(self, img):
        return self._faces_by_call.pop(0)


class _FakeSwapper:
    def get(self, target_img, target_face, source_face, paste_back=True):
        # Return a same-shape array so PIL can re-encode it as a valid PNG.
        return target_img


def test_swap_face_returns_png_bytes_with_provenance_metadata(monkeypatch):
    monkeypatch.setattr(face_swap, "get_setting", lambda key, default=None: True)
    analyzer = _FakeAnalyzer([[_FakeFace()], [_FakeFace()]])
    swapper = _FakeSwapper()

    result = face_swap.swap_face(
        _make_png(color=(10, 20, 30)), _make_png(color=(200, 100, 50)),
        analyzer=analyzer, swapper=swapper,
    )

    out = Image.open(io.BytesIO(result))
    assert out.format == "PNG"
    assert out.text.get("assist:ai-edited") == "face-swap"


def test_swap_face_raises_when_license_not_accepted(monkeypatch):
    monkeypatch.setattr(face_swap, "get_setting", lambda key, default=None: False)

    with pytest.raises(face_swap.LicenseNotAcceptedError):
        face_swap.swap_face(
            _make_png(), _make_png(),
            analyzer=_FakeAnalyzer([[_FakeFace()], [_FakeFace()]]),
            swapper=_FakeSwapper(),
        )


def test_swap_face_raises_when_no_face_in_source(monkeypatch):
    monkeypatch.setattr(face_swap, "get_setting", lambda key, default=None: True)
    analyzer = _FakeAnalyzer([[]])  # source: no faces found

    with pytest.raises(face_swap.NoFaceDetectedError):
        face_swap.swap_face(
            _make_png(), _make_png(),
            analyzer=analyzer, swapper=_FakeSwapper(),
        )


def test_swap_face_raises_when_no_face_in_target(monkeypatch):
    monkeypatch.setattr(face_swap, "get_setting", lambda key, default=None: True)
    analyzer = _FakeAnalyzer([[_FakeFace()], []])  # source ok, target: none

    with pytest.raises(face_swap.NoFaceDetectedError):
        face_swap.swap_face(
            _make_png(), _make_png(),
            analyzer=analyzer, swapper=_FakeSwapper(),
        )


def test_swap_face_does_not_construct_real_analyzer_when_injected(monkeypatch):
    """Injection must genuinely bypass the real (license-gated, model-file-
    requiring) singleton construction, not just the swap call."""
    monkeypatch.setattr(face_swap, "get_setting", lambda key, default=None: True)

    def _fail_if_called():
        raise AssertionError("should not construct a real FaceAnalysis")

    monkeypatch.setattr(face_swap, "_get_analyzer", _fail_if_called)
    monkeypatch.setattr(face_swap, "_get_swapper", _fail_if_called)

    face_swap.swap_face(
        _make_png(), _make_png(),
        analyzer=_FakeAnalyzer([[_FakeFace()], [_FakeFace()]]),
        swapper=_FakeSwapper(),
    )
