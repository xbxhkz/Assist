"""bg_removal.remove_background runs U2Net inference via an injectable
onnxruntime session (mirroring src/vision/yolo.py's model=None pattern) so
tests never need the real u2net.onnx file, which only exists after
scripts/fetch_bg_removal_model.py runs at build time. See
docs/superpowers/specs/2026-08-12-image-editing-background-removal-design.md.
"""
import io

import numpy as np
import pytest
from PIL import Image

from src import bg_removal


def _make_jpeg(size=(64, 64), color=(200, 50, 50)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class _FakeInput:
    name = "input"


class _FakeSession:
    def get_inputs(self):
        return [_FakeInput()]

    def run(self, output_names, feed):
        arr = np.ones((1, 1, bg_removal._MODEL_INPUT_SIZE, bg_removal._MODEL_INPUT_SIZE), dtype=np.float32)
        return [arr]


def test_remove_background_returns_rgba_png_with_injected_session():
    result = bg_removal.remove_background(_make_jpeg(), session=_FakeSession())

    out = Image.open(io.BytesIO(result))
    assert out.format == "PNG"
    assert out.mode == "RGBA"
    assert out.size == (64, 64)


def test_remove_background_preserves_original_dimensions():
    result = bg_removal.remove_background(_make_jpeg(size=(100, 50)), session=_FakeSession())

    out = Image.open(io.BytesIO(result))
    assert out.size == (100, 50)


def test_missing_model_file_raises_a_clear_actionable_error(tmp_path, monkeypatch):
    """A dev/build environment where scripts/fetch_bg_removal_model.py never
    ran must fail with a specific message naming the expected path and the fix
    -- never a raw onnxruntime exception (the spec requires this explicitly)."""
    missing = str(tmp_path / "nowhere" / "u2net.onnx")
    monkeypatch.setattr(bg_removal, "_session", None)
    monkeypatch.setattr(bg_removal, "_model_path", lambda: missing)

    with pytest.raises(RuntimeError) as excinfo:
        bg_removal._get_session()

    msg = str(excinfo.value)
    assert missing in msg
    assert "fetch_bg_removal_model.py" in msg


def test_remove_background_without_a_session_surfaces_the_clear_error(tmp_path, monkeypatch):
    """Same check through the real public entry point: remove_background() with
    no injected session must surface the actionable message, not whatever
    onnxruntime would have thrown."""
    monkeypatch.setattr(bg_removal, "_session", None)
    monkeypatch.setattr(bg_removal, "_model_path", lambda: str(tmp_path / "u2net.onnx"))

    with pytest.raises(RuntimeError, match="fetch_bg_removal_model.py"):
        bg_removal.remove_background(_make_jpeg())


def test_remove_background_uses_injected_session_not_the_real_one(monkeypatch):
    calls = []

    class _TrackedSession(_FakeSession):
        def run(self, output_names, feed):
            calls.append("ran")
            return super().run(output_names, feed)

    def _fail_if_called():
        raise AssertionError("should not construct a real onnxruntime session")

    monkeypatch.setattr(bg_removal, "_get_session", _fail_if_called)

    bg_removal.remove_background(_make_jpeg(), session=_TrackedSession())

    assert calls == ["ran"]
