"""Background removal via a bundled U2Net ONNX model.

No rembg/transformers dependency: rembg is Windows-unsupported in this app's
own Cookbook UI (static/js/cookbook.js's _winUnsupported set), and the
frozen build has no pip available at runtime to install either. Mirrors
src/vision/yolo.py's lazy-singleton + injectable-session pattern so tests
never need the real model file, which only exists after
scripts/fetch_bg_removal_model.py runs at build time. See
docs/superpowers/specs/2026-08-12-image-editing-background-removal-design.md.
"""
import io
import os
import sys

import numpy as np
from PIL import Image

_MODEL_INPUT_SIZE = 320  # U2Net's standard input resolution
_session = None


def _model_path() -> str:
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        return os.path.join(base, "bg_removal", "u2net.onnx")
    return os.path.join(os.path.dirname(__file__), "..", "build_assets", "bg_removal", "u2net.onnx")


def _get_session():
    global _session
    if _session is None:
        path = _model_path()
        # Checked BEFORE importing onnxruntime / constructing the session so a
        # dev environment where the build-time fetch never ran gets a specific,
        # actionable error instead of a raw onnxruntime "No such file" (or an
        # ImportError that hides the real problem). Mirrors
        # src/localmodels/runtime.py's "run scripts/fetch_llama_server.py"
        # message for the same class of missing build asset.
        if not os.path.isfile(path):
            raise RuntimeError(
                f"Background-removal model not found at {path}. "
                "Run scripts/fetch_bg_removal_model.py to download it into "
                "build_assets/bg_removal/ (the frozen build bundles it from there)."
            )
        import onnxruntime
        _session = onnxruntime.InferenceSession(path)
    return _session


def _preprocess(img: Image.Image) -> np.ndarray:
    resized = img.convert("RGB").resize((_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE), Image.LANCZOS)
    arr = np.array(resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(arr, axis=0).astype(np.float32)


def remove_background(image_bytes: bytes, *, session=None) -> bytes:
    """Return RGBA PNG bytes with the background made transparent."""
    img = Image.open(io.BytesIO(image_bytes))
    original_size = img.size

    sess = session or _get_session()
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: _preprocess(img)})[0]

    mask = output[0][0]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(original_size, Image.LANCZOS)

    rgba = img.convert("RGBA")
    rgba.putalpha(mask_img)

    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return buf.getvalue()
