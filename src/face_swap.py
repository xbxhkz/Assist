"""Face-swap via a locally-run InsightFace pipeline (face detection/
alignment + the pretrained InSwapper model). Both the detection model pack
(buffalo_l) and the swap model (inswapper_128.onnx) are licensed for
non-commercial research use only by InsightFace -- neither is bundled in
Assist's installer or fetched automatically. This module never lets
InsightFace's own implicit downloader run without the user's explicit,
recorded acceptance (the face_swap_license_accepted setting) -- see
_ensure_models_available(). Output PNGs carry embedded provenance metadata
identifying them as AI-face-swapped (metadata only, no visible watermark,
per the approved design). See
docs/superpowers/specs/2026-08-13-image-editing-face-swap-design.md.

API verified directly against the installed insightface==1.0.1 package
(site-packages, not web search / training-data memory) before writing this
module:
  - FaceAnalysis.__init__(self, name='buffalo_l', root='~/.insightface',
    allowed_modules=None, **kwargs) -- root= is confirmed the right kwarg.
  - model_zoo.get_model(name, **kwargs) reads root= via kwargs (confirmed:
    `root = kwargs.get('root', '~/.insightface')`), mirroring
    FaceAnalysis(root=...). _get_swapper() passes the bare filename (not an
    absolute path) as `name`, since get_model()'s internal download_onnx()
    interpolates `name` directly into the fetch URL -- an absolute local
    path there would build a malformed, non-working download URL.
  - FaceAnalysis.get(img) returns a plain Python list ([] or list[Face]) --
    confirmed via source, matches the truthiness checks below.
No deviations from the plan's assumptions were needed.
"""
import io
import os

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from src.settings import get_setting

_MODEL_PACK_NAME = "buffalo_l"
_SWAP_MODEL_FILENAME = "inswapper_128.onnx"

_analyzer = None
_swapper = None


class LicenseNotAcceptedError(Exception):
    """Raised when a swap is attempted before the InsightFace model
    license (covering both buffalo_l and inswapper_128.onnx) has been
    explicitly accepted via the face_swap_license_accepted setting."""


class NoFaceDetectedError(Exception):
    """Raised when face detection finds no usable face in an input image."""


def _model_root() -> str:
    from src.constants import DATA_DIR
    return os.path.join(DATA_DIR, "face_swap_models")


def _ensure_models_available():
    """Gate: the InsightFace license must be explicitly accepted before
    InsightFace's own downloader is allowed to run. Its primary caller is
    swap_face() itself, unconditionally, so the gate applies even when
    analyzer/swapper are injected (see swap_face()'s docstring for why). It
    is also called from _get_analyzer() and _get_swapper() so neither
    singleton can be constructed (and neither model implicitly downloaded)
    on its own without passing this check first -- do not remove either
    call site as a "redundant cleanup"."""
    if get_setting("face_swap_license_accepted", False) is not True:
        raise LicenseNotAcceptedError(
            "Face-swap requires accepting InsightFace's model license first "
            "-- go to Settings -> AI Defaults -> Face Swap Model License to "
            "review and accept it."
        )


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _ensure_models_available()
        from insightface.app import FaceAnalysis
        _analyzer = FaceAnalysis(name=_MODEL_PACK_NAME, root=_model_root())
        _analyzer.prepare(ctx_id=0, det_size=(640, 640))
    return _analyzer


def _get_swapper():
    global _swapper
    if _swapper is None:
        _ensure_models_available()
        from insightface.model_zoo import get_model
        # Pass the bare filename (not a full path) as `name`, with root=
        # telling InsightFace where to look/save it -- mirroring
        # _get_analyzer()'s FaceAnalysis(root=...) pattern. get_model()
        # interpolates `name` directly into its download URL when it needs
        # to fetch the file; passing our absolute local path there would
        # build a malformed, non-working URL instead of a real download.
        _swapper = get_model(_SWAP_MODEL_FILENAME, download=True, root=_model_root())
    return _swapper


def _provenance_metadata() -> PngInfo:
    info = PngInfo()
    info.add_text("assist:ai-edited", "face-swap")
    return info


def swap_face(source_face_bytes: bytes, target_image_bytes: bytes, *,
              analyzer=None, swapper=None) -> bytes:
    """Swap the face from source_face_bytes into target_image_bytes.
    Returns PNG bytes with embedded provenance metadata. Raises
    LicenseNotAcceptedError, NoFaceDetectedError, or lets underlying
    inference errors propagate -- callers apply the never-raises discipline
    at their own boundary, matching src/bg_removal.py's convention.

    The license gate is checked here unconditionally -- not only inside
    _get_analyzer()/_get_swapper() -- so that license acceptance is required
    for every swap regardless of whether analyzer/swapper are injected. This
    deviates from the plan's candidate code, where the gate lived solely
    inside _get_analyzer()/_get_swapper() and was therefore bypassed by
    dependency injection; see the module-level docstring / task report for
    why that was a bug against this module's own test suite."""
    _ensure_models_available()
    analyzer = analyzer if analyzer is not None else _get_analyzer()
    swapper = swapper if swapper is not None else _get_swapper()

    source_img = np.array(Image.open(io.BytesIO(source_face_bytes)).convert("RGB"))[:, :, ::-1]
    target_img = np.array(Image.open(io.BytesIO(target_image_bytes)).convert("RGB"))[:, :, ::-1]

    source_faces = analyzer.get(source_img)
    if not source_faces:
        raise NoFaceDetectedError("No face detected in the source image")
    target_faces = analyzer.get(target_img)
    if not target_faces:
        raise NoFaceDetectedError("No face detected in the target image")

    result = swapper.get(target_img, target_faces[0], source_faces[0], paste_back=True)

    out = Image.fromarray(result[:, :, ::-1])
    buf = io.BytesIO()
    out.save(buf, format="PNG", pnginfo=_provenance_metadata())
    return buf.getvalue()
