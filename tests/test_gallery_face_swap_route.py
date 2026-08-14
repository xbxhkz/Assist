"""POST /api/image/face-swap accepts BOTH the flattened editor canvas
(target) and an uploaded source-face image as real file uploads via
request.form(), mirroring Style Transfer's established FormData pattern
(routes/gallery/gallery_routes.py's /api/gallery/style-transfer) exactly --
NOT the base64-JSON pattern remove-bg's route uses, since this route needs
two real file uploads, not one base64 field. Calls src.face_swap and
returns the result. See
docs/superpowers/specs/2026-08-13-image-editing-face-swap-design.md.
"""
import asyncio
import base64
import io

from PIL import Image

import routes.gallery.gallery_routes as gallery_routes_module
from src.face_swap import LicenseNotAcceptedError, NoFaceDetectedError


def _route(router, path, method):
    for r in router.routes:
        if r.path == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(path)


def _png_bytes(size=(40, 40), color=(10, 20, 30, 255)):
    img = Image.new("RGBA", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeUploadFile:
    """Matches read_upload_limited's contract: async .read(n) accepting a
    size argument (it calls upload.read(limit + 1)), not a bare .read()."""
    def __init__(self, data: bytes):
        self._data = data

    async def read(self, n=None):
        return self._data


class _FakeFormData(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


class _FakeRequest:
    def __init__(self, form_data):
        self._form_data = form_data

    async def form(self):
        return self._form_data


def _stub_gallery_saver(monkeypatch, saved_calls=None, exc=None):
    """Replace the archival Gallery save so tests never touch the real DB or
    generated_images dir. Returns the list every call is recorded into."""
    calls = saved_calls if saved_calls is not None else []

    def _saver(image_bytes, owner, *, prompt=None, model=None):
        calls.append({"bytes": image_bytes, "owner": owner, "prompt": prompt, "model": model})
        if exc is not None:
            raise exc
        return {"id": "gal-1", "filename": "gal-1.png"}

    monkeypatch.setattr(gallery_routes_module, "_default_gallery_saver", _saver)
    return calls


def test_face_swap_route_calls_face_swap_module(monkeypatch):
    monkeypatch.setattr(gallery_routes_module, "require_privilege", lambda request, priv: "alice")
    _stub_gallery_saver(monkeypatch)

    captured = {}

    def fake_swap_face(source_face_bytes, target_image_bytes, **kwargs):
        captured["source"] = source_face_bytes
        captured["target"] = target_image_bytes
        out = Image.open(io.BytesIO(target_image_bytes)).convert("RGBA")
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()

    monkeypatch.setattr(gallery_routes_module, "swap_face", fake_swap_face)

    router = gallery_routes_module.setup_gallery_routes()
    handler = _route(router, "/api/image/face-swap", "POST")

    form = _FakeFormData({
        "image": _FakeUploadFile(_png_bytes()),
        "source_face": _FakeUploadFile(_png_bytes(color=(50, 60, 70, 255))),
    })
    out = asyncio.run(handler(_FakeRequest(form)))

    assert captured["source"] == _png_bytes(color=(50, 60, 70, 255))
    assert captured["target"] == _png_bytes()
    assert "image" in out
    result_img = Image.open(io.BytesIO(base64.b64decode(out["image"])))
    assert result_img.mode == "RGBA"


def test_face_swap_route_returns_error_on_model_failure(monkeypatch):
    monkeypatch.setattr(gallery_routes_module, "require_privilege", lambda request, priv: "alice")
    _stub_gallery_saver(monkeypatch)

    def failing_swap_face(source_face_bytes, target_image_bytes, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(gallery_routes_module, "swap_face", failing_swap_face)

    router = gallery_routes_module.setup_gallery_routes()
    handler = _route(router, "/api/image/face-swap", "POST")

    form = _FakeFormData({
        "image": _FakeUploadFile(_png_bytes()),
        "source_face": _FakeUploadFile(_png_bytes()),
    })
    out = asyncio.run(handler(_FakeRequest(form)))

    assert "error" in out
    assert out["error"] == "Face swap failed"


# ---------------------------------------------------------------------------
# Provenance: the editor draws the returned PNG onto a canvas, which strips
# swap_face()'s embedded `assist:ai-edited` metadata, so the route persists an
# archival, still-marked copy server-side. Best-effort -- never fatal.
# ---------------------------------------------------------------------------

_MARKED_PNG_SENTINEL = b"\x89PNG\r\n\x1a\n-provenance-marked-swap-result"


def _run_swap_route(monkeypatch, swap_impl):
    monkeypatch.setattr(gallery_routes_module, "swap_face", swap_impl)
    router = gallery_routes_module.setup_gallery_routes()
    handler = _route(router, "/api/image/face-swap", "POST")
    form = _FakeFormData({
        "image": _FakeUploadFile(_png_bytes()),
        "source_face": _FakeUploadFile(_png_bytes(color=(50, 60, 70, 255))),
    })
    return asyncio.run(handler(_FakeRequest(form)))


def test_face_swap_route_persists_archival_gallery_copy(monkeypatch):
    monkeypatch.setattr(gallery_routes_module, "require_privilege", lambda request, priv: "carol")
    calls = _stub_gallery_saver(monkeypatch)

    out = _run_swap_route(
        monkeypatch,
        lambda source_face_bytes, target_image_bytes, **kw: _MARKED_PNG_SENTINEL,
    )

    assert len(calls) == 1, calls
    # The EXACT bytes swap_face returned (metadata intact), not a re-encode.
    assert calls[0]["bytes"] == _MARKED_PNG_SENTINEL
    # The user resolved by require_privilege, so the row is owned correctly.
    assert calls[0]["owner"] == "carol"
    assert calls[0]["model"] == "face_swap"
    assert base64.b64decode(out["image"]) == _MARKED_PNG_SENTINEL


def test_face_swap_route_still_returns_result_when_gallery_save_fails(monkeypatch):
    """Best-effort persistence: a saver failure must neither propagate nor
    cost the user the swap result the model already produced."""
    monkeypatch.setattr(gallery_routes_module, "require_privilege", lambda request, priv: "carol")
    calls = _stub_gallery_saver(monkeypatch, exc=RuntimeError("disk on fire"))

    out = _run_swap_route(
        monkeypatch,
        lambda source_face_bytes, target_image_bytes, **kw: _MARKED_PNG_SENTINEL,
    )

    assert len(calls) == 1, "saver should still have been attempted"
    assert "error" not in out, out
    assert base64.b64decode(out["image"]) == _MARKED_PNG_SENTINEL


# ---------------------------------------------------------------------------
# Actionable errors: license-not-accepted and no-face-detected must not be
# flattened into the generic "Face swap failed" by the catch-all handler.
# ---------------------------------------------------------------------------

def test_face_swap_route_reports_license_not_accepted_specifically(monkeypatch):
    monkeypatch.setattr(gallery_routes_module, "require_privilege", lambda request, priv: "alice")
    calls = _stub_gallery_saver(monkeypatch)

    def raise_license(source_face_bytes, target_image_bytes, **kw):
        raise LicenseNotAcceptedError("Face-swap requires accepting InsightFace's model license first")

    out = _run_swap_route(monkeypatch, raise_license)

    assert "error" in out, out
    assert out["error"] != "Face swap failed"
    assert "license" in out["error"].lower()
    assert "AI Defaults" in out["error"]
    assert calls == [], "nothing was produced, so nothing may be persisted"


def test_face_swap_route_reports_no_face_detected_specifically(monkeypatch):
    monkeypatch.setattr(gallery_routes_module, "require_privilege", lambda request, priv: "alice")
    calls = _stub_gallery_saver(monkeypatch)

    def raise_no_face(source_face_bytes, target_image_bytes, **kw):
        raise NoFaceDetectedError("No face detected in the source image")

    out = _run_swap_route(monkeypatch, raise_no_face)

    assert "error" in out, out
    assert out["error"] != "Face swap failed"
    assert "no face detected" in out["error"].lower()
    assert calls == []
