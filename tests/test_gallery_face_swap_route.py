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


def test_face_swap_route_calls_face_swap_module(monkeypatch):
    monkeypatch.setattr(gallery_routes_module, "require_privilege", lambda request, priv: "alice")

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
