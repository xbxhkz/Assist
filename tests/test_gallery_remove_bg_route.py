"""POST /api/image/remove-bg's model-inference step now calls
src.bg_removal.remove_background instead of the broken rembg/transformers
imports (rembg is Windows-unsupported; transformers/RMBG-1.4 is not
bundled). The surrounding hint-mask crop/compose/alpha-multiply logic is
unchanged. See
docs/superpowers/specs/2026-08-12-image-editing-background-removal-design.md.
"""
import asyncio
import base64
import io
from types import SimpleNamespace

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


class _FakeRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def test_remove_bg_route_calls_bg_removal_module(monkeypatch):
    monkeypatch.setattr(gallery_routes_module, "require_privilege", lambda request, priv: "alice")

    captured = {}

    def fake_run_bg_removal_model(image_bytes, **kwargs):
        captured["called_with_bytes"] = image_bytes
        # Return a same-size RGBA PNG so the route's compose step succeeds.
        out = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()

    monkeypatch.setattr(gallery_routes_module, "run_bg_removal_model", fake_run_bg_removal_model)

    router = gallery_routes_module.setup_gallery_routes()
    handler = _route(router, "/api/image/remove-bg", "POST")

    image_b64 = base64.b64encode(_png_bytes()).decode()
    out = asyncio.run(handler(_FakeRequest({"image": image_b64})))

    assert "called_with_bytes" in captured
    assert "image" in out
    result_img = Image.open(io.BytesIO(base64.b64decode(out["image"])))
    assert result_img.mode == "RGBA"


def test_remove_bg_route_returns_error_on_model_failure(monkeypatch):
    monkeypatch.setattr(gallery_routes_module, "require_privilege", lambda request, priv: "alice")

    def failing_run_bg_removal_model(image_bytes, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(gallery_routes_module, "run_bg_removal_model", failing_run_bg_removal_model)

    router = gallery_routes_module.setup_gallery_routes()
    handler = _route(router, "/api/image/remove-bg", "POST")

    image_b64 = base64.b64encode(_png_bytes()).decode()
    out = asyncio.run(handler(_FakeRequest({"image": image_b64})))

    assert "error" in out
