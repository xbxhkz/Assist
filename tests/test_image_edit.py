"""image_edit.edit_image POSTs to sd-server's /sdapi/v1/img2img (live-verified
by this sub-project's feasibility spike to genuinely condition on init_images,
unlike ControlNet's silently-ignored control_image) via an injectable poster,
mirroring src/bg_removal.py's session=None pattern so tests never need a real
running sd-server. See
docs/superpowers/specs/2026-08-12-image-editing-prompt-edit-design.md.
"""
import base64

import pytest

from src import image_edit


def _fake_poster(response=None, capture=None):
    response = response if response is not None else {"images": ["ZmFrZS1wbmc="]}

    def poster(base_url, payload, headers):
        if capture is not None:
            capture["base_url"] = base_url
            capture["payload"] = payload
            capture["headers"] = headers
        return response
    return poster


def test_edit_image_returns_decoded_bytes_from_response():
    result = image_edit.edit_image(
        b"input-png-bytes", "add a red hat", "http://127.0.0.1:8300",
        poster=_fake_poster(response={"images": [base64.b64encode(b"edited-png").decode()]}),
    )
    assert result == b"edited-png"


def test_edit_image_strips_data_uri_prefix_from_response():
    b64 = base64.b64encode(b"edited-png").decode()
    result = image_edit.edit_image(
        b"input-png-bytes", "add a red hat", "http://127.0.0.1:8300",
        poster=_fake_poster(response={"images": [f"data:image/png;base64,{b64}"]}),
    )
    assert result == b"edited-png"


def test_edit_image_sends_expected_payload_shape():
    capture = {}
    image_edit.edit_image(
        b"input-png-bytes", "add a red hat", "http://127.0.0.1:8300",
        poster=_fake_poster(capture=capture),
    )
    payload = capture["payload"]
    assert payload["prompt"] == "add a red hat"
    assert payload["denoising_strength"] == 0.6
    # `steps` is deliberately absent: src/imagemodels/runtime.py already tunes
    # --steps per model family at server-launch time, so sending a fixed value
    # here would override each model's own tuned recipe.
    assert "steps" not in payload
    assert payload["width"] == 512
    assert payload["height"] == 512
    assert payload["init_images"][0].startswith("data:image/png;base64,")
    decoded = base64.b64decode(payload["init_images"][0].split(",", 1)[1])
    assert decoded == b"input-png-bytes"


def test_edit_image_passes_base_url_and_headers_through():
    capture = {}
    image_edit.edit_image(
        b"input-png-bytes", "add a red hat", "http://127.0.0.1:8300",
        headers={"Authorization": "Bearer x"},
        poster=_fake_poster(capture=capture),
    )
    assert capture["base_url"] == "http://127.0.0.1:8300"
    assert capture["headers"] == {"Authorization": "Bearer x"}


def test_img2img_url_strips_v1_suffix_from_registered_endpoint():
    """The production case. A locally served image model is registered as
    http://127.0.0.1:{port}/v1 (src/imagemodels/runtime.py's
    local_image_endpoint_url), and the tool's base_url derivation only strips
    /chat/completions/v1/messages -- neither of which is present -- so the /v1
    survives to here. sd-server's A1111 routes hang off the SERVER ROOT, so
    naively appending would produce a 404 at /v1/sdapi/v1/img2img."""
    assert image_edit._img2img_url("http://127.0.0.1:8200/v1") == \
        "http://127.0.0.1:8200/sdapi/v1/img2img"


def test_img2img_url_uses_base_without_v1_suffix_as_is():
    assert image_edit._img2img_url("http://127.0.0.1:8200") == \
        "http://127.0.0.1:8200/sdapi/v1/img2img"


def test_img2img_url_handles_trailing_slash_on_either_form():
    assert image_edit._img2img_url("http://127.0.0.1:8200/") == \
        "http://127.0.0.1:8200/sdapi/v1/img2img"
    # A trailing slash must not hide the /v1 suffix from the strip check.
    assert image_edit._img2img_url("http://127.0.0.1:8200/v1/") == \
        "http://127.0.0.1:8200/sdapi/v1/img2img"


def test_edit_image_raises_when_no_images_in_response():
    with pytest.raises(RuntimeError):
        image_edit.edit_image(
            b"input-png-bytes", "add a red hat", "http://127.0.0.1:8300",
            poster=_fake_poster(response={"images": []}),
        )
