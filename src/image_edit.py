"""Natural-language image editing via img2img on the bundled sd-server
binary. This sub-project's own feasibility spike live-verified that
init_images/denoising_strength genuinely condition sd-server's output --
the opposite result from a prior ControlNet sub-project, where control_image
was accepted by the API but silently ignored. See
docs/superpowers/specs/2026-08-12-image-editing-prompt-edit-design.md.
"""
import base64

import httpx

# Fixed generation parameters for v1 (not exposed as tool parameters, per the
# approved design's "no tunable knobs" decision):
_WIDTH = 512
_HEIGHT = 512  # Matches this app's own established OOM-avoidance precedent
# for local diffusion on small GPUs (src/ai_interaction.py's
# _local_diffusion_size_hint: FLUX OOMs at 1024x1024 on a 6GB card, works
# reliably at 512x512) and the feasibility spike's own tested configuration.
_DENOISING_STRENGTH = 0.6  # The spike's validated balance point: a real,
# visible edit that still resembles the original image.
_STEPS = 20


def _post(base_url: str, payload: dict, headers: dict) -> dict:
    with httpx.Client(timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)) as client:
        resp = client.post(base_url + "/sdapi/v1/img2img", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


def edit_image(image_bytes: bytes, prompt: str, base_url: str, *, headers=None, poster=None) -> bytes:
    """Edit image_bytes per prompt via img2img on the sd-server instance at
    base_url. Returns PNG bytes. Raises on failure -- the never-raises
    discipline is applied by callers, matching src/bg_removal.py's convention.
    """
    payload = {
        "init_images": ["data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")],
        "prompt": prompt,
        "denoising_strength": _DENOISING_STRENGTH,
        "steps": _STEPS,
        "width": _WIDTH,
        "height": _HEIGHT,
    }
    post = poster or _post
    data = post(base_url, payload, headers or {})

    images = data.get("images") or []
    if not images:
        raise RuntimeError("sd-server returned no image from img2img")
    b64 = images[0]
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)
