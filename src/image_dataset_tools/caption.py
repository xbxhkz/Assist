"""Auto-caption an image with the shipped vision model, for LoRA-style
training-dataset captions. Pure wrapper, injectable vl_call. Never raises."""

DEFAULT_CAPTION_PROMPT = (
    "Describe this image in one plain, concrete sentence suitable as a "
    "caption for image-generation training data. Do not mention that it is "
    "an image or photo; just describe the subject, setting, and style."
)


def _default_vl_call(path, owner=None, *, prompt=None):
    from src.document_processor import analyze_image_with_vl_result
    return analyze_image_with_vl_result(path, owner=owner, prompt=prompt)


def caption_image(path, *, vl_call=None, prompt=None, owner=None):
    """Return (caption, error) -- exactly one is None. Never raises."""
    try:
        call = vl_call or _default_vl_call
        result = call(path, owner=owner, prompt=prompt or DEFAULT_CAPTION_PROMPT)
    except Exception as e:  # noqa: BLE001
        try:
            msg = str(e).strip()
        except Exception:  # noqa: BLE001
            msg = ""
        return None, msg if msg else "caption failed (no error detail)"
    if not isinstance(result, dict):
        return None, "vision call returned an unexpected result"
    text = result.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, "vision model returned an empty caption"
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        # analyze_image_with_vl_result signals "vision disabled" / "no model
        # configured" with a bracketed marker string, not an exception.
        inner = stripped.strip("[]").strip()
        return None, inner if inner else "vision model is unavailable"
    return stripped, None
