from src.image_dataset_tools.caption import caption_image, DEFAULT_CAPTION_PROMPT


def test_caption_image_happy_path():
    def fake_vl(path, owner=None, *, prompt=None):
        assert prompt == DEFAULT_CAPTION_PROMPT  # default prompt used
        return {"text": "a red sports car on a city street", "model": "qwen-vl"}
    caption, error = caption_image("photo.jpg", vl_call=fake_vl)
    assert caption == "a red sports car on a city street" and error is None


def test_caption_image_custom_prompt_threaded():
    seen = {}
    def fake_vl(path, owner=None, *, prompt=None):
        seen["prompt"] = prompt
        return {"text": "x", "model": "m"}
    caption_image("p.jpg", vl_call=fake_vl, prompt="custom prompt")
    assert seen["prompt"] == "custom prompt"


def test_caption_image_reports_missing_text():
    def fake_vl(path, owner=None, *, prompt=None):
        return {"text": "", "model": "m"}
    caption, error = caption_image("p.jpg", vl_call=fake_vl)
    assert caption is None and "empty" in error.lower()


def test_caption_image_never_raises_on_vl_error():
    def boom(path, owner=None, *, prompt=None):
        raise RuntimeError("vision model unavailable")
    caption, error = caption_image("p.jpg", vl_call=boom)
    assert caption is None and "vision model unavailable" in error


def test_caption_image_never_raises_on_non_dict_result():
    def bad_vl(path, owner=None, *, prompt=None):
        return "not-a-dict"
    caption, error = caption_image("p.jpg", vl_call=bad_vl)
    assert caption is None and error is not None


def test_caption_image_bracketed_unavailable_marker_is_an_error():
    # analyze_image_with_vl_result signals unavailability with a bracketed
    # marker string in "text" (e.g. "[No vision model configured — ...]"),
    # not an exception -- must be treated as an error, not a real caption.
    def fake_vl(path, owner=None, *, prompt=None):
        return {"text": "[No vision model configured — set one in Settings → Vision]", "model": ""}
    caption, error = caption_image("p.jpg", vl_call=fake_vl)
    assert caption is None and "vision model" in error.lower()
