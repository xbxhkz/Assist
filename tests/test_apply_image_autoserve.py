import asyncio
import src.ai_interaction as ai


def _run(coro):
    return asyncio.run(coro)


def test_explicit_model_skips_autoserve():
    calls = []
    def ensure(owner):
        calls.append(owner)
        return {"model": "should-not-be-used", "error": None}
    spec, err = _run(ai._apply_image_autoserve("gpt-image-1.5", True, "u", ensure=ensure))
    assert spec == "gpt-image-1.5" and err is None
    assert calls == []                      # ensure not called when explicit


def test_local_default_uses_served_id():
    def ensure(owner):
        return {"model": "flux.gguf", "error": None}
    spec, err = _run(ai._apply_image_autoserve("", False, "u", ensure=ensure))
    assert spec == "flux.gguf" and err is None


def test_serve_error_is_returned():
    def ensure(owner):
        return {"model": None, "error": "sd-server did not start"}
    spec, err = _run(ai._apply_image_autoserve("", False, "u", ensure=ensure))
    assert err == "sd-server did not start"


def test_external_default_leaves_spec_untouched():
    def ensure(owner):
        return {"model": None, "error": None}
    spec, err = _run(ai._apply_image_autoserve("gpt-image-1.5", False, "u", ensure=ensure))
    assert spec == "gpt-image-1.5" and err is None
