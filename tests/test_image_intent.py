"""The image-request heuristic that lets a normal text/agent chat route a clear
'make me a picture' message straight to the image model (instead of relying on
the model to emit a generate_image tool call)."""
from src.image_intent import looks_like_image_request


def test_clear_image_requests_match():
    for m in [
        "draw me a red cat",
        "draw a dog wearing sunglasses",
        "can you draw me a picture of a castle",
        "generate an image of a sunset over the mountains",
        "make a picture of a robot chef",
        "create a logo for my coffee shop",
        "a photo of a mountain lake at dawn",
        "paint a watercolor of a forest",
        "render an illustration of a spaceship",
        "sketch a portrait of an old sailor",
        "design an icon of a rocket",
    ]:
        assert looks_like_image_request(m), f"should match: {m!r}"


def test_non_image_requests_do_not_match():
    for m in [
        "how does image generation work?",
        "what is the best image model?",
        "explain how diffusion models create images",
        "tell me about the history of ai art",
        "why are my images blurry",
        "what's the difference between FLUX and SDXL",
        "write a poem about a sunset",
        "summarize this document",
        "draw me a cat and tell me about cats",  # 'tell me about' -> ambiguous, stays chat
        "hello",
        "",
        None,
    ]:
        assert not looks_like_image_request(m), f"should NOT match: {m!r}"
