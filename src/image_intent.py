"""Heuristic: does a chat message clearly ask to *generate an image*?

Used so a normal text/agent chat can route a plain-language image request
("draw me a red cat") straight to the configured image model, rather than
depending on the chat model to emit a `generate_image` tool call (which weak /
local models and plain Chat mode don't do). Biased toward PRECISION: it fires
only on clear requests, and anything ambiguous falls through to normal chat.
"""
import re

# A creation verb near an image noun, or "<image-noun> of ...", or "draw me a ...".
_IMAGE_REQUEST_RE = re.compile(
    r"\b(?:draw|sketch|paint|render|generate|create|make|produce|design)\b[^.?!\n]{0,40}?\b"
    r"(?:image|picture|photo|photograph|drawing|painting|illustration|artwork|logo|"
    r"icon|wallpaper|portrait|banner|avatar|sticker|poster|scene)s?\b"
    r"|\b(?:image|picture|photo|photograph|drawing|painting|illustration|artwork|portrait|logo)s?\s+of\b"
    r"|\b(?:draw|sketch|paint)\s+(?:me\s+|us\s+)?(?:a|an|the|some|my)\b",
    re.IGNORECASE,
)

# Framings that mean the user is asking ABOUT images, not FOR one. If any of
# these appear, we stay in normal chat even when a positive pattern also matches.
_IMAGE_NEGATIVE_RE = re.compile(
    r"\b(?:how|why|explain|explanation|tutorial|guide|difference|meaning|definition)\b"
    r"|\bwhat(?:'s| is| are| does)\b|\bhistory of\b|\btell me about\b",
    re.IGNORECASE,
)


def looks_like_image_request(text) -> bool:
    """True when `text` is clearly a request to generate an image."""
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t:
        return False
    if _IMAGE_NEGATIVE_RE.search(t):
        return False
    return bool(_IMAGE_REQUEST_RE.search(t))
