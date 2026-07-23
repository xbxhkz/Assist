"""Pair a trained adapter with a local base GGUF by name. Pure (no I/O).

Precision-biased: a wrong match causes garbage output, so this requires the
model family AND the parameter-size to agree. A missed match (false negative)
is safe — the UI falls back to letting the user pick/download the base.
"""
import re

_QUANT = re.compile(r"^(q\d.*|iq\d.*|f16|f32|bf16|k|m|s|l|xl|gguf|gg)$", re.I)
_SIZE = re.compile(r"^\d+(?:\.\d+)?[bm]$|^\d+x\d+(?:\.\d+)?b$", re.I)
# generic instruction-tuning suffixes shared across families — must not decide a match
_GENERIC = {"instruct", "chat", "it", "base", "sft", "dpo", "chatml"}


def _tokens(s):
    if not isinstance(s, str):
        return []
    s = s.rsplit("/", 1)[-1].lower()
    s = re.sub(r"\.gguf$", "", s)
    toks = [t for t in re.split(r"[^a-z0-9.]+", s) if t]
    return [t for t in toks if not _QUANT.match(t)]


def _size(toks):
    for t in toks:
        if _SIZE.match(t):
            return t
    return None


def _family(toks):
    return {t for t in toks if not _SIZE.match(t) and t not in _GENERIC}


def resolve_base_gguf(base_model, gguf_names) -> dict:
    """Best local GGUF whose name matches the adapter's base_model — same model
    family AND same parameter-size token. Returns {"matched","candidates"}
    (first-seen wins ties). No match when the base gives nothing concrete to
    match on, or on non-str input."""
    want = _tokens(base_model)
    want_size = _size(want)
    want_family = _family(want)
    if not want_family and want_size is None:
        return {"matched": None, "candidates": []}
    names = gguf_names if isinstance(gguf_names, list) else []
    matched, cands = None, []
    for name in names:
        if not isinstance(name, str):
            continue
        have = set(_tokens(name))
        if want_size is not None and want_size not in have:
            continue  # parameter size must agree
        if want_family and not want_family.issubset(have):
            continue  # every family token must be present
        cands.append(name)
        if matched is None:
            matched = name
    return {"matched": matched, "candidates": cands}
