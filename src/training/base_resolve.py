"""Pair a trained adapter with a local base GGUF by name. Pure (no I/O)."""
import re

_QUANT = re.compile(r"^(q\d.*|iq\d.*|f16|f32|bf16|k|m|s|l|xl|gguf|gg)$", re.I)
_SIZE = re.compile(r"^\d+(?:\.\d+)?b$", re.I)


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


def resolve_base_gguf(base_model, gguf_names) -> dict:
    """Best local GGUF whose name matches the adapter's base_model (same family
    + same parameter-size token). Returns {"matched","candidates"}."""
    want = _tokens(base_model)
    want_set = set(want)
    want_size = _size(want)
    names = gguf_names if isinstance(gguf_names, list) else []
    best, best_score, cands = None, 0, []
    for name in names:
        if not isinstance(name, str):
            continue
        have = set(_tokens(name))
        # a candidate must share the parameter-size token (when the base has one)
        if want_size is not None and want_size not in have:
            continue
        score = len(want_set & have)
        if score >= max(1, len(want_set) - 1):
            cands.append(name)
            if score > best_score:
                best, best_score = name, score
    return {"matched": best, "candidates": cands}
