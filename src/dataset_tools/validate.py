"""Validate a training dataset (all rows, non-raising) + compute stats. Pure."""
import json

from src.training.dataset import normalize_row


def _shape_of(row):
    if isinstance(row, dict):
        for key in ("text", "instruction", "prompt"):
            v = row.get(key)
            if isinstance(v, str) and v.strip():
                return key
    return "unknown"


def _len_stats(texts):
    if not texts:
        return {"char_len": {"min": 0, "max": 0, "avg": 0}, "approx_tokens": 0}
    lens = [len(t) for t in texts]
    return {"char_len": {"min": min(lens), "max": max(lens), "avg": round(sum(lens) / len(lens), 1)},
            "approx_tokens": round(sum(lens) / 4)}


def _report(total, texts, errors):
    shapes = {"text": 0, "instruction": 0, "prompt": 0}
    return {"total": total, "valid": len(texts), "invalid": len(errors),
            "errors": errors, "stats": {**_len_stats(texts), "shapes": shapes}}


def validate_rows(rows) -> dict:
    """Validate a list of already-parsed row dicts (numbered by position)."""
    if not isinstance(rows, list):
        return {"total": 0, "valid": 0, "invalid": 0,
                "errors": [{"line": 0, "message": "rows must be a list"}],
                "stats": _len_stats([]) | {"shapes": {"text": 0, "instruction": 0, "prompt": 0}}}
    errors, texts = [], []
    shapes = {"text": 0, "instruction": 0, "prompt": 0}
    for i, row in enumerate(rows, start=1):
        text, err = normalize_row(row)
        if err:
            errors.append({"line": i, "message": err})
            continue
        texts.append(text)
        s = _shape_of(row)
        if s in shapes:
            shapes[s] += 1
    return {"total": len(rows), "valid": len(texts), "invalid": len(errors),
            "errors": errors, "stats": {**_len_stats(texts), "shapes": shapes}}


def validate_jsonl_text(text) -> dict:
    """Validate raw JSONL text (numbered by source line; reports invalid JSON)."""
    errors, texts = [], []
    shapes = {"text": 0, "instruction": 0, "prompt": 0}
    total = 0
    for i, line in enumerate((text or "").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            obj = json.loads(line)
        except Exception as e:  # noqa: BLE001
            errors.append({"line": i, "message": f"invalid JSON ({e})"})
            continue
        t, err = normalize_row(obj)
        if err:
            errors.append({"line": i, "message": err})
            continue
        texts.append(t)
        s = _shape_of(obj)
        if s in shapes:
            shapes[s] += 1
    return {"total": total, "valid": len(texts), "invalid": len(errors),
            "errors": errors, "stats": {**_len_stats(texts), "shapes": shapes}}
