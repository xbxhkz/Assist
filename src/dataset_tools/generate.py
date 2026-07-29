"""Synthetic training-row generation: ask a model for rows in the target shape,
parse its (messy, untrusted) output, validate + dedup, stage for review.
Pure + injectable model_call. Never raises."""
import json
import math

from src.training.dataset import normalize_row

_SHAPE_DESC = {
    "text": 'exactly one key "text" (the training text)',
    "instruction": 'keys "instruction" and "response" (optionally also "input")',
    "prompt": 'keys "prompt" and "completion"',
}
_SHAPE_EXAMPLE = {
    "text": '{"text": "..."}',
    "instruction": '{"instruction": "...", "response": "..."}',
    "prompt": '{"prompt": "...", "completion": "..."}',
}


def build_generation_prompt(fmt, count, brief, seed_rows=None):
    """Build (system, user) messages instructing the model to emit JSONL rows of
    the target shape. Deterministic, string-only, never raises."""
    fmt = fmt if isinstance(fmt, str) and fmt in _SHAPE_DESC else "text"
    system = (
        "You generate high-quality fine-tuning data for a language model. "
        "Output ONLY JSONL: one JSON object per line, no prose, no markdown, no code fences. "
        f"Each object MUST have {_SHAPE_DESC[fmt]}. Example line: {_SHAPE_EXAMPLE[fmt]}"
    )
    try:
        n = int(count)
    except Exception:  # noqa: BLE001
        n = 1
    parts = [f"Generate {max(1, n)} diverse, high-quality training examples as JSONL."]
    b = brief if isinstance(brief, str) else ""
    if b.strip():
        parts.append("Topic / instructions: " + b.strip())
    examples = []
    for r in (seed_rows if isinstance(seed_rows, list) else []):
        if isinstance(r, dict):
            try:
                examples.append(json.dumps(r, ensure_ascii=False))
            except Exception:  # noqa: BLE001
                pass
        if len(examples) >= 3:
            break
    if examples:
        parts.append("Match the style and format of these examples:\n" + "\n".join(examples))
    parts.append("Output the JSONL now, one object per line.")
    return system, "\n\n".join(parts)


def parse_generated_rows(text):
    """Extract row dicts from a raw model response. Strips code fences, accepts a
    top-level JSON array OR line-delimited JSON objects, skips garbage/truncated
    lines. Returns only dicts. Never raises."""
    if not isinstance(text, str):
        return []
    s = text.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    # Whole-text JSON (array or single object) first.
    try:
        obj = json.loads(s)
    except Exception:  # noqa: BLE001
        obj = None
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        return [obj]
    # Fall back to line-delimited JSON objects.
    out = []
    for line in s.splitlines():
        line = line.strip().rstrip(",")
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(r, dict):
            out.append(r)
    return out


def _sig(row):
    try:
        return json.dumps(row, sort_keys=True, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return repr(row)


async def generate_rows(fmt, count, brief, *, seed_rows=None, existing=None,
                        model_call, batch_size=10, max_attempts=None):
    """Batched synthetic generation. Loops model_call in chunks, validating each
    row via normalize_row and deduping across batches, until it reaches `count`,
    exhausts `max_attempts`, or model_call raises. Never raises."""
    fmt = fmt if isinstance(fmt, str) and fmt in _SHAPE_DESC else "text"
    try:
        count = max(0, int(count))
    except Exception:  # noqa: BLE001
        count = 0
    try:
        batch_size = max(1, int(batch_size))
    except Exception:  # noqa: BLE001
        batch_size = 10
    _default_attempts = min(math.ceil(count / batch_size) * 3, 30) if count else 0
    if max_attempts is None:
        max_attempts = _default_attempts
    else:
        try:
            max_attempts = max(0, int(max_attempts))
        except (TypeError, ValueError):
            max_attempts = _default_attempts
    seen = set()
    for r in (existing if isinstance(existing, list) else []):
        if isinstance(r, dict):
            seen.add(_sig(r))
    seed_rows = seed_rows if isinstance(seed_rows, list) else []
    candidates, accepted, attempts, err = [], 0, 0, None
    while accepted < count and attempts < max_attempts:
        attempts += 1
        _system, _user = build_generation_prompt(fmt, min(batch_size, count - accepted), brief, seed_rows)
        try:
            raw = await model_call(_user, system=_system)
        except Exception as e:  # noqa: BLE001
            try:
                err = f"model call failed: {e}"
            except Exception:  # noqa: BLE001
                err = "model call failed"
            break
        for row in parse_generated_rows(raw if isinstance(raw, str) else ""):
            _text, verr = normalize_row(row)
            if verr:
                candidates.append({"row": row, "valid": False, "error": verr, "duplicate": False})
                continue
            sig = _sig(row)
            if sig in seen:
                candidates.append({"row": row, "valid": True, "error": None, "duplicate": True})
                continue
            seen.add(sig)
            candidates.append({"row": row, "valid": True, "error": None, "duplicate": False})
            accepted += 1
            if accepted >= count:
                break
    report = {
        "rows": candidates,
        "valid": sum(1 for c in candidates if c["valid"] and not c["duplicate"]),
        "invalid": sum(1 for c in candidates if not c["valid"]),
        "duplicates": sum(1 for c in candidates if c["duplicate"]),
        "requested": count,
        "produced": accepted,
        "attempts": attempts,
    }
    if err:
        report["error"] = err
    return report
