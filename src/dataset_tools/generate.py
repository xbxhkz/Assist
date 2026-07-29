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
