"""Load + normalize a JSONL training dataset into [{"text": ...}] rows.

Accepts three row shapes: {"text"}, {"instruction","response"} (optional
"input"), and {"prompt","completion"}. Pure except reading the file."""
import json


def _normalize(row: dict) -> str:
    text = row.get("text")
    if isinstance(text, str) and text.strip():
        return text
    instr = row.get("instruction")
    if isinstance(instr, str) and instr.strip():
        resp = row.get("response")
        if not (isinstance(resp, str) and resp.strip()):
            resp = row.get("output")
        if not (isinstance(resp, str) and resp.strip()):
            raise ValueError("'instruction' row needs a non-empty string 'response' (or 'output')")
        inp = row.get("input")
        head = f"{instr}\n{inp}" if isinstance(inp, str) and inp.strip() else instr
        return f"{head}\n{resp}"
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        comp = row.get("completion")
        if not (isinstance(comp, str) and comp.strip()):
            raise ValueError("'prompt' row needs a non-empty string 'completion'")
        return f"{prompt}\n{comp}"
    raise ValueError("row needs 'text', or 'instruction'(+response), or 'prompt'(+completion)")


def load_jsonl(path: str) -> list:
    """Return [{"text": str}, ...]. Raises FileNotFoundError / ValueError
    (naming the offending 1-based line)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {i}: invalid JSON ({e})") from e
            if not isinstance(obj, dict):
                raise ValueError(f"line {i}: each row must be a JSON object")
            try:
                text = _normalize(obj)
            except ValueError as e:
                raise ValueError(f"line {i}: {e}") from e
            rows.append({"text": text})
    if not rows:
        raise ValueError("dataset is empty")
    return rows
