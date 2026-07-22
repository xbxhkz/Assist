"""Load + normalize a JSONL training dataset into [{"text": ...}] rows.

Accepts three row shapes: {"text"}, {"instruction","response"} (optional
"input"), and {"prompt","completion"}. Pure except reading the file."""
import json


def _normalize(row: dict) -> str:
    if isinstance(row.get("text"), str) and row["text"].strip():
        return row["text"]
    if isinstance(row.get("instruction"), str):
        instr = row["instruction"]
        inp = row.get("input")
        resp = row.get("response", row.get("output", ""))
        head = f"{instr}\n{inp}" if isinstance(inp, str) and inp.strip() else instr
        return f"{head}\n{resp}"
    if isinstance(row.get("prompt"), str):
        return f"{row['prompt']}\n{row.get('completion', '')}"
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
