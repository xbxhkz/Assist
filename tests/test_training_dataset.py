import json
import pytest
from src.training.dataset import load_jsonl


def _write(tmp_path, rows):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def test_text_rows(tmp_path):
    out = load_jsonl(_write(tmp_path, [{"text": "hello"}, {"text": "world"}]))
    assert out == [{"text": "hello"}, {"text": "world"}]


def test_instruction_response_rows(tmp_path):
    out = load_jsonl(_write(tmp_path, [{"instruction": "greet", "response": "hi"}]))
    assert out[0]["text"].startswith("greet") and "hi" in out[0]["text"]


def test_prompt_completion_rows(tmp_path):
    out = load_jsonl(_write(tmp_path, [{"prompt": "Q", "completion": "A"}]))
    assert "Q" in out[0]["text"] and "A" in out[0]["text"]


def test_bad_row_raises_with_line(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"text": "ok"}\n{"nope": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        load_jsonl(str(p))
    assert "2" in str(ei.value)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_jsonl(str(tmp_path / "no.jsonl"))


def test_instruction_missing_response_raises(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"instruction": "greet"}\n', encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        load_jsonl(str(p))
    assert "1" in str(ei.value)


def test_null_completion_raises_with_line(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"text": "ok"}\n{"prompt": "Q", "completion": null}\n', encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        load_jsonl(str(p))
    assert "2" in str(ei.value)


def test_output_is_response_fallback(tmp_path):
    out = load_jsonl(_write(tmp_path, [{"instruction": "greet", "output": "hi there"}]))
    assert "hi there" in out[0]["text"]


def test_blank_lines_skipped(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('\n{"text": "a"}\n\n{"text": "b"}\n\n', encoding="utf-8")
    assert load_jsonl(str(p)) == [{"text": "a"}, {"text": "b"}]


def test_all_blank_raises_empty(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('\n\n', encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        load_jsonl(str(p))
    assert "empty" in str(ei.value)


def test_invalid_json_line_raises_with_line(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"text": "ok"}\n{not json\n', encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        load_jsonl(str(p))
    assert "2" in str(ei.value)


def test_non_object_line_raises(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('[1, 2, 3]\n', encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        load_jsonl(str(p))
    assert "1" in str(ei.value)
