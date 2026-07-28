from src.dataset_tools.validate import validate_rows, validate_jsonl_text


def test_all_valid_stats():
    rep = validate_rows([{"text": "hello"}, {"instruction": "a", "response": "bb"}])
    assert rep["total"] == 2 and rep["valid"] == 2 and rep["invalid"] == 0
    assert rep["stats"]["shapes"] == {"text": 1, "instruction": 1, "prompt": 0}
    assert rep["stats"]["char_len"]["max"] >= 4 and rep["stats"]["approx_tokens"] >= 1


def test_collects_every_bad_row():
    rep = validate_rows([{"text": "ok"}, {"nope": 1}, {"instruction": "x"}])
    assert rep["valid"] == 1 and rep["invalid"] == 2
    assert [e["line"] for e in rep["errors"]] == [2, 3]


def test_text_reports_invalid_json_and_line_numbers():
    rep = validate_jsonl_text('{"text": "ok"}\n\n{not json\n{"prompt":"Q","completion":"A"}')
    assert rep["valid"] == 2 and rep["invalid"] == 1
    assert rep["errors"][0]["line"] == 3 and "JSON" in rep["errors"][0]["message"]


def test_never_raises_on_hostile():
    assert validate_rows("junk")["invalid"] >= 0
    assert validate_rows(None)["total"] == 0
    assert validate_jsonl_text(None)["total"] == 0
