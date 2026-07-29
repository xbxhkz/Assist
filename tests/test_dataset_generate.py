from src.dataset_tools.generate import build_generation_prompt, parse_generated_rows


def test_prompt_includes_shape_count_and_brief():
    system, user = build_generation_prompt("instruction", 5, "about VFD faults", None)
    assert "instruction" in system and "response" in system
    assert "5" in user and "VFD faults" in user
    assert isinstance(system, str) and isinstance(user, str)


def test_prompt_renders_seed_examples_and_survives_bad_seeds():
    system, user = build_generation_prompt("text", 3, "", [{"text": "seed one"}, "not-a-dict", 42])
    assert "seed one" in user  # dict seed rendered; non-dict seeds skipped, no raise


def test_prompt_unknown_format_falls_back_to_text():
    system, user = build_generation_prompt("bogus", 2, "x", None)
    assert '"text"' in system


def test_parse_plain_jsonl():
    rows = parse_generated_rows('{"text": "a"}\n{"text": "b"}')
    assert rows == [{"text": "a"}, {"text": "b"}]


def test_parse_fenced_and_array_and_garbage():
    fenced = parse_generated_rows('```json\n{"text": "a"}\n{"text": "b"}\n```')
    assert fenced == [{"text": "a"}, {"text": "b"}]
    arr = parse_generated_rows('[{"prompt": "p", "completion": "c"}, {"x": 1}]')
    assert arr == [{"prompt": "p", "completion": "c"}, {"x": 1}]
    assert parse_generated_rows("total garbage, no json here") == []


def test_parse_tolerates_prose_and_truncated_tail():
    text = 'Here you go:\n{"text": "a"}\n{"text": "b"}\n{"text": "trунc'
    rows = parse_generated_rows(text)
    assert {"text": "a"} in rows and {"text": "b"} in rows  # truncated last line dropped


def test_parse_never_raises_on_non_str():
    for bad in (None, 42, ["x"], {"a": 1}):
        assert parse_generated_rows(bad) == []


def test_prompt_unhashable_fmt_never_raises():
    for bad in (["text"], {"a": 1}, {"text"}):
        system, user = build_generation_prompt(bad, 3, "b", None)
        assert '"text"' in system  # falls back to text shape, no raise
