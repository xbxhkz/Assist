from src.training.dataset import normalize_row


def test_text_row():
    assert normalize_row({"text": "hi"}) == ("hi", None)


def test_instruction_response():
    t, e = normalize_row({"instruction": "greet", "response": "hi"})
    assert e is None and t == "greet\nhi"


def test_instruction_with_input_and_output_fallback():
    t, e = normalize_row({"instruction": "sum", "input": "1+1", "output": "2"})
    assert e is None and t == "sum\n1+1\n2"


def test_prompt_completion():
    assert normalize_row({"prompt": "Q", "completion": "A"}) == ("Q\nA", None)


def test_missing_companion_is_error_not_raise():
    t, e = normalize_row({"instruction": "x"})
    assert t is None and "response" in e


def test_unknown_and_nondict():
    assert normalize_row({"nope": 1})[0] is None
    assert normalize_row("junk")[0] is None
    assert normalize_row(None)[1] is not None
