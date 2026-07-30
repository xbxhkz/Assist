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


import asyncio
from src.dataset_tools.generate import generate_rows


def _run(coro):
    return asyncio.run(coro)


def test_generate_happy_reaches_count():
    async def fake(prompt, system=None):
        return '{"text": "a"}\n{"text": "b"}'
    rep = _run(generate_rows("text", 2, "brief", model_call=fake, batch_size=10))
    assert rep["produced"] == 2 and rep["valid"] == 2 and rep["requested"] == 2
    assert [c["row"] for c in rep["rows"] if c["valid"] and not c["duplicate"]] == [{"text": "a"}, {"text": "b"}]


def test_generate_batches_across_calls():
    calls = {"n": 0}
    async def fake(prompt, system=None):
        calls["n"] += 1
        return '{"text": "row%d"}' % calls["n"]  # one new row per call
    rep = _run(generate_rows("text", 3, "b", model_call=fake, batch_size=1))
    assert rep["produced"] == 3 and calls["n"] == 3


def test_generate_flags_duplicates_against_existing_and_within_batch():
    async def fake(prompt, system=None):
        return '{"text": "dup"}\n{"text": "dup"}\n{"text": "new"}'
    rep = _run(generate_rows("text", 5, "b", existing=[{"text": "dup"}], model_call=fake, batch_size=10, max_attempts=1))
    assert rep["duplicates"] >= 2 and rep["produced"] == 1
    assert any(c["row"] == {"text": "new"} and c["valid"] and not c["duplicate"] for c in rep["rows"])


def test_generate_reports_invalid_rows():
    async def fake(prompt, system=None):
        return '{"nope": 1}\n{"text": "ok"}'
    rep = _run(generate_rows("text", 5, "b", model_call=fake, batch_size=10, max_attempts=1))
    assert rep["invalid"] >= 1 and rep["produced"] == 1


def test_generate_model_error_is_reported_not_raised():
    async def boom(prompt, system=None):
        raise RuntimeError("no default model endpoint configured")
    rep = _run(generate_rows("text", 3, "b", model_call=boom, batch_size=10))
    assert rep["produced"] == 0 and "error" in rep and "endpoint" in rep["error"]


def test_generate_max_attempts_bounds_a_duplicate_only_model():
    async def fake(prompt, system=None):
        return '{"text": "same"}'
    rep = _run(generate_rows("text", 10, "b", model_call=fake, batch_size=1, max_attempts=4))
    assert rep["produced"] == 1 and rep["attempts"] == 4  # never reaches 10


def test_generate_hostile_model_output_never_raises():
    async def fake(prompt, system=None):
        return 42  # not a string
    rep = _run(generate_rows("text", 3, "b", model_call=fake, batch_size=1, max_attempts=2))
    assert rep["produced"] == 0 and rep["attempts"] == 2


def test_generate_unhashable_fmt_never_raises():
    async def fake(prompt, system=None):
        return '{"text": "a"}'
    rep = _run(generate_rows(["text"], 1, "b", model_call=fake, batch_size=1))
    assert rep["produced"] == 1  # unhashable fmt falls back to text, no raise


def test_generate_bad_max_attempts_never_raises():
    async def fake(prompt, system=None):
        return '{"text": "a"}'
    rep = _run(generate_rows("text", 2, "b", model_call=fake, batch_size=1, max_attempts="oops"))
    assert rep["produced"] == 1  # coerced to default, no raise


def test_generate_overflow_max_attempts_never_raises():
    async def fake(prompt, system=None):
        return '{"text": "a"}'
    rep = _run(generate_rows("text", 3, "b", model_call=fake, batch_size=1, max_attempts=float("inf")))
    assert rep["produced"] == 1  # inf max_attempts coerced to default, no raise


def test_prompt_context_grounds_and_is_backward_compatible():
    base_sys, base_usr = build_generation_prompt("text", 3, "b", None)
    # no context (explicit None) == prior behavior
    n_sys, n_usr = build_generation_prompt("text", 3, "b", None, None)
    assert (n_sys, n_usr) == (base_sys, base_usr)
    assert "Source text" not in base_usr
    # with context: grounding instruction in system + the passage in the user message
    g_sys, g_usr = build_generation_prompt("text", 3, "b", None, "PUMP TRIP CODE E12 = overpressure")
    assert "ONLY" in g_sys and "source" in g_sys.lower()
    assert "E12 = overpressure" in g_usr and "Source text" in g_usr


def test_generate_rows_threads_context_into_prompt():
    seen = {}
    async def fake(prompt, system=None):
        seen["prompt"] = prompt
        return '{"text": "a"}'
    rep = _run(generate_rows("text", 1, "", model_call=fake, batch_size=1,
                              context="GROUNDING SOURCE XYZ"))
    assert "GROUNDING SOURCE XYZ" in seen["prompt"]


def test_generate_surfaces_model_call_error_verbatim_not_wrapped():
    # A model_call RuntimeError with clear, actionable text must reach rep["error"]
    # unprefixed -- no generic "model call failed:" boilerplate burying the message.
    async def boom(prompt, system=None):
        raise RuntimeError("no chat model available — serve a local model or set a Default Chat Model in Settings")
    rep = _run(generate_rows("text", 1, "b", model_call=boom, batch_size=1))
    assert rep["error"] == "no chat model available — serve a local model or set a Default Chat Model in Settings"
    assert not rep["error"].startswith("model call failed")


def test_generate_names_empty_message_timeout_exception():
    # httpx's own timeout exceptions (ReadTimeout et al.) str() to '' on a real
    # timeout -- the error must name the failure kind, not read as blank/opaque.
    class _FakeReadTimeout(Exception):
        pass
    _FakeReadTimeout.__name__ = "ReadTimeout"

    async def boom(prompt, system=None):
        raise _FakeReadTimeout("")  # empty message, mirrors real httpx behavior

    rep = _run(generate_rows("text", 1, "b", model_call=boom, batch_size=1))
    assert "did not respond in time" in rep["error"] and "ReadTimeout" in rep["error"]


def test_generate_names_empty_message_non_timeout_exception():
    class _FakeWeirdError(Exception):
        pass

    async def boom(prompt, system=None):
        raise _FakeWeirdError()  # empty message, not a timeout

    rep = _run(generate_rows("text", 1, "b", model_call=boom, batch_size=1))
    assert rep["error"] == "model call failed (_FakeWeirdError)"
