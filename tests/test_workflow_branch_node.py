import asyncio
import src.workflows.nodes as nd


def _run(coro):
    return asyncio.run(coro)


async def _never(*a, **k):
    raise AssertionError("match mode must not call the model")


def test_match_routes_to_equal_case_case_insensitive():
    out = _run(nd.run_branch({"mode": "match", "cases": ["Yes", "No"]},
                             {"value": " yes "}, model_call=_never))
    assert out == {"active": "Yes", "value": " yes "}


def test_match_falls_to_else_on_no_match():
    out = _run(nd.run_branch({"mode": "match", "cases": ["yes", "no"]},
                             {"value": "maybe"}, model_call=_never))
    assert out["active"] == "else"


def test_llm_routes_to_returned_case():
    async def fake(prompt, model=None, system=None):
        return "no"
    out = _run(nd.run_branch({"mode": "llm", "cases": ["yes", "no"], "prompt": "reply?"},
                             {"value": "an angry email"}, model_call=fake))
    assert out["active"] == "no"


def test_llm_tolerates_chatty_answer_via_substring():
    async def fake(prompt, model=None, system=None):
        return "I think the answer is YES."
    out = _run(nd.run_branch({"mode": "llm", "cases": ["yes", "no"]},
                             {"value": "x"}, model_call=fake))
    assert out["active"] == "yes"


def test_llm_off_list_falls_to_else():
    async def fake(prompt, model=None, system=None):
        return "purple"
    out = _run(nd.run_branch({"mode": "llm", "cases": ["yes", "no"]},
                             {"value": "x"}, model_call=fake))
    assert out["active"] == "else"
