from src.training.base_resolve import resolve_base_gguf

GGUFS = ["qwen2.5-0.5b-instruct-q4_k_m.gguf", "qwen2.5-1.5b-instruct-q4_k_m.gguf",
         "llama-3.2-1b-instruct-q8_0.gguf"]


def test_matches_same_size_family():
    out = resolve_base_gguf("Qwen/Qwen2.5-0.5B-Instruct", GGUFS)
    assert out["matched"] == "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def test_wrong_size_not_matched():
    # only a 1.5B gguf present for a 0.5B base -> no match
    out = resolve_base_gguf("Qwen/Qwen2.5-0.5B-Instruct",
                            ["qwen2.5-1.5b-instruct-q4_k_m.gguf"])
    assert out["matched"] is None


def test_no_candidates_when_family_absent():
    out = resolve_base_gguf("mistralai/Mistral-7B-Instruct-v0.3", GGUFS)
    assert out["matched"] is None and out["candidates"] == []


def test_non_str_inputs_safe():
    assert resolve_base_gguf(None, GGUFS)["matched"] is None
    assert resolve_base_gguf("Qwen/Qwen2.5-0.5B", None)["candidates"] == []
