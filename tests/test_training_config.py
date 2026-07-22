from src.training.config import (TrainingConfig, estimate_vram_gb, fit_level,
                                 parse_params_b)


def _cfg(**kw):
    base = dict(base_model="Qwen/Qwen2.5-0.5B-Instruct", dataset_path="d.jsonl",
                lora_r=8, lora_alpha=16, lora_dropout=0.05, steps=100, epochs=None,
                batch_size=1, learning_rate=2e-4)
    base.update(kw)
    return TrainingConfig(**base)


def test_valid_config_has_no_errors():
    assert _cfg().validate() == []


def test_invalid_configs_report_errors():
    assert _cfg(base_model="").validate()
    assert _cfg(lora_r=0).validate()
    assert _cfg(lora_dropout=1.5).validate()
    assert _cfg(batch_size=0).validate()
    assert _cfg(learning_rate=0).validate()
    assert _cfg(steps=None, epochs=None).validate()      # need one
    assert _cfg(steps=100, epochs=3).validate()          # not both


def test_estimate_and_fit_level():
    assert estimate_vram_gb(0.5) < 2.0
    assert fit_level(0.5, 6.44) == "fits"
    assert fit_level(13, 6.44) == "too_big"
    assert fit_level(3, 6.44) == "fits"
    assert fit_level(0.5, None) == "unknown"


def test_parse_params_b():
    assert parse_params_b("Qwen/Qwen2.5-0.5B-Instruct") == 0.5
    assert parse_params_b("meta-llama/Meta-Llama-3-8B") == 8.0
    assert parse_params_b("TinyLlama/TinyLlama-1.1B") == 1.1
    assert parse_params_b("openai-community/gpt2") is None
