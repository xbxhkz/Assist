from src.image_training.config import ImageTrainingConfig


def _valid():
    return ImageTrainingConfig(dataset_name="ds1", output_name="my-lora")


def test_validate_accepts_defaults_plus_required():
    assert _valid().validate() == []


def test_validate_rejects_missing_dataset_name():
    cfg = ImageTrainingConfig(dataset_name="", output_name="my-lora")
    errs = cfg.validate()
    assert any("dataset_name" in e for e in errs)


def test_validate_rejects_missing_output_name():
    cfg = ImageTrainingConfig(dataset_name="ds1", output_name="")
    errs = cfg.validate()
    assert any("output_name" in e for e in errs)


def test_validate_rejects_non_positive_rank():
    cfg = ImageTrainingConfig(dataset_name="ds1", output_name="my-lora", rank=0)
    errs = cfg.validate()
    assert any("rank" in e for e in errs)


def test_validate_rejects_bad_learning_rate():
    cfg = ImageTrainingConfig(dataset_name="ds1", output_name="my-lora", learning_rate=0)
    errs = cfg.validate()
    assert any("learning_rate" in e for e in errs)


def test_validate_rejects_unsupported_base_model():
    cfg = ImageTrainingConfig(dataset_name="ds1", output_name="my-lora",
                              base_model="black-forest-labs/FLUX.1-schnell")
    errs = cfg.validate()
    assert any("not supported" in e for e in errs)


def test_validate_rejects_non_positive_steps_and_resolution():
    cfg = ImageTrainingConfig(dataset_name="ds1", output_name="my-lora", steps=0, resolution=-1)
    errs = cfg.validate()
    assert any("steps" in e for e in errs)
    assert any("resolution" in e for e in errs)


def test_to_dict_roundtrip():
    cfg = _valid()
    d = cfg.to_dict()
    assert d["dataset_name"] == "ds1" and d["output_name"] == "my-lora"
    assert d["rank"] == 4 and d["lora_alpha"] == 4 and d["resolution"] == 1024
