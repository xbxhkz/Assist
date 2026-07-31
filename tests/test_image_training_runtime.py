import pytest
from src.image_training import runtime


def test_resolve_prefers_frozen(tmp_path):
    base = tmp_path / "frozen"
    (base / "image_training_sidecar").mkdir(parents=True)
    scr = base / "image_training_sidecar" / "train_sdxl_lora.py"
    scr.write_text("x")
    assert runtime.resolve_image_sidecar_script(frozen_base=str(base)) == str(scr)


def test_resolve_falls_back_to_dev(tmp_path):
    dev = tmp_path / "image_training_sidecar"
    dev.mkdir(parents=True)
    scr = dev / "train_sdxl_lora.py"
    scr.write_text("x")
    assert runtime.resolve_image_sidecar_script(frozen_base=str(tmp_path / "none"),
                                                 dev_base=str(dev)) == str(scr)


def test_resolve_raises_when_missing(tmp_path):
    with pytest.raises(RuntimeError):
        runtime.resolve_image_sidecar_script(frozen_base=str(tmp_path / "a"),
                                              dev_base=str(tmp_path / "b"))
