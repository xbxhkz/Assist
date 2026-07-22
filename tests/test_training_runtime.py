import os
import pytest
from src.training import runtime


def test_resolve_uv_prefers_frozen(tmp_path):
    base = tmp_path / "frozen"
    (base / "uv").mkdir(parents=True)
    exe = base / "uv" / "uv.exe"
    exe.write_text("x")
    assert runtime.resolve_uv_binary(frozen_base=str(base)) == str(exe)


def test_resolve_uv_falls_back_to_dev(tmp_path):
    dev = tmp_path / "build_assets" / "uv"
    dev.mkdir(parents=True)
    exe = dev / "uv.exe"
    exe.write_text("x")
    assert runtime.resolve_uv_binary(frozen_base=str(tmp_path / "none"),
                                     dev_base=str(dev)) == str(exe)


def test_resolve_uv_raises_when_missing(tmp_path):
    with pytest.raises(RuntimeError):
        runtime.resolve_uv_binary(frozen_base=str(tmp_path / "a"),
                                  dev_base=str(tmp_path / "b"))


def test_resolve_sidecar_script(tmp_path):
    base = tmp_path / "frozen"
    (base / "training_sidecar").mkdir(parents=True)
    scr = base / "training_sidecar" / "train.py"
    scr.write_text("x")
    assert runtime.resolve_sidecar_script(frozen_base=str(base)) == str(scr)
