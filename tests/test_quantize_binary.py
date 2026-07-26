import os
import pytest
from src.localmodels.runtime import resolve_quantize_binary


def test_resolves_bundled_frozen(tmp_path):
    d = tmp_path / "llama" / "cpu"
    d.mkdir(parents=True)
    exe = d / ("llama-quantize.exe" if os.name == "nt" else "llama-quantize")
    exe.write_text("x")
    assert resolve_quantize_binary(device="cpu", frozen_base=str(tmp_path)) == str(exe)


def test_resolves_dev(tmp_path):
    d = tmp_path / "vulkan"
    d.mkdir(parents=True)
    exe = d / ("llama-quantize.exe" if os.name == "nt" else "llama-quantize")
    exe.write_text("x")
    assert resolve_quantize_binary(device="gpu", frozen_base=str(tmp_path / "none"),
                                   dev_base=str(tmp_path)) == str(exe)


def test_raises_when_missing(tmp_path):
    with pytest.raises(RuntimeError):
        resolve_quantize_binary(frozen_base=str(tmp_path / "a"), dev_base=str(tmp_path / "b"))
