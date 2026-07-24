import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_convert_py_parses():
    p = ROOT / "training_sidecar" / "convert.py"
    ast.parse(p.read_text(encoding="utf-8"))  # parse only; never import (needs gguf/torch)


def test_vendored_gguf_matches_conversion_package():
    """The vendored conversion/ package (llama.cpp c0bc859) references newer gguf
    enums (e.g. MODEL_ARCH.DFLASH) that the PyPI gguf 0.19.0 *release* lacks — both
    self-report version '0.19.0', which masks the mismatch. The matching gguf MUST
    be vendored beside the convert scripts so `import gguf` resolves to it (sys.path[0]
    when convert_lora_to_gguf.py runs), not a stale site-packages copy."""
    consts = ROOT / "training_sidecar" / "gguf" / "constants.py"
    assert consts.is_file(), "vendored training_sidecar/gguf/ package is missing"
    assert "DFLASH" in consts.read_text(encoding="utf-8"), \
        "vendored gguf is stale — it doesn't match the conversion/ package (no DFLASH)"


def test_stack_does_not_pin_pypi_gguf():
    """gguf is vendored (matched to conversion/), so the sidecar venv must NOT pip a
    mismatched PyPI gguf that would self-report the same version but lack newer enums."""
    from src.training.env import STACK
    assert not any(str(s).lower().startswith("gguf") for s in STACK), \
        "STACK pins a PyPI gguf — the vendored training_sidecar/gguf must be used instead"
