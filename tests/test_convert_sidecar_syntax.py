import ast
import pathlib


def test_convert_py_parses():
    p = pathlib.Path(__file__).resolve().parents[1] / "training_sidecar" / "convert.py"
    ast.parse(p.read_text(encoding="utf-8"))  # parse only; never import (needs gguf/torch)
