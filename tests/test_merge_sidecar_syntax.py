import ast
import pathlib


def test_merge_py_parses():
    p = pathlib.Path(__file__).resolve().parents[1] / "training_sidecar" / "merge.py"
    ast.parse(p.read_text(encoding="utf-8"))  # parse only; never import (needs torch/gguf)
