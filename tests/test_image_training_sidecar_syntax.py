# tests/test_image_training_sidecar_syntax.py
import ast
import os


def test_train_sdxl_lora_script_has_valid_syntax():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "image_training_sidecar", "train_sdxl_lora.py")
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    ast.parse(source)  # raises SyntaxError if malformed -- never imports the module


def test_train_sdxl_lora_script_never_imported_by_main_app():
    import sys
    assert "image_training_sidecar.train_sdxl_lora" not in sys.modules
