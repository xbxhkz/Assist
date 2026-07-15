import os
from contextlib import contextmanager
import pytest
import src.imagemodels.loras as loras


def test_list_only_safetensors_and_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(loras, "IMAGE_MODELS_DIR", str(tmp_path))
    d = loras.loras_dir()
    open(os.path.join(d, "styleA.safetensors"), "wb").write(b"x")
    open(os.path.join(d, "note.txt"), "w").write("n")
    assert [x["name"] for x in loras.list_loras()] == ["styleA"]
    assert loras.list_loras()[0]["filename"] == "styleA.safetensors"
    assert loras.delete_lora("styleA") is True
    assert loras.list_loras() == []
    assert loras.delete_lora("styleA") is False   # already gone


def test_delete_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(loras, "IMAGE_MODELS_DIR", str(tmp_path))
    for bad in ("../secrets", "a/b", "..\\x"):
        with pytest.raises(ValueError):
            loras.delete_lora(bad)


def test_download_is_atomic(tmp_path, monkeypatch):
    monkeypatch.setattr(loras, "IMAGE_MODELS_DIR", str(tmp_path))
    @contextmanager
    def fake_stream(url, headers):
        yield 3, iter([b"ab", b"c"])
    res = loras.download_to_loras("http://x/f", "myLora", http_stream=fake_stream)
    assert res == {"name": "myLora", "filename": "myLora.safetensors", "size": 3}
    files = os.listdir(loras.loras_dir())
    assert files == ["myLora.safetensors"]         # no leftover .part
