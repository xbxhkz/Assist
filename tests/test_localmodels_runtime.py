"""Unit tests for the pure native-local-model runtime helpers."""
import os

import src.localmodels.runtime as rt


def test_local_endpoint_url():
    assert rt.local_endpoint_url(8123) == "http://127.0.0.1:8123/v1"


def test_build_serve_argv_has_model_host_port():
    argv = rt.build_serve_argv("/x/llama-server", "/m/model.gguf", 8123)
    assert argv[0] == "/x/llama-server"
    assert "--model" in argv and "/m/model.gguf" in argv
    assert "--host" in argv and "127.0.0.1" in argv
    assert "--port" in argv and "8123" in argv
    assert "--ctx-size" in argv and "4096" in argv


def test_build_serve_argv_aliases_to_basename():
    """The served model must be advertised under its .gguf filename, not the
    full path, so the model picker shows a clean, recognizable name."""
    argv = rt.build_serve_argv(
        "/x/llama-server",
        r"C:\Users\Admin\.assist\data\models\my-model.gguf", 8123)
    assert "--alias" in argv
    assert argv[argv.index("--alias") + 1] == "my-model.gguf"


def test_resolve_prefers_path_binary():
    got = rt.resolve_llama_binary(path_lookup=lambda n: "/usr/bin/llama-server"
                                  if n == "llama-server" else None)
    assert got == "/usr/bin/llama-server"


def test_resolve_uses_bundled_when_no_path(tmp_path):
    bundle = tmp_path / "llama"
    bundle.mkdir()
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    (bundle / name).write_text("stub")
    got = rt.resolve_llama_binary(path_lookup=lambda n: None,
                                  frozen_base=str(tmp_path))
    assert got == str(bundle / name)


def test_resolve_raises_when_nothing_found(tmp_path):
    import pytest
    with pytest.raises(RuntimeError):
        rt.resolve_llama_binary(path_lookup=lambda n: None,
                                frozen_base=str(tmp_path),
                                dev_base=str(tmp_path / "nope"))


def test_list_gguf_models_filters_and_reports_size(tmp_path):
    (tmp_path / "a.gguf").write_bytes(b"xxxx")
    (tmp_path / "b.txt").write_text("no")
    models = rt.list_gguf_models(str(tmp_path))
    assert [m["name"] for m in models] == ["a.gguf"]
    assert models[0]["size"] == 4
    assert models[0]["path"] == str(tmp_path / "a.gguf")


def test_list_gguf_models_missing_dir_is_empty():
    assert rt.list_gguf_models("/no/such/dir") == []


def _gguf_with_arch(arch: str) -> bytes:
    import struct
    kb, vb = b"general.architecture", arch.encode()
    kv = struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1) + kv


def test_list_gguf_models_excludes_image_architectures(tmp_path):
    """A FLUX (diffusion) .gguf downloaded into the models dir must not be
    offered to llama-server — it fails at load with 'unknown architecture'."""
    (tmp_path / "flux1-dev.gguf").write_bytes(_gguf_with_arch("flux"))
    (tmp_path / "chat.gguf").write_bytes(_gguf_with_arch("llama"))
    (tmp_path / "weird.gguf").write_bytes(b"not-a-gguf")  # unknown arch stays listed
    models = rt.list_gguf_models(str(tmp_path))
    assert [m["name"] for m in models] == ["chat.gguf", "weird.gguf"]
