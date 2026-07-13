"""GGUF header architecture sniffing (src/gguf_meta.py).

The LLM and image-model cards both list *.gguf files; general.architecture
is the only reliable way to route a file to the right runtime (llama-server
vs sd-server). These tests craft minimal valid GGUF v3 headers.
"""
import struct

import src.gguf_meta as gm


def _kv_string(key: str, value: str) -> bytes:
    kb, vb = key.encode(), value.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb


def _kv_u32(key: str, value: int) -> bytes:
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_u32_array(key: str, values) -> bytes:
    kb = key.encode()
    body = struct.pack("<I", 4) + struct.pack("<Q", len(values)) + b"".join(struct.pack("<I", v) for v in values)
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 9) + body


def _gguf(kvs: list) -> bytes:
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs)) + b"".join(kvs)


def test_reads_architecture(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(_gguf([_kv_string("general.architecture", "flux")]))
    assert gm.read_gguf_architecture(str(p)) == "flux"


def test_skips_earlier_keys_including_arrays(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(_gguf([
        _kv_u32("general.file_type", 7),
        _kv_u32_array("some.array", [1, 2, 3]),
        _kv_string("general.name", "llama x"),
        _kv_string("general.architecture", "llama"),
    ]))
    assert gm.read_gguf_architecture(str(p)) == "llama"


def test_not_gguf_returns_none(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"x" * 64)
    assert gm.read_gguf_architecture(str(p)) is None


def test_truncated_returns_none(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(_gguf([_kv_string("general.architecture", "flux")])[:20])
    assert gm.read_gguf_architecture(str(p)) is None


def test_missing_file_returns_none(tmp_path):
    assert gm.read_gguf_architecture(str(tmp_path / "nope.gguf")) is None


def _tensor_info(name: str) -> bytes:
    nb = name.encode()
    return (struct.pack("<Q", len(nb)) + nb + struct.pack("<I", 1) +
            struct.pack("<Q", 16) + struct.pack("<I", 0) + struct.pack("<Q", 0))


def _gguf_tensors(names, kvs=None) -> bytes:
    kvs = kvs or []
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(names)) +
            struct.pack("<Q", len(kvs)) + b"".join(kvs) +
            b"".join(_tensor_info(n) for n in names))


def test_read_tensor_names(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(_gguf_tensors(["a.weight", "b.bias"]))
    assert gm.read_gguf_tensor_names(str(p)) == ["a.weight", "b.bias"]


def test_read_tensor_names_skips_metadata_kv(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(_gguf_tensors(
        ["x.weight"],
        kvs=[_kv_u32("general.file_type", 7), _kv_string("general.architecture", "sdxl")]))
    assert gm.read_gguf_tensor_names(str(p)) == ["x.weight"]


def test_full_checkpoint_detected_by_embedded_text_encoder(tmp_path):
    # An all-in-one SDXL checkpoint embeds its text encoder + VAE.
    p = tmp_path / "sdxl.gguf"
    p.write_bytes(_gguf_tensors([
        "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
        "model.diffusion_model.time_embed.0.weight",
        "first_stage_model.decoder.norm_out.weight",
    ]))
    assert gm.gguf_is_full_checkpoint(str(p))


def test_bare_diffusion_gguf_is_not_full_checkpoint(tmp_path):
    # FLUX/klein/chroma ship diffusion tensors ONLY (encoders are external).
    p = tmp_path / "flux.gguf"
    p.write_bytes(_gguf_tensors(
        ["model.diffusion_model.double_blocks.0.img_attn.qkv.weight",
         "model.diffusion_model.final_layer.linear.weight"],
        kvs=[_kv_string("general.architecture", "flux")]))
    assert not gm.gguf_is_full_checkpoint(str(p))


def test_is_image_architecture():
    assert gm.is_image_architecture("flux")
    assert gm.is_image_architecture("flux2")
    assert gm.is_image_architecture("sdxl")
    assert gm.is_image_architecture("lumina2")  # Z-Image (turbo)
    assert not gm.is_image_architecture("llama")
    assert not gm.is_image_architecture(None)


def test_is_llm_servable():
    # Unknown/unparseable stays LLM-servable (status quo for odd files)...
    assert gm.is_llm_servable(None)
    assert gm.is_llm_servable("llama")
    assert gm.is_llm_servable("qwen3moe")
    # ...but known diffusion + standalone-encoder architectures are not.
    assert not gm.is_llm_servable("flux")
    assert not gm.is_llm_servable("t5encoder")
    assert not gm.is_llm_servable("clip")
