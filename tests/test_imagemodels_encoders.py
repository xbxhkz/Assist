import pytest

import src.imagemodels.encoders as enc


def test_resolves_from_sibling(tmp_path, monkeypatch):
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(tmp_path / "img"))
    d = tmp_path / "flux"; d.mkdir()
    for f in ["flux.gguf", "t5xxl.gguf", "clip_l.safetensors", "ae.safetensors"]:
        (d / f).write_bytes(b"x")
    got = enc.resolve_flux_files(str(d / "flux.gguf"))
    assert got["t5xxl"].endswith("t5xxl.gguf")
    assert got["clip_l"].endswith("clip_l.safetensors")
    assert got["vae"].endswith("ae.safetensors")


def test_explicit_paths_win(tmp_path, monkeypatch):
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(tmp_path / "img"))
    d = tmp_path / "flux"; d.mkdir()
    (d / "flux.gguf").write_bytes(b"x")
    t5 = d / "my_t5.gguf"; t5.write_bytes(b"x")
    clip = d / "c.safetensors"; clip.write_bytes(b"x")
    vae = d / "v.safetensors"; vae.write_bytes(b"x")
    got = enc.resolve_flux_files(str(d / "flux.gguf"), t5xxl=str(t5), clip_l=str(clip), vae=str(vae))
    assert got["t5xxl"] == str(t5).replace("\\", "/") or got["t5xxl"].endswith("my_t5.gguf")


def test_resolves_from_shared_encoders_dir(tmp_path, monkeypatch):
    img = tmp_path / "img"; (img / "encoders").mkdir(parents=True)
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(img))
    for f in ["t5xxl.gguf", "clip_l.safetensors", "ae.safetensors"]:
        (img / "encoders" / f).write_bytes(b"x")
    d = tmp_path / "flux"; d.mkdir()
    (d / "flux.gguf").write_bytes(b"x")   # only the diffusion model beside it
    got = enc.resolve_flux_files(str(d / "flux.gguf"))
    assert "encoders" in got["t5xxl"]


def test_missing_raises_named(tmp_path, monkeypatch):
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(tmp_path / "img"))
    d = tmp_path / "flux"; d.mkdir()
    (d / "flux.gguf").write_bytes(b"x")
    with pytest.raises(enc.MissingEncoderError) as ei:
        enc.resolve_flux_files(str(d / "flux.gguf"))
    assert "t5xxl" in ei.value.missing and "vae" in ei.value.missing
