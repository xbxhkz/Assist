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


# ── FLUX.2 (klein): --llm text encoder + flux2 VAE ──────────────────────

def test_flux2_resolves_from_shared_encoders_dir(tmp_path, monkeypatch):
    img = tmp_path / "img"; (img / "encoders").mkdir(parents=True)
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(img))
    (img / "encoders" / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"x")
    (img / "encoders" / "flux2_ae.safetensors").write_bytes(b"x")
    d = tmp_path / "m"; d.mkdir()
    (d / "flux-2-klein-4b-Q8_0.gguf").write_bytes(b"x")
    got = enc.resolve_flux2_files(str(d / "flux-2-klein-4b-Q8_0.gguf"))
    assert got["llm"].endswith("Qwen3-4B-Q4_K_M.gguf")
    assert got["vae"].endswith("flux2_ae.safetensors")
    assert got["diffusion_model"].endswith("flux-2-klein-4b-Q8_0.gguf")


def test_flux2_llm_matches_quant_variants_case_insensitively(tmp_path, monkeypatch):
    """Encoder ggufs come in many quant suffixes; match by qwen3/mistral
    prefix pattern, not an exact-name list."""
    img = tmp_path / "img"; (img / "encoders").mkdir(parents=True)
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(img))
    (img / "encoders" / "qwen3-4b-q8_0.gguf").write_bytes(b"x")
    (img / "encoders" / "flux2_ae.safetensors").write_bytes(b"x")
    d = tmp_path / "m"; d.mkdir()
    (d / "flux2.gguf").write_bytes(b"x")
    got = enc.resolve_flux2_files(str(d / "flux2.gguf"))
    assert got["llm"].endswith("qwen3-4b-q8_0.gguf")


def test_flux2_does_not_take_flux1_vae(tmp_path, monkeypatch):
    """FLUX.1's ae.safetensors shares the encoders dir but is a different
    autoencoder — the flux2 resolver must not silently pick it up."""
    img = tmp_path / "img"; (img / "encoders").mkdir(parents=True)
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(img))
    (img / "encoders" / "ae.safetensors").write_bytes(b"x")        # FLUX.1 VAE
    (img / "encoders" / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"x")
    d = tmp_path / "m"; d.mkdir()
    (d / "flux2.gguf").write_bytes(b"x")
    with pytest.raises(enc.MissingEncoderError) as ei:
        enc.resolve_flux2_files(str(d / "flux2.gguf"))
    assert ei.value.missing == ["vae"]


def test_find_taesd_in_encoders_dir(tmp_path, monkeypatch):
    img = tmp_path / "img"; (img / "encoders").mkdir(parents=True)
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(img))
    (img / "encoders" / "taef1.safetensors").write_bytes(b"x")
    d = tmp_path / "m"; d.mkdir()
    (d / "flux1-dev.gguf").write_bytes(b"x")
    assert enc.find_taesd(str(d / "flux1-dev.gguf")).endswith("taef1.safetensors")


def test_find_taesd_none_for_flux2(tmp_path, monkeypatch):
    """taef1 is the FLUX.1 tiny autoencoder; using it on FLUX.2 latents would
    misdecode, so the finder must refuse flux2-named models."""
    img = tmp_path / "img"; (img / "encoders").mkdir(parents=True)
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(img))
    (img / "encoders" / "taef1.safetensors").write_bytes(b"x")
    d = tmp_path / "m"; d.mkdir()
    (d / "flux-2-klein.gguf").write_bytes(b"x")
    assert enc.find_taesd(str(d / "flux-2-klein.gguf")) is None


def test_find_taesd_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(tmp_path / "img"))
    d = tmp_path / "m"; d.mkdir()
    (d / "flux1-dev.gguf").write_bytes(b"x")
    assert enc.find_taesd(str(d / "flux1-dev.gguf")) is None


def test_flux2_9b_prefers_qwen3_8b_encoder(tmp_path, monkeypatch):
    """klein-9B pairs with the Qwen3-8B encoder; picking the 4B one silently
    degrades conditioning, so model size must drive the preference."""
    img = tmp_path / "img"; (img / "encoders").mkdir(parents=True)
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(img))
    (img / "encoders" / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"x")
    (img / "encoders" / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"x")
    (img / "encoders" / "flux2_ae.safetensors").write_bytes(b"x")
    d = tmp_path / "m"; d.mkdir()
    (d / "flux-2-klein-9b-Q4_K_M.gguf").write_bytes(b"x")
    got = enc.resolve_flux2_files(str(d / "flux-2-klein-9b-Q4_K_M.gguf"))
    assert got["llm"].endswith("Qwen3-8B-Q4_K_M.gguf")
    # ...and the 4B model keeps preferring the 4B encoder.
    (d / "flux-2-klein-4b-Q8_0.gguf").write_bytes(b"x")
    got4 = enc.resolve_flux2_files(str(d / "flux-2-klein-4b-Q8_0.gguf"))
    assert got4["llm"].endswith("Qwen3-4B-Q4_K_M.gguf")


def test_zimage_resolves_qwen3_llm_and_flux1_vae(tmp_path, monkeypatch):
    """Z-Image (lumina2): Qwen3 text encoder + the FLUX.1 ae VAE (per sd.cpp
    docs/z_image.md) — both already shipped for FLUX.1 users."""
    img = tmp_path / "img"; (img / "encoders").mkdir(parents=True)
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(img))
    (img / "encoders" / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"x")
    (img / "encoders" / "ae.safetensors").write_bytes(b"x")
    d = tmp_path / "m"; d.mkdir()
    (d / "z-image-turbo-Q5_0.gguf").write_bytes(b"x")
    got = enc.resolve_zimage_files(str(d / "z-image-turbo-Q5_0.gguf"))
    assert got["llm"].endswith("Qwen3-4B-Q4_K_M.gguf")
    assert got["vae"].endswith("ae.safetensors")


def test_zimage_missing_raises_named(tmp_path, monkeypatch):
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(tmp_path / "img"))
    d = tmp_path / "m"; d.mkdir()
    (d / "z-image-turbo-Q5_0.gguf").write_bytes(b"x")
    with pytest.raises(enc.MissingEncoderError) as ei:
        enc.resolve_zimage_files(str(d / "z-image-turbo-Q5_0.gguf"))
    assert set(ei.value.missing) == {"llm", "vae"}


def test_flux2_explicit_paths_win(tmp_path, monkeypatch):
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(tmp_path / "img"))
    d = tmp_path / "m"; d.mkdir()
    (d / "flux2.gguf").write_bytes(b"x")
    llm = d / "my_llm.gguf"; llm.write_bytes(b"x")
    vae = d / "my_vae.safetensors"; vae.write_bytes(b"x")
    got = enc.resolve_flux2_files(str(d / "flux2.gguf"), llm=str(llm), vae=str(vae))
    assert got["llm"].endswith("my_llm.gguf")
    assert got["vae"].endswith("my_vae.safetensors")
