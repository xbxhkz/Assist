import pytest
import src.imagemodels.serve_resolve as sr
from src.imagemodels.encoders import MissingEncoderError


@pytest.fixture
def patched(monkeypatch):
    calls = {}
    monkeypatch.setattr(sr, "gguf_is_full_checkpoint", lambda p: calls.get("checkpoint", False))
    monkeypatch.setattr(sr, "read_gguf_architecture", lambda p: calls.get("arch"))
    monkeypatch.setattr(sr, "looks_like_flux2", lambda p: calls.get("flux2", False))
    monkeypatch.setattr(sr, "looks_like_chroma", lambda p: calls.get("chroma", False))
    monkeypatch.setattr(sr, "resolve_flux_files", lambda p, **k: {"kind": "flux", **k})
    monkeypatch.setattr(sr, "resolve_flux2_files", lambda p, **k: {"kind": "flux2", **k})
    monkeypatch.setattr(sr, "resolve_chroma_files", lambda p, **k: {"kind": "chroma", **k})
    monkeypatch.setattr(sr, "resolve_zimage_files", lambda p, **k: {"kind": "zimage", **k})
    return calls


def test_all_in_one_checkpoint(patched):
    patched["checkpoint"] = True
    out = sr.resolve_image_files("C:/m/sdxl.gguf")
    assert set(out) == {"checkpoint"}


def test_zimage_by_arch(patched):
    patched["arch"] = "lumina2"
    assert sr.resolve_image_files("C:/m/z.gguf")["kind"] == "zimage"


def test_flux2(patched):
    patched["flux2"] = True
    assert sr.resolve_image_files("C:/m/klein.gguf")["kind"] == "flux2"


def test_chroma_by_arch(patched):
    patched["arch"] = "chroma"
    assert sr.resolve_image_files("C:/m/chroma.gguf")["kind"] == "chroma"


def test_default_flux(patched):
    # nothing special detected → FLUX.1
    assert sr.resolve_image_files("C:/m/flux.gguf")["kind"] == "flux"


def test_overrides_forwarded(patched):
    out = sr.resolve_image_files("C:/m/flux.gguf", t5xxl="T", clip_l="C", vae="V")
    assert out["t5xxl"] == "T" and out["clip_l"] == "C" and out["vae"] == "V"


def test_missing_encoder_gets_hint(patched, monkeypatch):
    # override the fixture's working resolver with one that raises
    def boom(p, **k):
        raise MissingEncoderError(["t5xxl", "vae"])
    monkeypatch.setattr(sr, "resolve_flux_files", boom)
    with pytest.raises(MissingEncoderError) as ei:
        sr.resolve_image_files("C:/m/flux.gguf")
    assert "t5xxl" in ei.value.hint and "FLUX" in ei.value.hint
