"""Local-diffusion generation failures must be actionable, not dead ends.

sd-server answers a mid-generation OOM with an opaque 500 ("generate_image
returned no results"); on small GPUs that's almost always the compute buffer
at the requested resolution (verified live: FLUX.1 12B OOMs at 1024x1024 on
a 6GB card, works at 512x512)."""
from src.ai_interaction import _local_diffusion_size_hint


def test_hint_points_at_size_for_local_500_at_large_size():
    hint = _local_diffusion_size_hint(True, 500, "1024x1024")
    assert "size" in hint.lower()
    assert "512" in hint


def test_hint_mentions_log_when_already_small():
    hint = _local_diffusion_size_hint(True, 500, "512x512")
    assert "sd-server.log" in hint


def test_no_hint_for_cloud_models_or_other_codes():
    assert _local_diffusion_size_hint(False, 500, "1024x1024") == ""
    assert _local_diffusion_size_hint(True, 400, "1024x1024") == ""
