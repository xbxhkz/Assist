"""Image-LoRA training-run config + validation. Pure (no I/O). Mirrors
src/training/config.py's TrainingConfig, scoped to the single family/
toolchain the feasibility spike proved: SDXL via diffusers, precompute-
then-offload, rank-4 LoRA, gradient checkpointing, 8-bit AdamW (see
docs/superpowers/specs/2026-07-31-image-lora-training-engine-design.md)."""
from dataclasses import dataclass, asdict

SUPPORTED_BASE_MODELS = {"stabilityai/stable-diffusion-xl-base-1.0"}


@dataclass
class ImageTrainingConfig:
    dataset_name: str
    output_name: str
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    rank: int = 4
    lora_alpha: int = 4
    learning_rate: float = 1e-4
    steps: int = 1000
    resolution: int = 1024

    def validate(self) -> list:
        errs = []
        if not isinstance(self.dataset_name, str) or not self.dataset_name.strip():
            errs.append("dataset_name is required")
        if not isinstance(self.output_name, str) or not self.output_name.strip():
            errs.append("output_name is required")
        if not isinstance(self.base_model, str) or not self.base_model.strip():
            errs.append("base_model is required")
        elif self.base_model not in SUPPORTED_BASE_MODELS:
            errs.append(f"base_model '{self.base_model}' is not supported on this hardware "
                       f"(only SDXL base is proven feasible; supported: {sorted(SUPPORTED_BASE_MODELS)})")
        for name in ("rank", "lora_alpha", "steps", "resolution"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                errs.append(f"{name} must be a positive integer")
        if not (isinstance(self.learning_rate, (int, float)) and self.learning_rate > 0):
            errs.append("learning_rate must be > 0")
        return errs

    def to_dict(self) -> dict:
        return asdict(self)
