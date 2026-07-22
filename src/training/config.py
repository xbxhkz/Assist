"""Training-run config, validation, and a VRAM-fit estimate. Pure (no I/O)."""
import re
from dataclasses import dataclass, asdict
from typing import Optional

_FIXED_OVERHEAD_GB = 1.0   # CUDA context + compute buffers (empirical)
_PER_B_GB = 0.8            # 4-bit weights + LoRA + activations + optimizer, per 1B params


@dataclass
class TrainingConfig:
    base_model: str
    dataset_path: str
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    steps: Optional[int] = 100
    epochs: Optional[float] = None
    batch_size: int = 1
    learning_rate: float = 2e-4
    max_seq_length: int = 512

    def validate(self) -> list:
        errs = []
        if not isinstance(self.base_model, str) or not self.base_model.strip():
            errs.append("base_model is required")
        if not isinstance(self.dataset_path, str) or not self.dataset_path.strip():
            errs.append("dataset_path is required")
        for name in ("lora_r", "lora_alpha", "batch_size"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                errs.append(f"{name} must be a positive integer")
        if not (isinstance(self.lora_dropout, (int, float)) and 0 <= self.lora_dropout < 1):
            errs.append("lora_dropout must be in [0, 1)")
        if not (isinstance(self.learning_rate, (int, float)) and self.learning_rate > 0):
            errs.append("learning_rate must be > 0")
        has_steps = isinstance(self.steps, int) and not isinstance(self.steps, bool) and self.steps > 0
        has_epochs = isinstance(self.epochs, (int, float)) and not isinstance(self.epochs, bool) and self.epochs > 0
        if has_steps == has_epochs:            # need exactly one
            errs.append("set exactly one of steps or epochs")
        return errs

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_vram_gb(params_b: float) -> float:
    """Rough peak VRAM (GiB) for a QLoRA 4-bit run. Calibrated to the spike
    (0.5B -> ~1.16 GB measured; this returns a slightly conservative estimate)."""
    try:
        pb = float(params_b)
    except (TypeError, ValueError):
        return _FIXED_OVERHEAD_GB
    return round(_FIXED_OVERHEAD_GB + _PER_B_GB * max(pb, 0.0), 2)


def fit_level(params_b, free_gib) -> str:
    """'fits' | 'tight' | 'too_big' | 'unknown' for a QLoRA run of `params_b`
    on `free_gib` free VRAM."""
    if free_gib is None or params_b is None:
        return "unknown"
    est = estimate_vram_gb(params_b)
    if est <= 0.8 * free_gib:
        return "fits"
    if est <= free_gib:
        return "tight"
    return "too_big"


def parse_params_b(model_id) -> Optional[float]:
    """Extract the parameter count in billions from a model name, e.g.
    '...-0.5B-...' -> 0.5, '...-8B' -> 8.0. None when absent."""
    if not isinstance(model_id, str):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_id.replace("_", "-"))
    return float(m.group(1)) if m else None
