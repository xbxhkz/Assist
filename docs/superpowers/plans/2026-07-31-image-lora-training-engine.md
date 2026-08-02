# Image AI Studio — LoRA Training Engine (Sub-project 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A headless engine that trains an SDXL image-LoRA from a dataset produced by the shipped Dataset Prep tool, on the user's 6GB RTX 4050, and writes the finished `.safetensors` straight into the existing LoRA registry `sd-server` already resolves.

**Architecture:** Extend the existing Python 3.11 CUDA sidecar venv (`<DATA_DIR>/training/venv`) with `diffusers`; a thin `ImageTrainingManager` spawns a training script in that venv as a subprocess and streams line-delimited JSON progress back, mirroring the text-LoRA trainer's `TrainingEnv`/`TrainingManager`/`training_sidecar` shape exactly. The training script itself implements the precompute-then-offload technique the feasibility spike proved fits this hardware.

**Tech Stack:** diffusers + peft + bitsandbytes (already in the sidecar venv) + torch, orchestrated from FastAPI via `subprocess.Popen`, output consumed by the existing `src/imagemodels/loras.py` registry.

## Global Constraints

- **Feasibility spike is DONE (do not re-run it as a task).** Run live against real hardware on 2026-07-31: SDXL via diffusers + precompute-then-offload peaked at **5.87GB @ native 1024px, ~32s/step** (feasible, under the 6.44GB card). FLUX.1-schnell via the same technique peaked at **23.8GB just to hold the transformer backbone resident** (before LoRA/optimizer/activations) — **empirically confirmed unsupported** on this hardware. This plan builds **SDXL only**; FLUX is out of scope, not attempted, not stubbed.
  - **Superseded by the final-review fix wave (2026-08-02):** the shipped script trains LoRA params as fp32 master weights under fp16 autocast + GradScaler (not pure fp16) — a correctness fix for gradient underflow, verified live (40 real steps, 5 images: zero steps with fully-zeroed gradients, final weights non-zero and finite). This raised the measured peak to **6.04GB @ native 1024px, ~20.7s/step** (still feasible, still under the 6.44GB card, tighter headroom than the original number). Treat **6.04GB** as the current proven peak.
- **Reuse the existing sidecar venv** (`<DATA_DIR>/training/venv`) — do not create a second CUDA venv. Only `diffusers` needs adding; `torch`/`peft`/`bitsandbytes`/`accelerate` are already installed there by the text-LoRA trainer.
- **No kohya-ss.** diffusers alone proved sufficient for the one family in scope — nothing from kohya-ss is vendored.
- **Proven training config (use these exact values, not placeholders):** `rank=4`, `lora_alpha=4`, `target_modules=["to_k","to_q","to_v","to_out.0"]`, gradient checkpointing enabled on the UNet, 8-bit AdamW via `bitsandbytes`, base model `stabilityai/stable-diffusion-xl-base-1.0`, resolution `1024` (native — do not downscale). Precompute order matters: VAE + both SDXL text encoders stay resident and encode every image/caption pair *once*, then move to CPU before the UNet is loaded, so peak VRAM never includes both stages at once.
- **Output is a `.safetensors` file placed directly in `src/imagemodels/loras.py`'s `loras_dir()`** — no conversion step (unlike text-LoRA training's GGUF conversion). `sd-server` already resolves `<lora:name:weight>` tags against that directory.
- **Progress channel is line-delimited JSON on stdout, with library progress bars disabled** — the exact cross-seam bug the text-LoRA trainer's whole-branch review caught (a default tqdm bar's `\r` redraws corrupting the JSON channel). Call `huggingface_hub.utils.disable_progress_bars()` before any heavy import in the sidecar script.
- **The manager never raises** — `start`/`status`/`stop`/`env_status`/`setup_env` all return status dicts, mirroring `src/training/manager.py`'s `TrainingManager` contract exactly.
- **No GUI this sub-project** — a later sub-project, matching the text-LoRA training arc (engine → GUI → serve).
- **pytest runs with `--import-mode=importlib`** (project convention).
- **Main app (Python 3.14) must never import torch/diffusers/peft/bitsandbytes/transformers** — all heavy imports live only inside `image_training_sidecar/train_sdxl_lora.py`, which is never imported, only spawned as a subprocess.
- Stage specific files when committing; never `git add -A`. Do not stage `installer/Output/Assist-Setup.exe`. Commits end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Config

**Files:**
- Create: `src/image_training/__init__.py` (empty)
- Create: `src/image_training/config.py`
- Test: `tests/test_image_training_config.py`

**Interfaces:**
- Produces: `ImageTrainingConfig` dataclass with fields `dataset_name: str`, `output_name: str`, `base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"`, `rank: int = 4`, `lora_alpha: int = 4`, `learning_rate: float = 1e-4`, `steps: int = 1000`, `resolution: int = 1024`; methods `.validate() -> list` (error strings, empty list means valid) and `.to_dict() -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_image_training_config.py
from src.image_training.config import ImageTrainingConfig


def _valid():
    return ImageTrainingConfig(dataset_name="ds1", output_name="my-lora")


def test_validate_accepts_defaults_plus_required():
    assert _valid().validate() == []


def test_validate_rejects_missing_dataset_name():
    cfg = ImageTrainingConfig(dataset_name="", output_name="my-lora")
    errs = cfg.validate()
    assert any("dataset_name" in e for e in errs)


def test_validate_rejects_missing_output_name():
    cfg = ImageTrainingConfig(dataset_name="ds1", output_name="")
    errs = cfg.validate()
    assert any("output_name" in e for e in errs)


def test_validate_rejects_non_positive_rank():
    cfg = ImageTrainingConfig(dataset_name="ds1", output_name="my-lora", rank=0)
    errs = cfg.validate()
    assert any("rank" in e for e in errs)


def test_validate_rejects_bad_learning_rate():
    cfg = ImageTrainingConfig(dataset_name="ds1", output_name="my-lora", learning_rate=0)
    errs = cfg.validate()
    assert any("learning_rate" in e for e in errs)


def test_validate_rejects_non_positive_steps_and_resolution():
    cfg = ImageTrainingConfig(dataset_name="ds1", output_name="my-lora", steps=0, resolution=-1)
    errs = cfg.validate()
    assert any("steps" in e for e in errs)
    assert any("resolution" in e for e in errs)


def test_to_dict_roundtrip():
    cfg = _valid()
    d = cfg.to_dict()
    assert d["dataset_name"] == "ds1" and d["output_name"] == "my-lora"
    assert d["rank"] == 4 and d["lora_alpha"] == 4 and d["resolution"] == 1024
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_image_training_config.py -v --import-mode=importlib`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.image_training'`

- [ ] **Step 3: Create the package and write the config**

Create `src/image_training/__init__.py` (empty file).

```python
# src/image_training/config.py
"""Image-LoRA training-run config + validation. Pure (no I/O). Mirrors
src/training/config.py's TrainingConfig, scoped to the single family/
toolchain the feasibility spike proved: SDXL via diffusers, precompute-
then-offload, rank-4 LoRA, gradient checkpointing, 8-bit AdamW (see
docs/superpowers/specs/2026-07-31-image-lora-training-engine-design.md)."""
from dataclasses import dataclass, asdict


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
        for name in ("rank", "lora_alpha", "steps", "resolution"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                errs.append(f"{name} must be a positive integer")
        if not (isinstance(self.learning_rate, (int, float)) and self.learning_rate > 0):
            errs.append("learning_rate must be > 0")
        return errs

    def to_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_image_training_config.py -v --import-mode=importlib`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/image_training/__init__.py src/image_training/config.py tests/test_image_training_config.py
git commit -m "feat(image-training): add ImageTrainingConfig"
```

---

### Task 2: Sidecar venv extension (diffusers)

**Files:**
- Create: `src/image_training/env.py`
- Test: `tests/test_image_training_env.py`

**Interfaces:**
- Consumes: `src.training.env.TrainingEnv` — `.ensure_ready(progress=None) -> {"ready": bool, "error": str|None}`, `.venv_python() -> str`, `.status() -> "ready"|"not_installed"` (existing, unchanged).
- Produces: `ImageTrainingEnv(training_env=None, uv_binary=None, run=None)` with `.venv_python() -> str`, `.status() -> "ready"|"not_installed"`, `.ensure_ready(progress=None) -> {"ready": bool, "error": str|None}` (never raises). Later tasks call `ImageTrainingEnv().ensure_ready()` and `.venv_python()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_image_training_env.py
import os
from src.image_training.env import ImageTrainingEnv


class FakeBaseEnv:
    def __init__(self, ready=True, venv_dir=None):
        self._ready = ready
        self._venv_dir = venv_dir

    def ensure_ready(self, progress=None):
        return {"ready": self._ready, "error": None if self._ready else "base env not ready"}

    def venv_python(self):
        return os.path.join(self._venv_dir, "Scripts", "python.exe")

    def status(self):
        return "ready" if self._ready else "not_installed"


def test_status_not_installed_when_base_env_not_ready(tmp_path):
    env = ImageTrainingEnv(training_env=FakeBaseEnv(ready=False, venv_dir=str(tmp_path)))
    assert env.status() == "not_installed"


def test_status_not_installed_when_base_ready_but_diffusers_marker_missing(tmp_path):
    env = ImageTrainingEnv(training_env=FakeBaseEnv(ready=True, venv_dir=str(tmp_path)))
    assert env.status() == "not_installed"


def test_ensure_ready_installs_diffusers_and_marks_ready(tmp_path):
    calls = []

    def fake_run(argv):
        calls.append(argv)
        return (0, "ok")

    env = ImageTrainingEnv(training_env=FakeBaseEnv(ready=True, venv_dir=str(tmp_path)),
                           uv_binary="uv.exe", run=fake_run)
    out = env.ensure_ready()
    assert out == {"ready": True, "error": None}
    assert calls and calls[0][:2] == ["uv.exe", "pip"] and "diffusers" in calls[0]
    assert env.status() == "ready"


def test_ensure_ready_idempotent_skips_when_already_ready(tmp_path):
    env = ImageTrainingEnv(training_env=FakeBaseEnv(ready=True, venv_dir=str(tmp_path)),
                           uv_binary="uv.exe", run=lambda argv: (0, ""))
    env.ensure_ready()
    calls = []
    env2 = ImageTrainingEnv(training_env=FakeBaseEnv(ready=True, venv_dir=str(tmp_path)),
                            uv_binary="uv.exe", run=lambda argv: (calls.append(argv), (0, ""))[1])
    out = env2.ensure_ready()
    assert out == {"ready": True, "error": None}
    assert calls == []


def test_ensure_ready_propagates_base_env_error(tmp_path):
    env = ImageTrainingEnv(training_env=FakeBaseEnv(ready=False, venv_dir=str(tmp_path)),
                           uv_binary="uv.exe", run=lambda argv: (0, ""))
    out = env.ensure_ready()
    assert out["ready"] is False and "base env not ready" in out["error"]


def test_ensure_ready_never_raises_on_install_failure(tmp_path):
    env = ImageTrainingEnv(training_env=FakeBaseEnv(ready=True, venv_dir=str(tmp_path)),
                           uv_binary="uv.exe", run=lambda argv: (1, "boom"))
    out = env.ensure_ready()
    assert out["ready"] is False and "boom" in out["error"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_image_training_env.py -v --import-mode=importlib`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.image_training.env'`

- [ ] **Step 3: Write the implementation**

```python
# src/image_training/env.py
"""On-demand extension of the existing training sidecar venv with
`diffusers`, for SDXL image-LoRA training. Reuses the SAME Py3.11 CUDA
venv the text-LoRA trainer provisions (torch/peft/bitsandbytes/accelerate
already installed there) instead of standing up a second multi-gigabyte
CUDA venv -- diffusers is the only package this needs on top. Never
raises."""
import os

STACK = ["diffusers"]


def _default_run(argv):
    import subprocess
    p = subprocess.run(argv, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return p.returncode, (p.stdout or "") + (p.stderr or "")


class ImageTrainingEnv:
    def __init__(self, training_env=None, uv_binary=None, run=None):
        if training_env is None:
            from src.training.env import TrainingEnv
            training_env = TrainingEnv()
        self._env = training_env
        self._uv = uv_binary
        self._run = run or _default_run

    def venv_python(self) -> str:
        return self._env.venv_python()

    def _marker(self) -> str:
        venv_dir = os.path.dirname(os.path.dirname(self.venv_python()))
        return os.path.join(venv_dir, ".assist_image_training_ready")

    def status(self) -> str:
        if self._env.status() != "ready":
            return "not_installed"
        return "ready" if os.path.isfile(self._marker()) else "not_installed"

    def _uv_bin(self):
        if self._uv is None:
            from src.training.runtime import resolve_uv_binary
            self._uv = resolve_uv_binary()
        return self._uv

    def ensure_ready(self, progress=None) -> dict:
        """Idempotently ensure the base training venv exists AND has
        `diffusers` installed. Returns {"ready", "error"}. Never raises."""
        base = self._env.ensure_ready(progress=progress)
        if not base.get("ready"):
            return base
        if self.status() == "ready":
            return {"ready": True, "error": None}
        try:
            uv = self._uv_bin()
            py = self.venv_python()
            if progress:
                try:
                    progress({"event": "install", "cmd": "pip diffusers"})
                except Exception:
                    pass
            rc, out = self._run([uv, "pip", "install", "--python", py] + STACK)
            if rc != 0:
                return {"ready": False, "error": f"diffusers install failed: {out[-500:]}"}
            os.makedirs(os.path.dirname(self._marker()), exist_ok=True)
            open(self._marker(), "w").close()
            return {"ready": True, "error": None}
        except Exception as e:  # noqa: BLE001
            return {"ready": False, "error": f"image training env setup failed: {e}"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_image_training_env.py -v --import-mode=importlib`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/image_training/env.py tests/test_image_training_env.py
git commit -m "feat(image-training): extend sidecar venv with diffusers"
```

---

### Task 3: Sidecar script path resolution

**Files:**
- Create: `src/image_training/runtime.py`
- Test: `tests/test_image_training_runtime.py`

**Interfaces:**
- Produces: `resolve_image_sidecar_script(frozen_base=None, dev_base=None) -> str` — raises `RuntimeError` if the script isn't found. Consumed by Task 5's manager.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_image_training_runtime.py
import pytest
from src.image_training import runtime


def test_resolve_prefers_frozen(tmp_path):
    base = tmp_path / "frozen"
    (base / "image_training_sidecar").mkdir(parents=True)
    scr = base / "image_training_sidecar" / "train_sdxl_lora.py"
    scr.write_text("x")
    assert runtime.resolve_image_sidecar_script(frozen_base=str(base)) == str(scr)


def test_resolve_falls_back_to_dev(tmp_path):
    dev = tmp_path / "image_training_sidecar"
    dev.mkdir(parents=True)
    scr = dev / "train_sdxl_lora.py"
    scr.write_text("x")
    assert runtime.resolve_image_sidecar_script(frozen_base=str(tmp_path / "none"),
                                                 dev_base=str(dev)) == str(scr)


def test_resolve_raises_when_missing(tmp_path):
    with pytest.raises(RuntimeError):
        runtime.resolve_image_sidecar_script(frozen_base=str(tmp_path / "a"),
                                              dev_base=str(tmp_path / "b"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_image_training_runtime.py -v --import-mode=importlib`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.image_training.runtime'`

- [ ] **Step 3: Write the implementation**

```python
# src/image_training/runtime.py
"""Locate the image-training sidecar script at runtime. Mirrors
src/training/runtime.py's resolve_sidecar_script frozen/dev resolution."""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _frozen_base(explicit):
    if explicit is not None:
        return explicit
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", None)
    return None


def resolve_image_sidecar_script(frozen_base=None, dev_base=None) -> str:
    """Path to image_training_sidecar/train_sdxl_lora.py. Frozen:
    <_MEIPASS>/image_training_sidecar/train_sdxl_lora.py; dev:
    <repo>/image_training_sidecar/train_sdxl_lora.py. Raises RuntimeError
    if missing."""
    base = _frozen_base(frozen_base)
    if base:
        cand = os.path.join(base, "image_training_sidecar", "train_sdxl_lora.py")
        if os.path.isfile(cand):
            return cand
    if dev_base is None:
        dev_base = os.path.join(_REPO_ROOT, "image_training_sidecar")
    cand = os.path.join(dev_base, "train_sdxl_lora.py")
    if os.path.isfile(cand):
        return cand
    raise RuntimeError("image_training_sidecar/train_sdxl_lora.py not found.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_image_training_runtime.py -v --import-mode=importlib`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/image_training/runtime.py tests/test_image_training_runtime.py
git commit -m "feat(image-training): resolve the sidecar script path"
```

---

### Task 4: SDXL LoRA training sidecar script

**Files:**
- Create: `image_training_sidecar/__init__.py` (empty)
- Create: `image_training_sidecar/train_sdxl_lora.py`
- Modify: `Assist.spec:49` (add a `datas` entry right after the existing `training_sidecar` line)
- Test: `tests/test_image_training_sidecar_syntax.py`

**Interfaces:**
- Consumes: a JSON config file with keys `images` (`[{"image": abs_path, "caption": str}, ...]`), `base_model`, `rank`, `lora_alpha`, `learning_rate`, `steps`, `resolution`, `lora_path` (final destination `.safetensors` path), `run_dir` (scratch directory for the training run) — all produced by Task 5's manager.
- Produces: line-delimited JSON on stdout — `{"event":"start","total_steps":N}`, `{"event":"step","step":N,"loss":F,"vram_gb":F}`, `{"event":"done","lora_path":str,"peak_vram_gb":F}`, `{"event":"error","message":str,"trace":str}` — read by Task 5's manager `_pump`. Writes the finished LoRA to the exact `lora_path` given in the config.

This script runs ONLY inside the Py3.11 CUDA sidecar venv and is never imported by the main app — it cannot be unit-tested with real torch/diffusers here (that stack is not installed in the app's Python 3.14 environment, and must never be). Its only test is a syntax check; its real behavior was already validated by the feasibility spike (`spike_sdxl_lora_offload.py`, run live on this hardware) which this script generalizes from one hardcoded image to the full dataset loop.

- [ ] **Step 1: Write the failing test (syntax gate)**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_image_training_sidecar_syntax.py -v --import-mode=importlib`
Expected: FAIL with `FileNotFoundError` (the script doesn't exist yet)

- [ ] **Step 3: Write the sidecar script**

Create `image_training_sidecar/__init__.py` (empty file).

```python
# image_training_sidecar/train_sdxl_lora.py
"""SDXL LoRA training sidecar. Runs INSIDE the Py3.11 CUDA venv (never
imported by the Py3.14 app). Reads a JSON config describing a saved image
dataset (image + caption pairs), trains a rank-N LoRA against the SDXL
UNet using the precompute-then-offload technique the feasibility spike
proved fits this hardware (see docs/superpowers/specs/2026-07-31-image-
lora-training-engine-design.md): every image/caption pair is VAE- and
text-encoder-encoded ONCE up front while those modules are resident, then
the text encoders + VAE are moved off the GPU entirely so only the UNet
(+ LoRA + 8-bit optimizer state) stays resident for the training loop.
Streams JSON-line progress to stdout and writes the finished LoRA
straight to the path the manager resolved in imagemodels' loras_dir()
registry -- no conversion step, unlike text-LoRA training's GGUF step.

Progress protocol (one JSON object per line on stdout):
  {"event":"start","total_steps":N}
  {"event":"step","step":N,"loss":F,"vram_gb":F}
  {"event":"done","lora_path":str,"peak_vram_gb":F}
  {"event":"error","message":str,"trace":str}
"""
import argparse
import json
import os
import random
import sys


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    # stdout is the JSON-progress channel back to the app; force UTF-8 with
    # a non-raising error handler so a non-ASCII message can never crash
    # emit() (Windows Py3.11 stdout defaults to the locale encoding).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Suppress huggingface_hub/diffusers download+loading progress bars --
        # they redraw with \r on stdout and would corrupt the JSON progress
        # channel above, exactly the cross-seam bug the text-LoRA trainer's
        # whole-branch review caught (its fix: disable_tqdm=True on Trainer;
        # this script has no Trainer, so the equivalent is disabling the
        # library-wide progress bars before anything else prints).
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()

        import torch
        from diffusers import StableDiffusionXLPipeline
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict
        import bitsandbytes as bnb
        from PIL import Image

        images = cfg["images"]  # [{"image": path, "caption": text}, ...]
        base_model = cfg["base_model"]
        rank = int(cfg.get("rank", 4))
        lora_alpha = int(cfg.get("lora_alpha", 4))
        learning_rate = float(cfg.get("learning_rate", 1e-4))
        steps = int(cfg.get("steps", 1000))
        resolution = int(cfg.get("resolution", 1024))
        lora_path = cfg["lora_path"]
        run_dir = cfg["run_dir"]
        os.makedirs(run_dir, exist_ok=True)

        device = "cuda"
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

        # --- Phase 1: precompute every image's latents + text embedding ONCE,
        # while the text encoders + VAE are still resident. ---
        pipe.vae.to(device, dtype=torch.float32)
        pipe.text_encoder.to(device, dtype=torch.float16)
        pipe.text_encoder_2.to(device, dtype=torch.float16)
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder_2.requires_grad_(False)

        examples = []
        for item in images:
            img = Image.open(item["image"]).convert("RGB")
            pixel_values = pipe.image_processor.preprocess(img, height=resolution, width=resolution)
            pixel_values = pixel_values.to(device, dtype=torch.float32)
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = (latents * pipe.vae.config.scaling_factor).to(dtype=torch.float16)
                (prompt_embeds, _, pooled_prompt_embeds, _) = pipe.encode_prompt(
                    prompt=item.get("caption") or "", device=device, num_images_per_prompt=1,
                    do_classifier_free_guidance=False)
            examples.append((latents.detach(), prompt_embeds.detach(), pooled_prompt_embeds.detach()))

        # --- Phase 2: offload text encoders + VAE, keep only the UNet resident. ---
        pipe.vae.to("cpu")
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        pipe.unet.to(device, dtype=torch.float16)
        pipe.unet.requires_grad_(False)
        pipe.unet.enable_gradient_checkpointing()

        lora_config = LoraConfig(r=rank, lora_alpha=lora_alpha,
                                 target_modules=["to_k", "to_q", "to_v", "to_out.0"])
        pipe.unet.add_adapter(lora_config)
        lora_params = [p for p in pipe.unet.parameters() if p.requires_grad]
        optimizer = bnb.optim.AdamW8bit(lora_params, lr=learning_rate)

        add_time_ids = torch.tensor(
            [[resolution, resolution, 0, 0, resolution, resolution]],
            device=device, dtype=torch.float16)

        emit({"event": "start", "total_steps": steps})
        for step in range(1, steps + 1):
            latents, prompt_embeds, pooled_prompt_embeds = random.choice(examples)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps,
                                      (1,), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
            model_pred = pipe.unet(
                noisy_latents, timesteps, encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs).sample

            loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            vram = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            emit({"event": "step", "step": step, "loss": round(float(loss.item()), 4), "vram_gb": vram})

        unet_lora_state_dict = get_peft_model_state_dict(pipe.unet)
        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=run_dir, unet_lora_layers=unet_lora_state_dict)
        produced = os.path.join(run_dir, "pytorch_lora_weights.safetensors")
        os.replace(produced, lora_path)

        peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        emit({"event": "done", "lora_path": lora_path, "peak_vram_gb": peak})
    except Exception as e:  # noqa: BLE001
        import traceback
        try:
            emit({"event": "error", "message": f"{e}", "trace": traceback.format_exc()[-1500:]})
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
```

Add the bundling entry to `Assist.spec` right after the existing `training_sidecar` line (currently line 49):

```python
    # The training sidecar script, run inside the training venv (never imported
    # by the Py3.14 app). Bundled so the frozen build can launch it.
    ('training_sidecar', 'training_sidecar'),
    # The image-training sidecar script (SDXL LoRA), run inside the SAME
    # training venv extended with diffusers. Never imported by the Py3.14 app.
    ('image_training_sidecar', 'image_training_sidecar'),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_image_training_sidecar_syntax.py -v --import-mode=importlib`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add image_training_sidecar/__init__.py image_training_sidecar/train_sdxl_lora.py Assist.spec tests/test_image_training_sidecar_syntax.py
git commit -m "feat(image-training): add SDXL LoRA training sidecar script"
```

---

### Task 5: Manager

**Files:**
- Create: `src/image_training/manager.py`
- Test: `tests/test_image_training_manager.py`

**Interfaces:**
- Consumes: `ImageTrainingConfig` (Task 1) — `.validate() -> list`, `.to_dict() -> dict`; `ImageTrainingEnv` (Task 2) — `.ensure_ready() -> dict`, `.venv_python() -> str`; `resolve_image_sidecar_script()` (Task 3); `src.image_dataset_tools.store.get_image_dataset_store()` (existing, shipped) — `.load(name) -> {"path": str, "trigger_word": str, "images": [{"filename": str, "caption": str}, ...]}` or `{"error": str}`; `src.imagemodels.loras.loras_dir() -> str` (existing, shipped).
- Produces: `ImageTrainingManager(env=None, spawn=None, dataset_store=None, runs_dir=None, loras_dir=None)` with `.start(config) -> {"started": True, "run_id": str} | {"error": str}`, `.status() -> dict` (`status`, `last_step`, `loss`, `vram_gb`, `peak_vram_gb`, `error`, `lora_path`), `.stop() -> {"stopped": bool}`, `.env_status() -> {"status": str}`, `.setup_env() -> dict`. Also `get_image_training_manager()` singleton accessor. Consumed by Task 6's routes.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_image_training_manager.py
import json
import os
from src.image_training.config import ImageTrainingConfig
from src.image_training.manager import ImageTrainingManager


class FakeProc:
    def __init__(self, lines):
        self._lines = list(lines)
        self.stdout = self

    def readline(self):
        return self._lines.pop(0) if self._lines else ""

    def poll(self):
        return None if self._lines else 0

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


class FakeEnv:
    def __init__(self, ready=True):
        self._ready = ready

    def ensure_ready(self):
        return {"ready": self._ready, "error": None if self._ready else "no env"}

    def venv_python(self):
        return "venv/python"

    def status(self):
        return "ready" if self._ready else "not_installed"


class FakeStore:
    def __init__(self, images=None, path="/ds", trigger_word="mytrigger", error=None):
        self._images = images if images is not None else [{"filename": "0000.png", "caption": "a cat"}]
        self._path = path
        self._trigger_word = trigger_word
        self._error = error

    def load(self, name):
        if self._error:
            return {"error": self._error}
        return {"name": name, "path": self._path, "trigger_word": self._trigger_word,
                "images": self._images}


def _cfg():
    return ImageTrainingConfig(dataset_name="ds1", output_name="my-lora", steps=2)


def _mgr(tmp_path, **kw):
    kw.setdefault("env", FakeEnv())
    kw.setdefault("spawn", lambda a: FakeProc([]))
    kw.setdefault("dataset_store", FakeStore())
    kw.setdefault("runs_dir", str(tmp_path / "runs"))
    kw.setdefault("loras_dir", lambda: str(tmp_path / "loras"))
    return ImageTrainingManager(**kw)


def test_start_runs_sidecar_and_tracks_progress(tmp_path):
    lines = [json.dumps({"event": "start", "total_steps": 2}) + "\n",
             json.dumps({"event": "step", "step": 1, "loss": 0.5, "vram_gb": 5.0}) + "\n",
             json.dumps({"event": "done", "lora_path": "x.safetensors", "peak_vram_gb": 5.9}) + "\n"]
    captured = {}

    def spawn(argv):
        captured["argv"] = argv
        return FakeProc(lines)

    mgr = _mgr(tmp_path, spawn=spawn)
    out = mgr.start(_cfg())
    assert out.get("started") is True
    import time
    for _ in range(200):
        if mgr.status()["status"] in ("done", "error"):
            break
        time.sleep(0.01)
    assert "--config" in captured["argv"]
    st = mgr.status()
    assert st["status"] == "done" and st["last_step"] == 1 and st["peak_vram_gb"] == 5.9


def test_start_prepends_trigger_word_to_captions_and_targets_loras_dir(tmp_path):
    captured = {}

    def spawn(argv):
        captured["argv"] = argv
        return FakeProc([])

    mgr = _mgr(tmp_path, spawn=spawn)
    mgr.start(_cfg())
    cfg_path = [a for a in captured["argv"] if a.endswith("config.json")][0]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["images"][0]["caption"] == "mytrigger, a cat"
    assert cfg["lora_path"] == os.path.join(str(tmp_path / "loras"), "my-lora.safetensors")


def test_start_rejects_invalid_config(tmp_path):
    mgr = _mgr(tmp_path)
    bad = ImageTrainingConfig(dataset_name="", output_name="", steps=1)
    out = mgr.start(bad)
    assert "error" in out


def test_start_errors_when_dataset_missing(tmp_path):
    mgr = _mgr(tmp_path, dataset_store=FakeStore(error="dataset not found"))
    out = mgr.start(_cfg())
    assert "error" in out and "dataset" in out["error"].lower()


def test_start_errors_when_dataset_has_no_images(tmp_path):
    mgr = _mgr(tmp_path, dataset_store=FakeStore(images=[]))
    out = mgr.start(_cfg())
    assert "error" in out


def test_start_errors_when_env_not_ready(tmp_path):
    mgr = _mgr(tmp_path, env=FakeEnv(ready=False))
    out = mgr.start(_cfg())
    assert "error" in out and "env" in out["error"].lower()


def test_start_never_raises_on_spawn_failure(tmp_path):
    def boom(argv):
        raise RuntimeError("cannot spawn")
    mgr = _mgr(tmp_path, spawn=boom)
    out = mgr.start(_cfg())
    assert "error" in out


def test_start_never_raises_when_dataset_store_raises(tmp_path):
    class BoomStore:
        def load(self, name):
            raise RuntimeError("disk error")
    mgr = _mgr(tmp_path, dataset_store=BoomStore())
    out = mgr.start(_cfg())
    assert "error" in out


def test_stop_when_not_running(tmp_path):
    mgr = _mgr(tmp_path)
    assert mgr.stop() == {"stopped": False}


class BlockingProc:
    """A process whose readline blocks until kill(), then reports a
    nonzero exit -- deterministic stand-in for "still running", unlike
    racing on how fast a fake stdout drains."""
    def __init__(self):
        import threading as _t
        self._ev = _t.Event()
        self._killed = False
        self.stdout = self

    def readline(self):
        self._ev.wait()
        return ""

    def poll(self):
        return -9 if self._killed else None

    def kill(self):
        self._killed = True
        self._ev.set()

    def wait(self, timeout=None):
        return -9


def test_second_start_rejected_while_running(tmp_path):
    mgr = _mgr(tmp_path, spawn=lambda a: BlockingProc())
    assert mgr.start(_cfg()).get("started") is True
    out = mgr.start(_cfg())
    assert "error" in out and "already in progress" in out["error"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_image_training_manager.py -v --import-mode=importlib`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.image_training.manager'`

- [ ] **Step 3: Write the implementation**

```python
# src/image_training/manager.py
"""Orchestrate the SDXL image-LoRA training sidecar from the Py3.14 app.
Never imports the training stack; never raises into the route. One run
at a time. Mirrors src/training/manager.py's TrainingManager shape,
scoped to the single family/toolchain combo the feasibility spike
proved (SDXL via diffusers)."""
import json
import os
import re
import subprocess
import threading

from src.image_training.runtime import resolve_image_sidecar_script


def _default_spawn(argv):
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1,
                            env=env,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _safe_output_name(name) -> str:
    try:
        base = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name or "").strip()).strip("-.")
    except Exception:  # noqa: BLE001
        base = ""
    return base or "lora"


def _new_run_id():
    import datetime
    return datetime.datetime.now().strftime("imgrun-%Y%m%d-%H%M%S")


class ImageTrainingManager:
    def __init__(self, env=None, spawn=None, dataset_store=None, runs_dir=None, loras_dir=None):
        if env is None:
            from src.image_training.env import ImageTrainingEnv
            env = ImageTrainingEnv()
        if dataset_store is None:
            from src.image_dataset_tools.store import get_image_dataset_store
            dataset_store = get_image_dataset_store()
        if runs_dir is None:
            from src.constants import DATA_DIR
            runs_dir = os.path.join(DATA_DIR, "training", "image_training_runs")
        if loras_dir is None:
            from src.imagemodels.loras import loras_dir as _loras_dir
            loras_dir = _loras_dir
        self._env = env
        self._spawn = spawn or _default_spawn
        self._dataset_store = dataset_store
        self._runs_dir = runs_dir
        self._loras_dir = loras_dir
        self._proc = None
        self._starting = False
        self._state = {"status": "idle", "last_step": None, "loss": None,
                       "vram_gb": None, "peak_vram_gb": None, "error": None, "lora_path": None}
        self._lock = threading.Lock()

    def start(self, config) -> dict:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return {"error": "a training run is already in progress"}
            if self._starting:
                return {"error": "a training run is already starting"}
            self._starting = True
        try:
            errs = config.validate()
            if errs:
                return {"error": "; ".join(errs)}
            try:
                ds = self._dataset_store.load(config.dataset_name)
            except Exception as e:  # noqa: BLE001
                return {"error": f"dataset: {e}"}
            if not isinstance(ds, dict) or ds.get("error"):
                return {"error": f"dataset: {(ds or {}).get('error', 'not found')}"}
            imgs = ds.get("images")
            if not isinstance(imgs, list) or not imgs:
                return {"error": "dataset has no images"}
            trigger = ds.get("trigger_word") or ""
            path = ds.get("path")
            if not isinstance(path, str) or not path:
                return {"error": "dataset has no path"}

            ready = self._env.ensure_ready()
            if not ready.get("ready"):
                return {"error": f"image training env not ready: {ready.get('error')}"}

            items = []
            for img in imgs:
                if not isinstance(img, dict):
                    continue
                fn = img.get("filename")
                if not isinstance(fn, str) or not fn:
                    continue
                caption = img.get("caption") if isinstance(img.get("caption"), str) else ""
                text = f"{trigger}, {caption}" if trigger and caption else (trigger or caption)
                items.append({"image": os.path.join(path, fn), "caption": text})
            if not items:
                return {"error": "dataset has no usable images"}

            run_id = _new_run_id()
            run_dir = os.path.join(self._runs_dir, run_id)
            filename = _safe_output_name(config.output_name) + ".safetensors"
            lora_path = os.path.join(self._loras_dir(), filename)
            try:
                os.makedirs(run_dir, exist_ok=True)
                cfg = config.to_dict()
                cfg["images"] = items
                cfg["lora_path"] = lora_path
                cfg["run_dir"] = run_dir
                cfg_path = os.path.join(run_dir, "config.json")
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f)
                argv = [self._env.venv_python(), resolve_image_sidecar_script(), "--config", cfg_path]
                with self._lock:
                    self._state = {"status": "running", "last_step": None, "loss": None,
                                   "vram_gb": None, "peak_vram_gb": None, "error": None,
                                   "lora_path": lora_path, "run_id": run_id}
                    self._proc = self._spawn(argv)
                    proc = self._proc
            except Exception as e:  # noqa: BLE001
                self._state["status"] = "error"
                self._state["error"] = f"could not start training: {e}"
                return {"error": self._state["error"]}
            threading.Thread(target=self._pump, args=(proc,), daemon=True).start()
            return {"started": True, "run_id": run_id}
        except Exception as e:  # noqa: BLE001
            return {"error": f"could not start training: {e}"}
        finally:
            with self._lock:
                self._starting = False

    def _pump(self, proc):
        tail = []
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                tail.append(line)
                tail[:] = tail[-40:]
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if self._proc is not proc:
                    continue  # a newer run replaced us -- stop touching shared state
                kind = ev.get("event")
                if kind == "step":
                    self._state.update(status="running", last_step=ev.get("step"),
                                       loss=ev.get("loss"), vram_gb=ev.get("vram_gb"))
                elif kind == "done":
                    self._state.update(status="done", peak_vram_gb=ev.get("peak_vram_gb"),
                                       lora_path=ev.get("lora_path", self._state.get("lora_path")))
                elif kind == "error":
                    self._state.update(status="error", error=ev.get("message"))
        except Exception:
            pass
        finally:
            if self._proc is proc:
                rc = proc.poll()
                if rc not in (0, None) and self._state["status"] not in ("done", "error", "stopped"):
                    self._state.update(status="error",
                                       error="training process exited: " + "".join(tail)[-500:])

    def status(self) -> dict:
        return dict(self._state)

    def stop(self) -> dict:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=5)
                except Exception:
                    pass
                self._state["status"] = "stopped"
                return {"stopped": True}
            return {"stopped": False}

    def env_status(self) -> dict:
        try:
            return {"status": self._env.status()}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def setup_env(self) -> dict:
        try:
            return self._env.ensure_ready()
        except Exception as e:
            return {"ready": False, "error": str(e)}


_manager = None


def get_image_training_manager():
    global _manager
    if _manager is None:
        _manager = ImageTrainingManager()
    return _manager
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_image_training_manager.py -v --import-mode=importlib`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/image_training/manager.py tests/test_image_training_manager.py
git commit -m "feat(image-training): add ImageTrainingManager"
```

---

### Task 6: API routes + app wiring

**Files:**
- Create: `routes/image_training_routes.py`
- Modify: `app.py` (near `app.py:770-771`, right after `setup_training_routes` is imported/included)
- Test: `tests/test_image_training_routes.py`

**Interfaces:**
- Consumes: `ImageTrainingConfig` (Task 1), `get_image_training_manager()` (Task 5) — `.env_status()`, `.setup_env()`, `.start(config)`, `.status()`, `.stop()`; `core.middleware.require_admin` (existing, shipped — same admin gate every other admin-only router uses).
- Produces: FastAPI router at prefix `/api/image-training` — `GET /env`, `POST /env/setup`, `POST /runs`, `GET /runs/current`, `POST /runs/stop`. Registered on the app via `setup_image_training_routes()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_image_training_routes.py
from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.image_training_routes as itr


class FakeMgr:
    def env_status(self):
        return {"status": "not_installed"}

    def status(self):
        return {"status": "idle"}


def _client(monkeypatch, mgr):
    monkeypatch.setattr(itr, "get_image_training_manager", lambda: mgr)
    monkeypatch.setattr(itr, "require_admin", lambda: None)  # bypass admin gate for shape tests
    app = FastAPI()
    app.include_router(itr.setup_image_training_routes())
    return TestClient(app)


def test_env_endpoint(monkeypatch):
    c = _client(monkeypatch, FakeMgr())
    r = c.get("/api/image-training/env")
    assert r.status_code == 200 and r.json()["status"] == "not_installed"


def test_run_status_endpoint(monkeypatch):
    c = _client(monkeypatch, FakeMgr())
    r = c.get("/api/image-training/runs/current")
    assert r.status_code == 200 and r.json()["status"] == "idle"


def test_start_run_validates_body(monkeypatch):
    mgr = FakeMgr()
    mgr.start = lambda cfg: {"started": True, "run_id": "run-x"}
    c = _client(monkeypatch, mgr)
    r = c.post("/api/image-training/runs",
              json={"dataset_name": "ds1", "output_name": "my-lora", "steps": 2})
    assert r.status_code == 200 and r.json()["started"] is True


def test_start_run_surfaces_manager_error(monkeypatch):
    mgr = FakeMgr()
    mgr.start = lambda cfg: {"error": "bad config"}
    c = _client(monkeypatch, mgr)
    r = c.post("/api/image-training/runs", json={"dataset_name": "", "output_name": "", "steps": 2})
    assert r.status_code == 400


def test_stop_endpoint(monkeypatch):
    mgr = FakeMgr()
    mgr.stop = lambda: {"stopped": True}
    c = _client(monkeypatch, mgr)
    r = c.post("/api/image-training/runs/stop")
    assert r.status_code == 200 and r.json()["stopped"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_image_training_routes.py -v --import-mode=importlib`
Expected: FAIL with `ModuleNotFoundError: No module named 'routes.image_training_routes'`

- [ ] **Step 3: Write the routes and wire them into the app**

```python
# routes/image_training_routes.py
"""Admin-gated Image LoRA Training API. All heavy work happens in the
image-training sidecar (the existing training venv extended with
diffusers); these routes just orchestrate it. Mirrors
routes/training_routes.py's shape, scoped to the single family/toolchain
the feasibility spike proved: SDXL via diffusers (see
docs/superpowers/specs/2026-07-31-image-lora-training-engine-design.md)."""
import asyncio

from fastapi import APIRouter, Body, Depends, HTTPException

from core.middleware import require_admin
from src.image_training.manager import get_image_training_manager
from src.image_training.config import ImageTrainingConfig


def setup_image_training_routes() -> APIRouter:
    router = APIRouter(prefix="/api/image-training",
                       dependencies=[Depends(require_admin)])

    @router.get("/env")
    async def env():
        return get_image_training_manager().env_status()

    @router.post("/env/setup")
    async def env_setup():
        # the install may pull diffusers (multi-hundred-MB); run it off the event loop
        return await asyncio.to_thread(get_image_training_manager().setup_env)

    @router.post("/runs")
    async def start_run(body: dict = Body(...)):
        try:
            cfg = ImageTrainingConfig(
                dataset_name=str(body.get("dataset_name", "")),
                output_name=str(body.get("output_name", "")),
                base_model=str(body.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")),
                rank=int(body.get("rank", 4)),
                lora_alpha=int(body.get("lora_alpha", 4)),
                learning_rate=float(body.get("learning_rate", 1e-4)),
                steps=int(body.get("steps", 1000)),
                resolution=int(body.get("resolution", 1024)),
            )
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"invalid config: {e}")
        out = get_image_training_manager().start(cfg)
        if "error" in out:
            raise HTTPException(400, out["error"])
        return out

    @router.get("/runs/current")
    async def current():
        return get_image_training_manager().status()

    @router.post("/runs/stop")
    async def stop():
        return get_image_training_manager().stop()

    return router
```

In `app.py`, right after the existing lines:
```python
from routes.training_routes import setup_training_routes
app.include_router(setup_training_routes())
```
add:
```python
from routes.image_training_routes import setup_image_training_routes
app.include_router(setup_image_training_routes())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_image_training_routes.py -v --import-mode=importlib`
Expected: PASS (5 tests)

Then verify the app still boots cleanly with the new router registered and nothing heavy gets imported at module load:

Run: `python -c "import app; print('/api/image-training/runs' in [r.path for r in app.app.routes])"`
Expected: prints `True` with no import errors

- [ ] **Step 5: Commit**

```bash
git add routes/image_training_routes.py app.py tests/test_image_training_routes.py
git commit -m "feat(image-training): wire up the image LoRA training API"
```

---

## Self-Review Notes

**Spec coverage:** The design spec's "Architecture" section maps to Tasks 1-5 (`src/image_training/{config,env,runtime,manager}.py` + `image_training_sidecar/train_sdxl_lora.py`) one-for-one; "Data flow" maps to Task 6's routes; "Output: a trained LoRA .safetensors... no conversion step" is satisfied by the manager resolving `lora_path` directly inside `loras_dir()` and the sidecar `os.replace`-ing straight there; "Error handling" (never raises, clear OOM-style messages, sidecar JSON-over-stdout/UTF-8/tqdm-disabled) is covered by Task 5's try/except shape and Task 4's `disable_progress_bars()` call plus the reused `_default_spawn` UTF-8 env vars. "Non-goals" (GUI, serving, export/conversion, multi-LoRA/dreambooth, unsupported families) are explicitly excluded and none of the 6 tasks touch them.

**Placeholder scan:** No TBD/TODO markers; every step has literal, runnable code; no "similar to Task N" cross-references — Task 4 repeats the full precompute-then-offload sequence inline rather than pointing back at the spike.

**Type consistency:** `ImageTrainingConfig` field names (`dataset_name`, `output_name`, `base_model`, `rank`, `lora_alpha`, `learning_rate`, `steps`, `resolution`) are used identically in Task 1's dataclass, Task 5's `config.to_dict()` consumer and route-body parsing in Task 6, and the sidecar config keys in Task 4 (`images`, `base_model`, `rank`, `lora_alpha`, `learning_rate`, `steps`, `resolution`, `lora_path`, `run_dir`). The progress-event shape (`step`/`loss`/`vram_gb`, `done`/`lora_path`/`peak_vram_gb`, `error`/`message`) is identical between Task 4's `emit()` calls and Task 5's `_pump()` parsing. `ImageTrainingEnv`/`ImageTrainingManager`/`resolve_image_sidecar_script` names are used consistently everywhere they're referenced across tasks.
