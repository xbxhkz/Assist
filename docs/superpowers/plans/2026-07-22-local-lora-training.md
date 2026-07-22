# Local LoRA/QLoRA Training Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune a small base model with QLoRA on the user's GPU, entirely locally, driven by an admin-gated `/api/training/*` HTTP API — producing a LoRA adapter. (The Training GUI is a deferred follow-up sub-project; this plan is the headless engine only.)

**Architecture:** The Python-3.14 main app cannot import torch/peft/bitsandbytes, so all training runs in a separate Python-3.11 CUDA venv set up on demand via a vendored `uv` binary; the app orchestrates it as a subprocess (like `llama-server`). A `training_sidecar/train.py` runs inside that venv, streaming JSON-line progress; the manager and routes never import the training stack.

**Tech Stack:** Python 3.14 (app) + a Python 3.11 CUDA venv (torch cu121, transformers, peft, bitsandbytes, accelerate, datasets, trl); `uv`; FastAPI.

## Global Constraints

- **The main app MUST NOT import torch/peft/bitsandbytes/transformers.** All training code lives ONLY in `training_sidecar/train.py`, which runs in the Py3.11 venv and is never imported by the app or its Py3.14 test suite.
- **The manager and routes NEVER raise** — degrade to `{"error": …}` / an error status.
- **Admin-only** — the `/api/training/*` routes are admin-gated (the GUI, when it lands, will reuse them).
- **Vendor `uv`** (single ~30 MB binary) into `build_assets/uv/`, and **bundle `training_sidecar/`**, both as `Assist.spec` datas; resolve each at runtime (frozen `sys._MEIPASS/<sub>`, dev repo path) mirroring `resolve_llama_binary`.
- **QLoRA (4-bit nf4) only**, base model = a HuggingFace repo id, dataset = a local JSONL file, one run at a time.
- **VRAM is binary GiB** — use `services.hwfit.hardware.free_vram_gb()` (returns GiB or None; never raises).
- **Feasibility is PROVEN GO** (spike: `torch 2.5.1+cu121`, `bitsandbytes 0.49.2`, `peft 0.19.1`, `transformers 5.14.1`; 0.5B QLoRA peaked 1.16 GB of 6.44; adapter saved). The spike script is at `C:\tmp\train_spike\train_spike.py` — Task 5 productionizes it. No feasibility-gate task.
- **No frontend in this plan.** The engine is headless and API-driven; the admin Training GUI (modal, live progress panel, adapters list, VRAM-fit hint, `trainingCore.js`) is the next sub-project and will consume these routes unchanged.
- pytest `--import-mode=importlib`. Commit directly to `dev`. Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Stage only the files each task names — never `git add -A`; never stage `installer/Output/Assist-Setup.exe`, `assistappicon.png`, `assistlogo.png`, or the vendored `build_assets/uv/uv.exe` (build-time artifact).
- ~220 unrelated pre-existing test failures exist elsewhere — run only the test files each task names.
- **Owed by the user (manual):** one real end-to-end train via the API on the 6 GB GPU (`POST /api/training/env/setup` then `POST /api/training/runs`, watch `GET /api/training/runs/current`); a frozen boot-check after a rebuild (uv + sidecar resolve, training routes register). `train.py` is covered by the spike + that real run, not the Py3.14 suite.

---

### Task 1: Vendor `uv` + sidecar resolution + bundling

**Files:**
- Create: `scripts/fetch_uv.py`, `src/training/__init__.py` (empty), `src/training/runtime.py`
- Modify: `Assist.spec` (datas)
- Test: `tests/test_training_runtime.py`

**Interfaces:**
- Produces: `src.training.runtime.resolve_uv_binary(frozen_base=None, dev_base=None) -> str` (path to `uv.exe`; raises `RuntimeError` if not found); `src.training.runtime.resolve_sidecar_script(frozen_base=None, dev_base=None) -> str` (path to `training_sidecar/train.py`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_training_runtime.py`:

```python
import os
import pytest
from src.training import runtime


def test_resolve_uv_prefers_frozen(tmp_path):
    base = tmp_path / "frozen"
    (base / "uv").mkdir(parents=True)
    exe = base / "uv" / "uv.exe"
    exe.write_text("x")
    assert runtime.resolve_uv_binary(frozen_base=str(base)) == str(exe)


def test_resolve_uv_falls_back_to_dev(tmp_path):
    dev = tmp_path / "build_assets" / "uv"
    dev.mkdir(parents=True)
    exe = dev / "uv.exe"
    exe.write_text("x")
    assert runtime.resolve_uv_binary(frozen_base=str(tmp_path / "none"),
                                     dev_base=str(dev)) == str(exe)


def test_resolve_uv_raises_when_missing(tmp_path):
    with pytest.raises(RuntimeError):
        runtime.resolve_uv_binary(frozen_base=str(tmp_path / "a"),
                                  dev_base=str(tmp_path / "b"))


def test_resolve_sidecar_script(tmp_path):
    base = tmp_path / "frozen"
    (base / "training_sidecar").mkdir(parents=True)
    scr = base / "training_sidecar" / "train.py"
    scr.write_text("x")
    assert runtime.resolve_sidecar_script(frozen_base=str(base)) == str(scr)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_training_runtime.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.training'`).

- [ ] **Step 3: Create the package + runtime resolver**

Create `src/training/__init__.py` (empty file).

Create `src/training/runtime.py`:

```python
"""Locate the vendored `uv` binary and the training sidecar script at runtime.

Mirrors src/localmodels/runtime.py's resolve_llama_binary frozen/dev resolution:
frozen builds look under sys._MEIPASS, dev under the repo's build_assets / root.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _frozen_base(explicit):
    if explicit is not None:
        return explicit
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", None)
    return None


def resolve_uv_binary(frozen_base=None, dev_base=None) -> str:
    """Path to the vendored uv.exe. Frozen: <_MEIPASS>/uv/uv.exe; dev:
    <repo>/build_assets/uv/uv.exe. Raises RuntimeError if not found."""
    name = "uv.exe" if os.name == "nt" else "uv"
    base = _frozen_base(frozen_base)
    if base:
        cand = os.path.join(base, "uv", name)
        if os.path.isfile(cand):
            return cand
    if dev_base is None:
        dev_base = os.path.join(_REPO_ROOT, "build_assets", "uv")
    cand = os.path.join(dev_base, name)
    if os.path.isfile(cand):
        return cand
    raise RuntimeError(
        "uv binary not found: run scripts/fetch_uv.py to vendor it into "
        "build_assets/uv/, or ensure it's bundled in the frozen build.")


def resolve_sidecar_script(frozen_base=None, dev_base=None) -> str:
    """Path to training_sidecar/train.py. Frozen: <_MEIPASS>/training_sidecar/
    train.py; dev: <repo>/training_sidecar/train.py. Raises RuntimeError if missing."""
    base = _frozen_base(frozen_base)
    if base:
        cand = os.path.join(base, "training_sidecar", "train.py")
        if os.path.isfile(cand):
            return cand
    if dev_base is None:
        dev_base = os.path.join(_REPO_ROOT, "training_sidecar")
    cand = os.path.join(dev_base, "train.py")
    if os.path.isfile(cand):
        return cand
    raise RuntimeError("training_sidecar/train.py not found.")
```

- [ ] **Step 4: Create the uv fetch script**

Create `scripts/fetch_uv.py`:

```python
"""Vendor the `uv` binary (Astral) into build_assets/uv/uv.exe.

uv sets up the on-demand Python 3.11 CUDA training venv on the user's machine.
Downloads the Windows x64 zip from GitHub releases and extracts uv.exe. Uses the
`latest` asset by default; pin via UV_RELEASE_TAG for reproducible builds.
"""
import io
import os
import sys
import urllib.request
import zipfile

TAG = os.getenv("UV_RELEASE_TAG", "").strip()
ASSET = "uv-x86_64-pc-windows-msvc.zip"
URL = (f"https://github.com/astral-sh/uv/releases/download/{TAG}/{ASSET}" if TAG
       else f"https://github.com/astral-sh/uv/releases/latest/download/{ASSET}")
DEST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build_assets", "uv"))


def main() -> int:
    exe = os.path.join(DEST, "uv.exe")
    if os.path.isfile(exe):
        print("uv: already vendored, skipping")
        return 0
    os.makedirs(DEST, exist_ok=True)
    print(f"Downloading uv from {URL} ...")
    with urllib.request.urlopen(URL) as resp:  # noqa: S310
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if os.path.basename(member).lower() == "uv.exe":
                with zf.open(member) as src, open(exe, "wb") as dst:
                    dst.write(src.read())
    if not os.path.isfile(exe):
        print("ERROR: uv.exe not found in the release zip", file=sys.stderr)
        return 1
    print(f"uv vendored into {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Add both to `Assist.spec` datas**

In `Assist.spec`, add two entries to the `datas` list (after the `build_assets/yolo` line, before `] + _collected_datas`):

```python
    # Vendored uv binary — sets up the on-demand Python 3.11 CUDA training venv.
    ('build_assets/uv', 'uv'),
    # The training sidecar script, run inside the training venv (never imported
    # by the Py3.14 app). Bundled so the frozen build can launch it.
    ('training_sidecar', 'training_sidecar'),
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_training_runtime.py --import-mode=importlib -q`
Expected: PASS (4 passed).

- [ ] **Step 7: Commit**

Do NOT run `scripts/fetch_uv.py` here and do NOT stage `build_assets/uv/uv.exe` — it is a build-time artifact fetched before packaging (like the llama/sd binaries), not committed.

```bash
git add scripts/fetch_uv.py src/training/__init__.py src/training/runtime.py Assist.spec tests/test_training_runtime.py
git commit -m "feat(training): vendor uv + resolve sidecar; bundle in Assist.spec

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `TrainingConfig` + VRAM estimate

**Files:**
- Create: `src/training/config.py`
- Test: `tests/test_training_config.py`

**Interfaces:**
- Produces: `TrainingConfig` (dataclass) with `.validate() -> list[str]` (empty = valid) and `.to_dict() -> dict`; `estimate_vram_gb(params_b) -> float`; `fit_level(params_b, free_gib) -> str` ("fits"|"tight"|"too_big"|"unknown"); `parse_params_b(model_id) -> float | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_training_config.py`:

```python
from src.training.config import (TrainingConfig, estimate_vram_gb, fit_level,
                                 parse_params_b)


def _cfg(**kw):
    base = dict(base_model="Qwen/Qwen2.5-0.5B-Instruct", dataset_path="d.jsonl",
                lora_r=8, lora_alpha=16, lora_dropout=0.05, steps=100, epochs=None,
                batch_size=1, learning_rate=2e-4)
    base.update(kw)
    return TrainingConfig(**base)


def test_valid_config_has_no_errors():
    assert _cfg().validate() == []


def test_invalid_configs_report_errors():
    assert _cfg(base_model="").validate()
    assert _cfg(lora_r=0).validate()
    assert _cfg(lora_dropout=1.5).validate()
    assert _cfg(batch_size=0).validate()
    assert _cfg(learning_rate=0).validate()
    assert _cfg(steps=None, epochs=None).validate()      # need one
    assert _cfg(steps=100, epochs=3).validate()          # not both


def test_estimate_and_fit_level():
    assert estimate_vram_gb(0.5) < 2.0
    assert fit_level(0.5, 6.44) == "fits"
    assert fit_level(13, 6.44) == "too_big"
    assert fit_level(3, 6.44) == "fits"
    assert fit_level(0.5, None) == "unknown"


def test_parse_params_b():
    assert parse_params_b("Qwen/Qwen2.5-0.5B-Instruct") == 0.5
    assert parse_params_b("meta-llama/Meta-Llama-3-8B") == 8.0
    assert parse_params_b("TinyLlama/TinyLlama-1.1B") == 1.1
    assert parse_params_b("openai-community/gpt2") is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_training_config.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.training.config'`).

- [ ] **Step 3: Implement**

Create `src/training/config.py`:

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_training_config.py --import-mode=importlib -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/training/config.py tests/test_training_config.py
git commit -m "feat(training): TrainingConfig + validation + VRAM-fit estimate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Dataset loader

**Files:**
- Create: `src/training/dataset.py`
- Test: `tests/test_training_dataset.py`

**Interfaces:**
- Produces: `load_jsonl(path) -> list[dict]` — each returned row is `{"text": <str>}`. Raises `ValueError` (message names the line) on a malformed row; raises `FileNotFoundError` if the file is missing.

- [ ] **Step 1: Write the failing test**

Create `tests/test_training_dataset.py`:

```python
import json
import pytest
from src.training.dataset import load_jsonl


def _write(tmp_path, rows):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def test_text_rows(tmp_path):
    out = load_jsonl(_write(tmp_path, [{"text": "hello"}, {"text": "world"}]))
    assert out == [{"text": "hello"}, {"text": "world"}]


def test_instruction_response_rows(tmp_path):
    out = load_jsonl(_write(tmp_path, [{"instruction": "greet", "response": "hi"}]))
    assert out[0]["text"].startswith("greet") and "hi" in out[0]["text"]


def test_prompt_completion_rows(tmp_path):
    out = load_jsonl(_write(tmp_path, [{"prompt": "Q", "completion": "A"}]))
    assert "Q" in out[0]["text"] and "A" in out[0]["text"]


def test_bad_row_raises_with_line(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"text": "ok"}\n{"nope": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        load_jsonl(str(p))
    assert "2" in str(ei.value)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_jsonl(str(tmp_path / "no.jsonl"))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_training_dataset.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.training.dataset'`).

- [ ] **Step 3: Implement**

Create `src/training/dataset.py`:

```python
"""Load + normalize a JSONL training dataset into [{"text": ...}] rows.

Accepts three row shapes: {"text"}, {"instruction","response"} (optional
"input"), and {"prompt","completion"}. Pure except reading the file."""
import json


def _normalize(row: dict) -> str:
    if isinstance(row.get("text"), str) and row["text"].strip():
        return row["text"]
    if isinstance(row.get("instruction"), str):
        instr = row["instruction"]
        inp = row.get("input")
        resp = row.get("response", row.get("output", ""))
        head = f"{instr}\n{inp}" if isinstance(inp, str) and inp.strip() else instr
        return f"{head}\n{resp}"
    if isinstance(row.get("prompt"), str):
        return f"{row['prompt']}\n{row.get('completion', '')}"
    raise ValueError("row needs 'text', or 'instruction'(+response), or 'prompt'(+completion)")


def load_jsonl(path: str) -> list:
    """Return [{"text": str}, ...]. Raises FileNotFoundError / ValueError
    (naming the offending 1-based line)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {i}: invalid JSON ({e})") from e
            if not isinstance(obj, dict):
                raise ValueError(f"line {i}: each row must be a JSON object")
            try:
                text = _normalize(obj)
            except ValueError as e:
                raise ValueError(f"line {i}: {e}") from e
            rows.append({"text": text})
    if not rows:
        raise ValueError("dataset is empty")
    return rows
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_training_dataset.py --import-mode=importlib -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/training/dataset.py tests/test_training_dataset.py
git commit -m "feat(training): JSONL dataset loader/normalizer

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `TrainingEnv` — on-demand venv setup

**Files:**
- Create: `src/training/env.py`
- Test: `tests/test_training_env.py`

**Interfaces:**
- Consumes: `resolve_uv_binary` (Task 1); `src.constants.DATA_DIR`.
- Produces: `TrainingEnv(base_dir=None, uv_binary=None, run=None)` with `venv_python() -> str`, `status() -> str` ("not_installed"|"ready"), `ensure_ready(progress=None) -> dict` (`{"ready": bool, "error": str|None}`; never raises). `run` is an injectable command runner `run(argv) -> (returncode, output)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_training_env.py`:

```python
import os
from src.training.env import TrainingEnv


def test_status_not_installed(tmp_path):
    env = TrainingEnv(base_dir=str(tmp_path), uv_binary="uv.exe")
    assert env.status() == "not_installed"


def test_ensure_ready_runs_uv_steps_then_marks_ready(tmp_path):
    calls = []

    def fake_run(argv):
        calls.append(argv)
        # simulate uv creating the venv python on the venv step
        if argv[:2] == ["uv.exe", "venv"]:
            os.makedirs(os.path.dirname(TrainingEnv(base_dir=str(tmp_path)).venv_python()),
                        exist_ok=True)
            open(TrainingEnv(base_dir=str(tmp_path)).venv_python(), "w").close()
        return (0, "ok")

    env = TrainingEnv(base_dir=str(tmp_path), uv_binary="uv.exe", run=fake_run)
    out = env.ensure_ready()
    assert out["ready"] is True and out["error"] is None
    # python install, venv, torch install, stack install
    kinds = [c[1] for c in calls]
    assert kinds[:2] == ["python", "venv"] and "pip" in kinds
    assert env.status() == "ready"


def test_ensure_ready_idempotent_skips_when_ready(tmp_path):
    env = TrainingEnv(base_dir=str(tmp_path), uv_binary="uv.exe",
                      run=lambda argv: (0, ""))
    os.makedirs(os.path.dirname(env.venv_python()), exist_ok=True)
    open(env.venv_python(), "w").close()
    open(env._marker(), "w").close()
    calls = []
    env2 = TrainingEnv(base_dir=str(tmp_path), uv_binary="uv.exe",
                       run=lambda argv: (calls.append(argv), (0, ""))[1])
    assert env2.ensure_ready()["ready"] is True
    assert calls == []            # nothing re-run


def test_ensure_ready_never_raises_on_failure(tmp_path):
    env = TrainingEnv(base_dir=str(tmp_path), uv_binary="uv.exe",
                      run=lambda argv: (1, "boom"))
    out = env.ensure_ready()
    assert out["ready"] is False and "boom" in (out["error"] or "")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_training_env.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.training.env'`).

- [ ] **Step 3: Implement**

Create `src/training/env.py`:

```python
"""On-demand Python 3.11 CUDA training venv, set up via the vendored `uv`.

The main app (Py3.14) can't run torch; this builds a side venv with the CUDA
training stack the spike proved. Never raises — ensure_ready returns a status
dict. The `run` callable is injectable for tests."""
import os
import subprocess

PY_VERSION = "3.11"
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
STACK = ["transformers", "peft", "bitsandbytes", "accelerate", "datasets", "trl"]


def _default_run(argv):
    p = subprocess.run(argv, capture_output=True, text=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return p.returncode, (p.stdout or "") + (p.stderr or "")


class TrainingEnv:
    def __init__(self, base_dir=None, uv_binary=None, run=None):
        if base_dir is None:
            from src.constants import DATA_DIR
            base_dir = os.path.join(DATA_DIR, "training")
        self._base = base_dir
        self._venv = os.path.join(base_dir, "venv")
        self._uv = uv_binary
        self._run = run or _default_run

    def _uv_bin(self):
        if self._uv is None:
            from src.training.runtime import resolve_uv_binary
            self._uv = resolve_uv_binary()
        return self._uv

    def venv_python(self) -> str:
        sub = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
        return os.path.join(self._venv, *sub)

    def _marker(self) -> str:
        return os.path.join(self._venv, ".assist_training_ready")

    def status(self) -> str:
        return "ready" if (os.path.isfile(self.venv_python())
                           and os.path.isfile(self._marker())) else "not_installed"

    def ensure_ready(self, progress=None) -> dict:
        """Idempotently build the training venv. Returns {"ready","error"}. Never raises."""
        if self.status() == "ready":
            return {"ready": True, "error": None}
        try:
            uv = self._uv_bin()
            py = self.venv_python()
            steps = [
                [uv, "python", "install", PY_VERSION],
                [uv, "venv", "--python", PY_VERSION, self._venv],
                [uv, "pip", "install", "--python", py, "torch", "--index-url", TORCH_INDEX],
                [uv, "pip", "install", "--python", py] + STACK,
            ]
            os.makedirs(self._base, exist_ok=True)
            for argv in steps:
                if progress:
                    try:
                        progress({"event": "install", "cmd": " ".join(argv[:3])})
                    except Exception:
                        pass
                rc, out = self._run(argv)
                if rc != 0:
                    return {"ready": False, "error": f"uv step failed ({argv[1]}): {out[-500:]}"}
            if not os.path.isfile(py):
                return {"ready": False, "error": "venv python missing after setup"}
            open(self._marker(), "w").close()
            return {"ready": True, "error": None}
        except Exception as e:  # noqa: BLE001
            return {"ready": False, "error": f"training env setup failed: {e}"}
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_training_env.py --import-mode=importlib -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/training/env.py tests/test_training_env.py
git commit -m "feat(training): TrainingEnv on-demand CUDA venv via uv (never-raises)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: The training sidecar (`training_sidecar/train.py`)

**Files:**
- Create: `training_sidecar/train.py`, `training_sidecar/__init__.py` (empty, so it can be a bundled dir)

**Interfaces:**
- Produces: a standalone script invoked as `python train.py --config <config.json>`. Config JSON keys: `base_model, dataset_path, output_dir, lora_r, lora_alpha, lora_dropout, steps|epochs, batch_size, learning_rate, max_seq_length`. Emits JSON lines to stdout: `{"event":"start"|"step"|"done"|"error", ...}`. Saves `adapter_model.safetensors` + `adapter_config.json` + `run_config.json` into `output_dir`.

**IMPORTANT:** This runs ONLY inside the Py3.11 training venv. It is NOT imported by the main app or its test suite (it imports torch/peft). It has **no unit test** — it is verified by the proven spike (`C:\tmp\train_spike\train_spike.py`) and by the user's real end-to-end run. Keep it self-contained and defensive; emit a JSON `error` event (never a bare traceback to stdout's data stream) on failure.

- [ ] **Step 1: Create the sidecar script**

Create `training_sidecar/__init__.py` (empty).

Create `training_sidecar/train.py` (productionized from the proven spike — QLoRA 4-bit, LoRA via peft, transformers `Trainer`, JSON-line progress):

```python
"""QLoRA training sidecar. Runs INSIDE the Python 3.11 CUDA venv (never imported
by the Py3.14 app). Reads a JSON config, fine-tunes a base model with a 4-bit
QLoRA, streams JSON-line progress to stdout, and saves a LoRA adapter.

Progress protocol (one JSON object per line on stdout):
  {"event":"start","model":..,"total_steps":..}
  {"event":"step","step":N,"loss":..,"vram_gb":..}
  {"event":"done","output_dir":..,"peak_vram_gb":..}
  {"event":"error","message":..}
"""
import argparse
import json
import os
import sys


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig, TrainingArguments, Trainer,
                                  DataCollatorForLanguageModeling, TrainerCallback)
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset

        # dataset is normalized to [{"text": ...}] by the app before launch
        with open(cfg["dataset_path"], "r", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]

        model_id = cfg["base_model"]
        out_dir = cfg["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        max_len = int(cfg.get("max_seq_length", 512))

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True)
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb,
                                                     device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora = LoraConfig(r=int(cfg.get("lora_r", 8)), lora_alpha=int(cfg.get("lora_alpha", 16)),
                          lora_dropout=float(cfg.get("lora_dropout", 0.05)), bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, lora)
        model.config.use_cache = False

        ds = Dataset.from_list(rows).map(
            lambda ex: tok(ex["text"], truncation=True, max_length=max_len, padding="max_length"),
            remove_columns=["text"])
        collator = DataCollatorForLanguageModeling(tok, mlm=False)

        steps = cfg.get("steps")
        epochs = cfg.get("epochs")
        targs = dict(output_dir=out_dir, per_device_train_batch_size=int(cfg.get("batch_size", 1)),
                     gradient_accumulation_steps=1, learning_rate=float(cfg.get("learning_rate", 2e-4)),
                     logging_steps=1, gradient_checkpointing=True, fp16=True,
                     report_to=[], save_strategy="no")
        if steps:
            targs["max_steps"] = int(steps)
        else:
            targs["num_train_epochs"] = float(epochs or 1)
        targ = TrainingArguments(**targs)

        total = int(steps) if steps else None
        emit({"event": "start", "model": model_id, "total_steps": total})

        class Cb(TrainerCallback):
            def on_log(self, a, state, control, logs=None, **kw):
                if logs and "loss" in logs:
                    v = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0
                    emit({"event": "step", "step": int(state.global_step),
                          "loss": round(float(logs["loss"]), 4), "vram_gb": v})

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        trainer = Trainer(model=model, args=targ, train_dataset=ds,
                          data_collator=collator, callbacks=[Cb()])
        trainer.train()

        trainer.model.save_pretrained(out_dir)
        with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0
        emit({"event": "done", "output_dir": out_dir, "peak_vram_gb": peak})
    except Exception as e:  # noqa: BLE001
        import traceback
        emit({"event": "error", "message": f"{e}", "trace": traceback.format_exc()[-1500:]})
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Byte-compile check (no execution — torch isn't in this Python)**

Run: `python -c "import ast; ast.parse(open('training_sidecar/train.py', encoding='utf-8').read()); print('syntax OK')"`
Expected: `syntax OK` (this only parses; it must NOT be imported/run in the Py3.14 app — it imports torch).

- [ ] **Step 3: Commit**

```bash
git add training_sidecar/__init__.py training_sidecar/train.py
git commit -m "feat(training): QLoRA training sidecar (runs in the CUDA venv)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: `TrainingManager`

**Files:**
- Create: `src/training/manager.py`
- Test: `tests/test_training_manager.py`

**Interfaces:**
- Consumes: `TrainingConfig` (Task 2), `load_jsonl` (Task 3), `TrainingEnv` (Task 4), `resolve_sidecar_script` (Task 1), `config.parse_params_b`/`fit_level`, `hwfit.free_vram_gb`.
- Produces: `TrainingManager(env=None, spawn=None, free_vram=None, adapters_dir=None)` with `start(config: TrainingConfig) -> dict`, `status() -> dict`, `stop() -> dict`, `list_adapters() -> list`, `env_status() -> dict`, `setup_env() -> dict`, plus a module singleton `get_training_manager()`. Never raises. `spawn(argv) -> proc` and `free_vram() -> float|None` are injectable.

- [ ] **Step 1: Write the failing test**

Create `tests/test_training_manager.py`:

```python
import json
from src.training.config import TrainingConfig
from src.training.manager import TrainingManager


class FakeProc:
    def __init__(self, lines):
        self._lines = list(lines)
        self.stdout = self
        self._killed = False
    def readline(self):
        return self._lines.pop(0) if self._lines else ""
    def poll(self):
        return None if self._lines else 0
    def kill(self):
        self._killed = True
    def wait(self, timeout=None):
        return 0


class FakeEnv:
    def __init__(self, ready=True):
        self._ready = ready
    def ensure_ready(self, progress=None):
        return {"ready": self._ready, "error": None if self._ready else "no env"}
    def venv_python(self):
        return "venv/python"


def _cfg(tmp_path):
    ds = tmp_path / "d.jsonl"
    ds.write_text('{"text":"hi"}\n', encoding="utf-8")
    return TrainingConfig(base_model="x/Qwen2.5-0.5B", dataset_path=str(ds), steps=2)


def test_start_runs_sidecar_and_tracks_progress(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.manager.resolve_sidecar_script", lambda: "train.py")
    lines = [json.dumps({"event": "start", "total_steps": 2}) + "\n",
             json.dumps({"event": "step", "step": 1, "loss": 2.0, "vram_gb": 1.1}) + "\n",
             json.dumps({"event": "done", "output_dir": "o", "peak_vram_gb": 1.2}) + "\n"]
    captured = {}
    def spawn(argv):
        captured["argv"] = argv
        return FakeProc(lines)
    mgr = TrainingManager(env=FakeEnv(), spawn=spawn, free_vram=lambda: 6.4,
                          adapters_dir=str(tmp_path / "ad"))
    out = mgr.start(_cfg(tmp_path))
    assert out.get("started") is True
    # the pump thread runs to completion on the fake proc
    import time
    for _ in range(200):
        if mgr.status()["status"] in ("done", "error"):
            break
        time.sleep(0.01)
    assert "train.py" in captured["argv"] and "--config" in captured["argv"]
    st = mgr.status()
    assert st["status"] == "done" and st["last_step"] == 1 and st["peak_vram_gb"] == 1.2


def test_start_rejects_invalid_config(tmp_path):
    mgr = TrainingManager(env=FakeEnv(), spawn=lambda a: None, free_vram=lambda: 6.4)
    bad = TrainingConfig(base_model="", dataset_path="d.jsonl", steps=1)
    out = mgr.start(bad)
    assert "error" in out


def test_start_errors_when_env_not_ready(tmp_path):
    mgr = TrainingManager(env=FakeEnv(ready=False), spawn=lambda a: None, free_vram=lambda: 6.4)
    out = mgr.start(_cfg(tmp_path))
    assert "error" in out and "env" in out["error"].lower()


def test_start_never_raises_on_spawn_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.manager.resolve_sidecar_script", lambda: "train.py")
    def boom(argv):
        raise RuntimeError("cannot spawn")
    mgr = TrainingManager(env=FakeEnv(), spawn=boom, free_vram=lambda: 6.4,
                          adapters_dir=str(tmp_path / "ad"))
    out = mgr.start(_cfg(tmp_path))
    assert "error" in out
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_training_manager.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.training.manager'`).

- [ ] **Step 3: Implement**

Create `src/training/manager.py`:

```python
"""Orchestrate the training sidecar from the Py3.14 app. Never imports the
training stack; never raises into the route. One run at a time."""
import json
import os
import subprocess
import threading

from src.training.runtime import resolve_sidecar_script


def _default_spawn(argv):
    return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _default_free_vram():
    try:
        from services.hwfit.hardware import free_vram_gb
        return free_vram_gb()
    except Exception:
        return None


class TrainingManager:
    def __init__(self, env=None, spawn=None, free_vram=None, adapters_dir=None):
        if env is None:
            from src.training.env import TrainingEnv
            env = TrainingEnv()
        if adapters_dir is None:
            from src.constants import DATA_DIR
            adapters_dir = os.path.join(DATA_DIR, "training", "adapters")
        self._env = env
        self._spawn = spawn or _default_spawn
        self._free_vram = free_vram or _default_free_vram
        self._adapters_dir = adapters_dir
        self._proc = None
        self._state = {"status": "idle", "last_step": None, "loss": None,
                       "vram_gb": None, "peak_vram_gb": None, "error": None, "output_dir": None}
        self._lock = threading.Lock()

    def start(self, config) -> dict:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return {"error": "a training run is already in progress"}
            errs = config.validate()
            if errs:
                return {"error": "; ".join(errs)}
            try:
                from src.training.dataset import load_jsonl
                rows = load_jsonl(config.dataset_path)
            except Exception as e:
                return {"error": f"dataset: {e}"}
            ready = self._env.ensure_ready()
            if not ready.get("ready"):
                return {"error": f"training env not ready: {ready.get('error')}"}

            # VRAM soft-gate (warn only — the user chose the model)
            from src.training.config import parse_params_b, fit_level
            warning = None
            level = fit_level(parse_params_b(config.base_model), self._free_vram())
            if level == "too_big":
                warning = "model may exceed available VRAM; it may fail with out-of-memory"

            run_id = _new_run_id()
            out_dir = os.path.join(self._adapters_dir, run_id)
            try:
                os.makedirs(out_dir, exist_ok=True)
                ds_path = os.path.join(out_dir, "dataset.jsonl")
                with open(ds_path, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                cfg = config.to_dict()
                cfg["dataset_path"] = ds_path
                cfg["output_dir"] = out_dir
                cfg_path = os.path.join(out_dir, "config.json")
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f)
                argv = [self._env.venv_python(), resolve_sidecar_script(), "--config", cfg_path]
                self._state = {"status": "running", "last_step": None, "loss": None,
                               "vram_gb": None, "peak_vram_gb": None, "error": warning,
                               "output_dir": out_dir, "run_id": run_id}
                self._proc = self._spawn(argv)
            except Exception as e:
                self._state["status"] = "error"
                self._state["error"] = f"could not start training: {e}"
                return {"error": self._state["error"]}
            threading.Thread(target=self._pump, args=(self._proc,), daemon=True).start()
            return {"started": True, "run_id": run_id, "warning": warning}

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
                kind = ev.get("event")
                if kind == "step":
                    self._state.update(status="running", last_step=ev.get("step"),
                                       loss=ev.get("loss"), vram_gb=ev.get("vram_gb"))
                elif kind == "done":
                    self._state.update(status="done", peak_vram_gb=ev.get("peak_vram_gb"),
                                       output_dir=ev.get("output_dir", self._state.get("output_dir")))
                elif kind == "error":
                    self._state.update(status="error", error=ev.get("message"))
        except Exception:
            pass
        finally:
            rc = proc.poll()
            if rc not in (0, None) and self._state["status"] not in ("done", "error"):
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

    def list_adapters(self) -> list:
        out = []
        if not os.path.isdir(self._adapters_dir):
            return out
        for name in sorted(os.listdir(self._adapters_dir), reverse=True):
            d = os.path.join(self._adapters_dir, name)
            if not os.path.isdir(d):
                continue
            has = os.path.isfile(os.path.join(d, "adapter_model.safetensors"))
            cfg = {}
            rc = os.path.join(d, "run_config.json")
            if os.path.isfile(rc):
                try:
                    with open(rc, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                except Exception:
                    cfg = {}
            out.append({"run_id": name, "complete": has,
                        "base_model": cfg.get("base_model"), "path": d})
        return out


def _new_run_id():
    import datetime
    return datetime.datetime.now().strftime("run-%Y%m%d-%H%M%S")


_manager = None


def get_training_manager():
    global _manager
    if _manager is None:
        _manager = TrainingManager()
    return _manager
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_training_manager.py --import-mode=importlib -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/training/manager.py tests/test_training_manager.py
git commit -m "feat(training): TrainingManager orchestrates the sidecar (never-raises)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Admin API routes

**Files:**
- Create: `routes/training_routes.py`
- Modify: `app.py` (include the router)
- Test: `tests/test_training_routes.py`

**Interfaces:**
- Consumes: `get_training_manager` (Task 6), `TrainingConfig` (Task 2), `require_admin` (`core.middleware`).
- Produces: `setup_training_routes() -> APIRouter` mounted at `/api/training`, admin-gated. Endpoints: `GET /env`, `POST /env/setup`, `POST /runs`, `GET /runs/current`, `POST /runs/stop`, `GET /adapters`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_training_routes.py`:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.training_routes as tr


class FakeMgr:
    def env_status(self):
        return {"status": "not_installed"}
    def status(self):
        return {"status": "idle"}
    def list_adapters(self):
        return [{"run_id": "run-1", "complete": True, "base_model": "x", "path": "p"}]


def _client(monkeypatch, mgr):
    monkeypatch.setattr(tr, "get_training_manager", lambda: mgr)
    monkeypatch.setattr(tr, "require_admin", lambda: None)  # bypass admin gate for shape tests
    app = FastAPI()
    app.include_router(tr.setup_training_routes())
    return TestClient(app)


def test_adapters_endpoint(monkeypatch):
    c = _client(monkeypatch, FakeMgr())
    r = c.get("/api/training/adapters")
    assert r.status_code == 200 and r.json()["adapters"][0]["run_id"] == "run-1"


def test_run_status_endpoint(monkeypatch):
    c = _client(monkeypatch, FakeMgr())
    r = c.get("/api/training/runs/current")
    assert r.status_code == 200 and r.json()["status"] == "idle"


def test_start_run_validates_body(monkeypatch):
    mgr = FakeMgr()
    mgr.start = lambda cfg: {"started": True, "run_id": "run-x"}
    c = _client(monkeypatch, mgr)
    r = c.post("/api/training/runs", json={"base_model": "x/Qwen2.5-0.5B",
                                           "dataset_path": "d.jsonl", "steps": 2})
    assert r.status_code == 200 and r.json()["started"] is True


def test_start_run_surfaces_manager_error(monkeypatch):
    mgr = FakeMgr()
    mgr.start = lambda cfg: {"error": "bad config"}
    c = _client(monkeypatch, mgr)
    r = c.post("/api/training/runs", json={"base_model": "", "dataset_path": "", "steps": 2})
    assert r.status_code == 400
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_training_routes.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'routes.training_routes'`).

- [ ] **Step 3: Implement the routes**

Create `routes/training_routes.py`:

```python
"""Admin-gated Local Training API. All heavy work happens in the training
sidecar (a separate CUDA venv); these routes just orchestrate it."""
import asyncio

from fastapi import APIRouter, Body, Depends, HTTPException

from core.middleware import require_admin
from src.training.manager import get_training_manager
from src.training.config import TrainingConfig


def setup_training_routes() -> APIRouter:
    router = APIRouter(prefix="/api/training",
                       dependencies=[Depends(require_admin)])

    @router.get("/env")
    async def env():
        return get_training_manager().env_status()

    @router.post("/env/setup")
    async def env_setup():
        # the install is long (multi-GB); run it off the event loop
        return await asyncio.to_thread(get_training_manager().setup_env)

    @router.post("/runs")
    async def start_run(body: dict = Body(...)):
        try:
            cfg = TrainingConfig(
                base_model=str(body.get("base_model", "")),
                dataset_path=str(body.get("dataset_path", "")),
                lora_r=int(body.get("lora_r", 8)),
                lora_alpha=int(body.get("lora_alpha", 16)),
                lora_dropout=float(body.get("lora_dropout", 0.05)),
                steps=body.get("steps"),
                epochs=body.get("epochs"),
                batch_size=int(body.get("batch_size", 1)),
                learning_rate=float(body.get("learning_rate", 2e-4)),
                max_seq_length=int(body.get("max_seq_length", 512)),
            )
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"invalid config: {e}")
        out = get_training_manager().start(cfg)
        if "error" in out:
            raise HTTPException(400, out["error"])
        return out

    @router.get("/runs/current")
    async def current():
        return get_training_manager().status()

    @router.post("/runs/stop")
    async def stop():
        return get_training_manager().stop()

    @router.get("/adapters")
    async def adapters():
        return {"adapters": get_training_manager().list_adapters()}

    return router
```

Wire it into `app.py` — after the imagemodels router include (`app.py:766-767`, the `from routes.imagemodels_routes import setup_imagemodels_routes; app.include_router(setup_imagemodels_routes())` pair), add:

```python
from routes.training_routes import setup_training_routes
app.include_router(setup_training_routes())
```

- [ ] **Step 4: Run the tests + import smoke**

Run: `python -m pytest tests/test_training_routes.py --import-mode=importlib -q`
Expected: PASS (4 passed).
Run: `python -c "import app"`
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add routes/training_routes.py app.py tests/test_training_routes.py
git commit -m "feat(training): admin /api/training routes + app wiring

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Finalize — engine doc + full suite + no-torch guard

**Files:**
- Create: `docs/training-engine.md`
- Test: (run the whole training suite + `import app` + a torch-not-imported guard)

- [ ] **Step 1: Write the engine doc**

Create `docs/training-engine.md`:

```markdown
# Local LoRA/QLoRA Training Engine

Fine-tune a small base model with QLoRA on your own GPU, entirely locally.
This is the **headless engine** — an admin-gated HTTP API. The Training GUI is
a follow-up sub-project that will consume these same routes.

## How it works

The main app (Python 3.14) cannot run CUDA torch, so training runs in a
separate Python 3.11 CUDA venv, set up on first use via a vendored `uv` binary.
The app orchestrates a sidecar (`training_sidecar/train.py`) as a subprocess
and never imports the training stack itself. One run at a time.

## API (all admin-only, `/api/training`)

- `GET /env` — `{"status": "not_installed" | "ready" | "error"}`.
- `POST /env/setup` — build the training venv (one-time, downloads ~3–4 GB:
  torch cu121 + transformers + peft + bitsandbytes + accelerate + datasets + trl).
- `POST /runs` — start a run. Body:
  `{"base_model": "Qwen/Qwen2.5-0.5B-Instruct", "dataset_path": "C:/data.jsonl",
    "steps": 100}` (or `"epochs"`; optional `lora_r`, `lora_alpha`,
  `lora_dropout`, `batch_size`, `learning_rate`, `max_seq_length`).
- `GET /runs/current` — `{"status", "last_step", "loss", "vram_gb",
  "peak_vram_gb", "error", "output_dir"}`.
- `POST /runs/stop` — kill the current run.
- `GET /adapters` — list produced adapters (`run_id`, `complete`, `base_model`, `path`).

## Dataset format (JSONL, one object per line)

Any of: `{"text": "..."}`, `{"instruction": "...", "response": "..."}`
(optional `"input"`), or `{"prompt": "...", "completion": "..."}`.

## Output

A LoRA adapter under `<DATA_DIR>/training/adapters/<run-id>/`
(`adapter_model.safetensors`, `adapter_config.json`, `run_config.json`).

## Owed (manual, needs the GPU)

- Before packaging: run `python scripts/fetch_uv.py` to vendor `uv` into
  `build_assets/uv/`, then rebuild the frozen exe.
- One real end-to-end run: `POST /env/setup`, then `POST /runs` with a small
  model (e.g. Qwen2.5-0.5B), watch `GET /runs/current`, confirm an adapter lands.
- Frozen boot-check: `resolve_uv_binary()` + `resolve_sidecar_script()` resolve
  in the bundle and the training routes register.

## Not yet (later sub-projects)

The admin Training GUI; merge-LoRA → GGUF + serving the tuned model;
full-fp16 (non-quantized) LoRA; multi-GPU; eval/checkpoint-resume.
```

- [ ] **Step 2: Run the full training test suite**

Run: `python -m pytest tests/test_training_runtime.py tests/test_training_config.py tests/test_training_dataset.py tests/test_training_env.py tests/test_training_manager.py tests/test_training_routes.py --import-mode=importlib -q`
Expected: PASS (all training tests green).

- [ ] **Step 3: Import smoke + no-torch guard**

Run: `python -c "import app, sys; assert 'torch' not in sys.modules, 'app must not import torch'; print('OK - app imports, torch not loaded')"`
Expected: `OK - app imports, torch not loaded` (proves the main app never pulls in the training stack).

- [ ] **Step 4: Commit**

```bash
git add docs/training-engine.md
git commit -m "docs(training): local LoRA/QLoRA training engine API + usage

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **The Py3.14 app must never import the training stack.** Only `training_sidecar/train.py` imports torch/peft — and it runs in the venv, is never imported by the app, and has no Py3.14 unit test (Task 5 syntax-checks it; the spike + a real run prove it). Task 8 asserts `torch` isn't in `sys.modules` after `import app`.
- **`uv` + `training_sidecar` must be vendored/bundled** (Task 1). Before packaging, run `python scripts/fetch_uv.py` so `build_assets/uv/uv.exe` exists for the build. Do NOT commit that binary (build-time artifact, like the llama/sd binaries).
- **Never-raises** on the manager/env/routes is load-bearing — a broken training env or a dying sidecar must degrade to an error status, never crash the app.
- **Owed by the user (manual, needs the GPU):** `python scripts/fetch_uv.py`; rebuild the frozen exe; then via the API: `POST /api/training/env/setup` (one-time ~3–4 GB download) and `POST /api/training/runs` for a tiny adapter end-to-end. A frozen boot-check confirms `resolve_uv_binary()` + `resolve_sidecar_script()` resolve in the bundle and the training routes register.
- **Scope:** headless engine + on-demand env + admin API only. NO GUI (next sub-project), NO merge-to-GGUF / serving the tuned model, NO full-fp16 LoRA, NO multi-GPU, NO dataset-creation tools.
