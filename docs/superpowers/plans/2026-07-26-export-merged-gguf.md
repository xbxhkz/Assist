# Export a Fine-Tuned Adapter as a Standalone GGUF — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From a trained LoRA adapter, produce one self-contained, quantized GGUF (for use outside Assist) — merge the adapter into the base, convert the full model to GGUF, and quantize — driven from the admin Adapters list.

**Architecture:** Merge + full-model convert run in the reused Python-3.11 CUDA sidecar venv (the app never imports the stack); quantization is the bundled `llama-quantize.exe` the app runs like `llama-server`. A never-raises `AdapterExporter` orchestrates both and cleans up the multi-GB temporaries. Frontend adds an Export control to the existing panel.

**Tech Stack:** Python 3.14 (app) + the Py3.11 CUDA sidecar venv (adds `sentencepiece`; vendored llama.cpp `convert_hf_to_gguf.py` reusing the already-vendored `conversion/` + `gguf/`); the bundled `llama-quantize.exe` (release `b9867`); FastAPI; vanilla ES-module frontend.

## Global Constraints

- **Feasibility is PROVEN GO** (spike ran merge → `convert_hf_to_gguf` → `llama-quantize` → serve on this machine; the standalone Q4_K_M GGUF generated coherent text with no `--lora`). No feasibility-gate task. Confirmed values: `PeftModel.from_pretrained(base, adapter).merge_and_unload()`; `convert_hf_to_gguf.py` needs **`sentencepiece`**; `llama-quantize.exe <f16> <out> <QUANT>` from release **`b9867`**. Sizes (0.5B): merged HF 988 MB → F16 994 MB → Q4_K_M 398 MB.
- **The main app (Py3.14) MUST NOT import torch/peft/transformers/sentencepiece/gguf or the merge/convert scripts.** They run ONLY in the sidecar venv via `training_sidecar/merge.py` (spawned, never imported).
- **Managers/routes NEVER raise** — degrade to `{"error": …}` / an error status.
- **Admin-only** — the export route is on the admin-gated `/api/training` router.
- **Reuse sub-project 1/3 infra:** the training sidecar venv (`TrainingEnv`), the vendored `conversion/` + `gguf/` packages, `Assist.spec`'s `training_sidecar/` bundling, and the bundled llama binaries. Add ONLY `sentencepiece` (STACK), `convert_hf_to_gguf.py` (vendored), and `llama-quantize.exe` (fetched).
- **Clean up temporaries** (merged HF dir, F16 GGUF) on BOTH success and failure — a failed export must not strand multi-GB files.
- **Quant set:** exactly `Q4_K_M` (default), `Q5_K_M`, `Q8_0`, `F16`. `F16` skips quantize (the F16 GGUF is the output).
- `build_assets/` is gitignored — `llama-quantize.exe` is a build-time artifact (fetched, never committed), like `llama-server.exe`.
- pytest `--import-mode=importlib`. Node on PATH for JS tests. Commit directly to `dev`; messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Stage only the files each task names — never `git add -A`; never stage `installer/Output/Assist-Setup.exe`, `assistappicon.png`, `assistlogo.png`, `build_assets/**`.
- ~220 unrelated pre-existing test failures exist elsewhere — run only the test files each task names. (A pre-existing SQLAlchemy `MovedIn20Warning` at collection is known.)

### Verified anchors
- `src/localmodels/runtime.py resolve_llama_binary(device="cpu", path_lookup=shutil.which, frozen_base=None, dev_base=None)` — bundled layout `<base>/llama/<cpu|vulkan>/<name>` (frozen `sys._MEIPASS`), dev `<repo>/build_assets/llama/<sub>/<name>`.
- `scripts/fetch_llama_server.py` extract filter: `if base == "llama-server.exe" or base.lower().endswith(".dll"):` (~line 42), `dest` is `build_assets/llama/<kind>`.
- `src/training/env.py:11 STACK`; `training_sidecar/` already holds `conversion/` + `gguf/` + `convert_lora_to_gguf.py` (from `c0bc859`); `resolve_convert_script`/`resolve_sidecar_script` pattern in `src/training/runtime.py`.
- `src/training/convert_manager.py AdapterConverter` (injectable spawn, blocking, never-raises) + `get_adapter_converter()` — the pattern to mirror.
- `routes/training_routes.py setup_training_routes()` (admin-gated); `os` imported; consumes `get_training_manager().list_adapters()` (each has `run_id`, `base_model`, `path`).
- `src/constants.py:12 DATA_DIR`, `:60 MODELS_DIR`. `src/desktop/apps.py launch(target, startfile=None)` uses `os.startfile`.
- The gate left a checkout at `C:\tmp\serve_spike\llama.cpp` (commit `c0bc859`) — vendor `convert_hf_to_gguf.py` from there.

---

### Task 1: Vendor `llama-quantize.exe` + `resolve_quantize_binary`

**Files:**
- Modify: `scripts/fetch_llama_server.py`, `src/localmodels/runtime.py`
- Test: `tests/test_quantize_binary.py`

**Interfaces:**
- Produces: `resolve_quantize_binary(device="cpu", frozen_base=None, dev_base=None) -> str` (path to `llama-quantize.exe`; raises `RuntimeError` if not found).

- [ ] **Step 1: Write the failing test**

Create `tests/test_quantize_binary.py`:

```python
import os
import pytest
from src.localmodels.runtime import resolve_quantize_binary


def test_resolves_bundled_frozen(tmp_path):
    d = tmp_path / "llama" / "cpu"
    d.mkdir(parents=True)
    exe = d / ("llama-quantize.exe" if os.name == "nt" else "llama-quantize")
    exe.write_text("x")
    assert resolve_quantize_binary(device="cpu", frozen_base=str(tmp_path)) == str(exe)


def test_resolves_dev(tmp_path):
    d = tmp_path / "vulkan"
    d.mkdir(parents=True)
    exe = d / ("llama-quantize.exe" if os.name == "nt" else "llama-quantize")
    exe.write_text("x")
    assert resolve_quantize_binary(device="gpu", frozen_base=str(tmp_path / "none"),
                                   dev_base=str(tmp_path)) == str(exe)


def test_raises_when_missing(tmp_path):
    with pytest.raises(RuntimeError):
        resolve_quantize_binary(frozen_base=str(tmp_path / "a"), dev_base=str(tmp_path / "b"))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_quantize_binary.py --import-mode=importlib -q`
Expected: FAIL (`cannot import name 'resolve_quantize_binary'`).

- [ ] **Step 3: Add the resolver + fetch extraction**

In `scripts/fetch_llama_server.py`, change the extract filter line to also take the quantize exe:

```python
            if base in ("llama-server.exe", "llama-quantize.exe") or base.lower().endswith(".dll"):
```

In `src/localmodels/runtime.py`, add after `resolve_llama_binary` (it already imports `os`, `sys`):

```python
def resolve_quantize_binary(device: str = "cpu", frozen_base: str = None,
                            dev_base: str = None) -> str:
    """Resolve the bundled llama-quantize executable. Mirrors resolve_llama_binary's
    bundled-layout resolution (`<base>/llama/<cpu|vulkan>/<name>`, dev fallback under
    build_assets/llama) but does NO PATH lookup — the quantizer must match the bundled
    ggml/quantize DLLs. Raises RuntimeError if not found."""
    sub = "vulkan" if device == "gpu" else "cpu"
    name = "llama-quantize.exe" if os.name == "nt" else "llama-quantize"
    base = frozen_base
    if base is None and getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None)
    if base:
        for rel in (os.path.join("llama", sub, name), os.path.join("llama", name)):
            cand = os.path.join(base, rel)
            if os.path.isfile(cand):
                return cand
    if dev_base is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dev_base = os.path.join(repo_root, "build_assets", "llama")
    for cand in (os.path.join(dev_base, sub, name), os.path.join(dev_base, name)):
        if os.path.isfile(cand):
            return cand
    raise RuntimeError("llama-quantize not found under the bundled/dev llama dir "
                       "(run scripts/fetch_llama_server.py).")
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_quantize_binary.py --import-mode=importlib -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_llama_server.py src/localmodels/runtime.py tests/test_quantize_binary.py
git commit -m "feat(export-gguf): fetch + resolve llama-quantize.exe

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Vendor `convert_hf_to_gguf.py` + `sentencepiece` + `merge.py` sidecar

**Files:**
- Modify: `src/training/env.py` (`STACK`)
- Create: `training_sidecar/merge.py`; vendor `training_sidecar/convert_hf_to_gguf.py` (from `c0bc859`)
- Test: `tests/test_merge_sidecar_syntax.py`

**Interfaces:**
- Produces: a sidecar `python merge.py --adapter <dir> --outfile <f16.gguf> [--base <id>]` that merges the adapter into its base, writes an **F16 GGUF**, and emits one JSON line (`{"event":"done","f16_gguf":<path>}` / `{"event":"error","message":<str>}`).

**IMPORTANT:** `merge.py` + `convert_hf_to_gguf.py` import torch/transformers/gguf — NOT importable by the Py3.14 app/suite. Only an `ast.parse` gate covers `merge.py`; the vendored upstream file is third-party (not parse-gated). `merge.py` SPAWNS `convert_hf_to_gguf.py` (does not import it); its `conversion/`+`gguf/` deps are already beside it in `training_sidecar/`. `Assist.spec` bundles `training_sidecar/` wholesale.

- [ ] **Step 1: Add `sentencepiece` to STACK**

In `src/training/env.py`, change the `STACK` line (full-model vocab conversion needs it):

```python
STACK = ["transformers", "peft", "bitsandbytes", "accelerate", "datasets", "trl", "sentencepiece"]
```

- [ ] **Step 2: Vendor `convert_hf_to_gguf.py`**

Copy `C:\tmp\serve_spike\llama.cpp\convert_hf_to_gguf.py` → `training_sidecar\convert_hf_to_gguf.py` (do not edit; it reuses the already-vendored `conversion/` + `gguf/`).

- [ ] **Step 3: Write `merge.py`**

Create `training_sidecar/merge.py`:

```python
"""Merge a LoRA adapter into its base and export a full F16 GGUF. Runs INSIDE the
Py3.11 CUDA venv (never imported by the Py3.14 app). Emits one JSON line to stdout:
  {"event":"done","f16_gguf":<path>}  |  {"event":"error","message":<str>}
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--base", default=None)
    args = ap.parse_args()
    tmp = None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base = args.base
        if not base:
            with open(os.path.join(args.adapter, "adapter_config.json"), "r", encoding="utf-8") as f:
                base = json.load(f).get("base_model_name_or_path")
        if not base:
            emit({"event": "error", "message": "base model id not found in adapter_config.json"})
            sys.exit(1)

        tmp = tempfile.mkdtemp(prefix="assist_merge_")
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16)
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        model.save_pretrained(tmp)
        AutoTokenizer.from_pretrained(base).save_pretrained(tmp)

        argv = [sys.executable, os.path.join(HERE, "convert_hf_to_gguf.py"), tmp,
                "--outfile", args.outfile, "--outtype", "f16"]
        p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0 or not os.path.isfile(args.outfile):
            tail = ((p.stdout or "") + (p.stderr or ""))[-1500:]
            emit({"event": "error", "message": "convert_hf_to_gguf failed: " + tail})
            sys.exit(1)
        emit({"event": "done", "f16_gguf": args.outfile})
    except Exception as e:  # noqa: BLE001
        try:
            emit({"event": "error", "message": f"{e}"})
        except Exception:
            pass
        sys.exit(1)
    finally:
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Syntax-gate `merge.py`**

Create `tests/test_merge_sidecar_syntax.py`:

```python
import ast
import pathlib


def test_merge_py_parses():
    p = pathlib.Path(__file__).resolve().parents[1] / "training_sidecar" / "merge.py"
    ast.parse(p.read_text(encoding="utf-8"))  # parse only; never import (needs torch/gguf)
```

Run: `python -m pytest tests/test_merge_sidecar_syntax.py --import-mode=importlib -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/training/env.py training_sidecar/merge.py training_sidecar/convert_hf_to_gguf.py tests/test_merge_sidecar_syntax.py
git commit -m "feat(export-gguf): merge.py sidecar + vendor convert_hf_to_gguf + sentencepiece

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `AdapterExporter` orchestration

**Files:**
- Create: `src/training/export_manager.py`
- Test: `tests/test_export_manager.py`

**Interfaces:**
- Consumes: `TrainingEnv` (env.py), `resolve_convert_script`-style resolution (a new `resolve_merge_script` in `src/training/runtime.py`, mirroring `resolve_convert_script`), `resolve_quantize_binary` (Task 1), `DATA_DIR`.
- Produces: `VALID_QUANTS = ("Q4_K_M", "Q5_K_M", "Q8_0", "F16")`; `AdapterExporter(env=None, spawn=None, exports_dir=None, quantize_resolver=None)` with `export(adapter_dir, quant, base_model=None) -> dict` (`{"ok":True,"gguf":path}` | `{"error":str}`; blocking; **never raises**) and `get_adapter_exporter()`. `spawn(argv) -> (returncode, stdout)` injectable.

- [ ] **Step 1: Write the failing test**

Create `tests/test_export_manager.py`:

```python
import json
import os
from src.training.export_manager import AdapterExporter, VALID_QUANTS


class FakeEnv:
    def __init__(self, ready=True): self._ready = ready
    def ensure_ready(self, progress=None):
        return {"ready": self._ready, "error": None if self._ready else "no env"}
    def venv_python(self): return "venv/python"


def _adapter(tmp_path):
    d = tmp_path / "run-1"; d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"}))
    (d / "run_config.json").write_text(json.dumps({"base_model": "Qwen/Qwen2.5-0.5B-Instruct"}))
    return str(d)


def test_export_quantized_runs_merge_then_quantize_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.export_manager.resolve_merge_script", lambda: "merge.py")
    exports = tmp_path / "exports"
    calls = []

    def spawn(argv):
        calls.append(argv)
        if argv[1].endswith("merge.py"):
            f16 = argv[argv.index("--outfile") + 1]
            open(f16, "w").close()  # sidecar writes the F16 GGUF
            return (0, json.dumps({"event": "done", "f16_gguf": f16}))
        # llama-quantize: argv = [quantize_exe, <f16>, <out>, <QUANT>]
        open(argv[2], "w").close()
        return (0, "quantized")

    exp = AdapterExporter(env=FakeEnv(), spawn=spawn, exports_dir=str(exports),
                          quantize_resolver=lambda device="cpu": "llama-quantize.exe")
    out = exp.export(_adapter(tmp_path), "Q4_K_M")
    assert out.get("ok") is True and out["gguf"].endswith("Q4_K_M.gguf")
    assert os.path.isfile(out["gguf"])
    # merge ran, then quantize ran
    assert calls[0][1].endswith("merge.py") and calls[1][0] == "llama-quantize.exe" and calls[1][3] == "Q4_K_M"
    # the F16 intermediate was cleaned up
    f16 = calls[0][calls[0].index("--outfile") + 1]
    assert not os.path.isfile(f16)


def test_f16_export_skips_quantize(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.export_manager.resolve_merge_script", lambda: "merge.py")
    def spawn(argv):
        f16 = argv[argv.index("--outfile") + 1]; open(f16, "w").close()
        return (0, json.dumps({"event": "done", "f16_gguf": f16}))
    exp = AdapterExporter(env=FakeEnv(), spawn=spawn, exports_dir=str(tmp_path / "e"),
                          quantize_resolver=lambda device="cpu": "q.exe")
    out = exp.export(_adapter(tmp_path), "F16")
    assert out.get("ok") is True and out["gguf"].endswith("F16.gguf") and os.path.isfile(out["gguf"])


def test_invalid_quant_rejected(tmp_path):
    exp = AdapterExporter(env=FakeEnv(), spawn=lambda a: (0, ""), exports_dir=str(tmp_path))
    out = exp.export(_adapter(tmp_path), "Q3_K_S")
    assert "error" in out and "quant" in out["error"].lower()


def test_env_not_ready_errors(tmp_path):
    exp = AdapterExporter(env=FakeEnv(ready=False), spawn=lambda a: (0, ""), exports_dir=str(tmp_path))
    out = exp.export(_adapter(tmp_path), "Q4_K_M")
    assert "error" in out and "env" in out["error"].lower()


def test_merge_error_surfaced_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.export_manager.resolve_merge_script", lambda: "merge.py")
    def spawn(argv): return (1, json.dumps({"event": "error", "message": "arch unsupported"}))
    exp = AdapterExporter(env=FakeEnv(), spawn=spawn, exports_dir=str(tmp_path / "e"),
                          quantize_resolver=lambda device="cpu": "q.exe")
    out = exp.export(_adapter(tmp_path), "Q4_K_M")
    assert "error" in out and "arch unsupported" in out["error"]


def test_never_raises_on_spawn_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.export_manager.resolve_merge_script", lambda: "merge.py")
    def boom(argv): raise RuntimeError("cannot spawn")
    exp = AdapterExporter(env=FakeEnv(), spawn=boom, exports_dir=str(tmp_path / "e"),
                          quantize_resolver=lambda device="cpu": "q.exe")
    assert "error" in exp.export(_adapter(tmp_path), "Q4_K_M")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_export_manager.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.training.export_manager'`).

- [ ] **Step 3: Add `resolve_merge_script` + implement `AdapterExporter`**

In `src/training/runtime.py`, add (mirrors `resolve_convert_script`):

```python
def resolve_merge_script(frozen_base=None, dev_base=None) -> str:
    """Path to training_sidecar/merge.py. Raises RuntimeError if missing."""
    base = _frozen_base(frozen_base)
    if base:
        cand = os.path.join(base, "training_sidecar", "merge.py")
        if os.path.isfile(cand):
            return cand
    if dev_base is None:
        dev_base = os.path.join(_REPO_ROOT, "training_sidecar")
    cand = os.path.join(dev_base, "merge.py")
    if os.path.isfile(cand):
        return cand
    raise RuntimeError("training_sidecar/merge.py not found.")
```

Create `src/training/export_manager.py`:

```python
"""Orchestrate merge->convert->quantize export of a tuned adapter to a standalone
GGUF. Never imports the merge stack; never raises. Blocking (call via asyncio.to_thread)."""
import json
import os
import re
import subprocess

from src.training.runtime import resolve_merge_script

VALID_QUANTS = ("Q4_K_M", "Q5_K_M", "Q8_0", "F16")


def _default_spawn(argv):
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s or "model").rsplit("/", 1)[-1]).strip("-") or "model"


class AdapterExporter:
    def __init__(self, env=None, spawn=None, exports_dir=None, quantize_resolver=None):
        if env is None:
            from src.training.env import TrainingEnv
            env = TrainingEnv()
        if exports_dir is None:
            from src.constants import DATA_DIR
            exports_dir = os.path.join(DATA_DIR, "training", "exports")
        if quantize_resolver is None:
            from src.localmodels.runtime import resolve_quantize_binary
            quantize_resolver = resolve_quantize_binary
        self._env = env
        self._spawn = spawn or _default_spawn
        self._exports_dir = exports_dir
        self._resolve_quant = quantize_resolver

    def export(self, adapter_dir, quant, base_model=None) -> dict:
        f16 = None
        try:
            if quant not in VALID_QUANTS:
                return {"error": f"invalid quant '{quant}' (allowed: {', '.join(VALID_QUANTS)})"}
            if not (isinstance(adapter_dir, str) and os.path.isdir(adapter_dir)):
                return {"error": "adapter directory not found"}
            ready = self._env.ensure_ready()
            if not ready.get("ready"):
                return {"error": f"training env not ready: {ready.get('error')}"}

            os.makedirs(self._exports_dir, exist_ok=True)
            run_id = os.path.basename(adapter_dir.rstrip("/\\"))
            if not base_model:
                base_model = self._read_base(adapter_dir)
            out_name = f"{_slug(base_model)}-{run_id}-{quant}.gguf"
            final = os.path.join(self._exports_dir, out_name)
            f16 = os.path.join(self._exports_dir, f".{run_id}.f16.gguf")

            # 1) merge + convert -> F16 GGUF (in the venv)
            argv = [self._env.venv_python(), resolve_merge_script(),
                    "--adapter", adapter_dir, "--outfile", f16]
            if base_model:
                argv += ["--base", base_model]
            rc, out = self._spawn(argv)
            ev = _last_json(out)
            if ev.get("event") == "error":
                return {"error": ev.get("message", "merge failed")}
            if not os.path.isfile(f16):
                return {"error": "merge/convert failed: " + (out or "")[-500:]}

            # 2) quantize (or, for F16, the F16 GGUF is the output)
            if quant == "F16":
                os.replace(f16, final)
                f16 = None
                return {"ok": True, "gguf": final}
            qexe = self._resolve_quant()
            rc, out = self._spawn([qexe, f16, final, quant])
            if rc != 0 or not os.path.isfile(final):
                return {"error": "quantize failed: " + (out or "")[-500:]}
            return {"ok": True, "gguf": final}
        except Exception as e:  # noqa: BLE001
            return {"error": f"export error: {e}"}
        finally:
            if f16 and os.path.isfile(f16):
                try:
                    os.remove(f16)
                except Exception:
                    pass

    def _read_base(self, adapter_dir):
        for name, key in (("adapter_config.json", "base_model_name_or_path"),
                          ("run_config.json", "base_model")):
            p = os.path.join(adapter_dir, name)
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        v = json.load(f).get(key)
                    if v:
                        return v
                except Exception:
                    pass
        return None

    def list_exports(self, run_id) -> list:
        out = []
        if not os.path.isdir(self._exports_dir):
            return out
        for fn in sorted(os.listdir(self._exports_dir)):
            if fn.endswith(".gguf") and f"-{run_id}-" in fn:
                out.append(os.path.join(self._exports_dir, fn))
        return out

    def exports_dir(self):
        return self._exports_dir


def _last_json(text):
    ev = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                ev = json.loads(line)
            except Exception:
                pass
    return ev


_exporter = None


def get_adapter_exporter():
    global _exporter
    if _exporter is None:
        _exporter = AdapterExporter()
    return _exporter
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_export_manager.py tests/test_training_runtime.py --import-mode=importlib -q`
Expected: PASS (6 export + the runtime tests).

- [ ] **Step 5: Commit**

```bash
git add src/training/runtime.py src/training/export_manager.py tests/test_export_manager.py
git commit -m "feat(export-gguf): AdapterExporter (merge->quantize, cleanup, never-raises)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Admin export route + reveal + status

**Files:**
- Modify: `routes/training_routes.py`
- Test: `tests/test_export_routes.py`

**Interfaces:**
- Consumes: `get_adapter_exporter` + `VALID_QUANTS` (Task 3), `get_training_manager().list_adapters()`.
- Produces: `POST /api/training/adapters/{run_id}/export` (body `{quant}`), `POST /api/training/exports/reveal`, and `GET /api/training/adapters/{run_id}` gains `exports` (list of exported file paths).

- [ ] **Step 1: Write the failing test**

Create `tests/test_export_routes.py`:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.training_routes as tr


class FakeMgr:
    def list_adapters(self):
        return [{"run_id": "run-1", "complete": True, "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
                 "path": "p", "converted": False, "adapter_gguf": None}]


def _client(monkeypatch, exporter):
    monkeypatch.setattr(tr, "require_admin", lambda: None)
    monkeypatch.setattr(tr, "get_training_manager", lambda: FakeMgr())
    monkeypatch.setattr(tr, "get_adapter_exporter", lambda: exporter)
    app = FastAPI(); app.include_router(tr.setup_training_routes())
    return TestClient(app)


class Exp:
    def __init__(self, result): self._r = result
    def export(self, adapter_dir, quant, base_model=None): self.seen = (adapter_dir, quant); return self._r
    def list_exports(self, run_id): return ["p/exports/x.gguf"]
    def exports_dir(self): return "p/exports"


def test_export_ok(monkeypatch):
    e = Exp({"ok": True, "gguf": "p/exports/Qwen-run-1-Q4_K_M.gguf"})
    c = _client(monkeypatch, e)
    r = c.post("/api/training/adapters/run-1/export", json={"quant": "Q4_K_M"})
    assert r.status_code == 200 and r.json()["ok"] is True and e.seen[1] == "Q4_K_M"


def test_export_invalid_quant_400(monkeypatch):
    c = _client(monkeypatch, Exp({"ok": True}))
    r = c.post("/api/training/adapters/run-1/export", json={"quant": "BOGUS"})
    assert r.status_code == 400


def test_export_manager_error_400(monkeypatch):
    c = _client(monkeypatch, Exp({"error": "quantize failed"}))
    r = c.post("/api/training/adapters/run-1/export", json={"quant": "Q4_K_M"})
    assert r.status_code == 400


def test_adapter_status_lists_exports(monkeypatch):
    c = _client(monkeypatch, Exp({"ok": True}))
    r = c.get("/api/training/adapters/run-1")
    assert r.status_code == 200 and r.json()["exports"] == ["p/exports/x.gguf"]
```

Note: `GET /api/training/adapters/{run_id}` already exists (serve-adapter feature) and returns `base_match`; this task ADDS `exports` to its response. If the serve-adapter feature isn't present, add the `exports` key to whatever that route returns.

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_export_routes.py --import-mode=importlib -q`
Expected: FAIL (`get_adapter_exporter` attr / route missing).

- [ ] **Step 3: Implement**

In `routes/training_routes.py`, add imports at the top:

```python
from src.training.export_manager import get_adapter_exporter, VALID_QUANTS
```

In the existing `adapter_status` route (`GET /adapters/{run_id}`), add the exports list to its returned dict — change its `return {**a, "base_match": match}` to:

```python
        return {**a, "base_match": match, "exports": get_adapter_exporter().list_exports(run_id)}
```

Add these routes inside `setup_training_routes()` (after the serve route):

```python
    @router.post("/adapters/{run_id}/export")
    async def export_adapter(run_id: str, body: dict = Body(...)):
        a = _find_adapter(run_id)
        if not a:
            raise HTTPException(404, "adapter not found")
        quant = str(body.get("quant", "Q4_K_M"))
        if quant not in VALID_QUANTS:
            raise HTTPException(400, f"invalid quant (allowed: {', '.join(VALID_QUANTS)})")
        out = await asyncio.to_thread(get_adapter_exporter().export, a["path"], quant, a.get("base_model"))
        if "error" in out:
            raise HTTPException(400, out["error"])
        return out

    @router.post("/exports/reveal")
    async def reveal_exports():
        d = get_adapter_exporter().exports_dir()
        try:
            os.makedirs(d, exist_ok=True)
            sf = getattr(os, "startfile", None)
            if sf:
                sf(d)  # open the exports folder in the OS file manager (Windows)
            return {"ok": True, "path": d}
        except Exception as e:
            return {"ok": False, "path": d, "error": str(e)}
```

(`_find_adapter`, `asyncio`, `os`, `Body`, `HTTPException` already exist in the module.)

- [ ] **Step 4: Run the tests + import smoke**

Run: `python -m pytest tests/test_export_routes.py tests/test_serve_adapter_routes.py tests/test_training_routes.py --import-mode=importlib -q`
Expected: PASS.
Run: `python -c "import app"`
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add routes/training_routes.py tests/test_export_routes.py
git commit -m "feat(export-gguf): admin export + reveal routes; adapter status lists exports

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: GUI — Export control

**Files:**
- Create: `static/js/exportCore.js`
- Modify: `static/js/training.js`
- Test: `tests/test_export_core_js.py`, `tests/test_training_ui.py` (extend)

**Interfaces:**
- Consumes: `/api/training/adapters/{run_id}/export`, `/api/training/exports/reveal`.
- Produces: `exportCore.js` exports `EXPORT_QUANTS` (array) and `exportButtonState(adapter) -> {canExport}`; `training.js` renders a per-adapter quant `<select>` + **Export GGUF** button + **Open folder**.

- [ ] **Step 1: Write the failing test (pure core)**

Create `tests/test_export_core_js.py`:

```python
import json, subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "static" / "js" / "exportCore.js"


def _node(expr):
    s = f"import * as m from {json.dumps(CORE.as_uri())};\n{expr}"
    p = subprocess.run(["node", "--input-type=module", "-e", s],
                       capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def test_quants_default_first():
    out = _node("console.log(JSON.stringify(m.EXPORT_QUANTS));")
    q = json.loads(out)
    assert q[0] == "Q4_K_M" and set(q) == {"Q4_K_M", "Q5_K_M", "Q8_0", "F16"}


def test_export_button_state():
    out = _node("console.log(JSON.stringify(["
                "m.exportButtonState({complete:true}).canExport,"
                "m.exportButtonState({complete:false}).canExport]));")
    assert json.loads(out) == [True, False]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_export_core_js.py --import-mode=importlib -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `exportCore.js`**

Create `static/js/exportCore.js`:

```javascript
// Pure helpers for the adapter Export control — no DOM.
export const EXPORT_QUANTS = ['Q4_K_M', 'Q5_K_M', 'Q8_0', 'F16'];

export function exportButtonState(a) {
  return { canExport: !!(a && a.complete) };
}
```

- [ ] **Step 4: Wire the Export control into `training.js`**

In `static/js/training.js`, add the import next to the serveCore import:

```javascript
import { EXPORT_QUANTS, exportButtonState } from './exportCore.js';
```

In `refreshAdapters`, extend each adapter row to include an export control (a quant `<select>` + Export button) when `exportButtonState(a).canExport`. Inside the row-building `.map(...)`, after the existing `btns` string, add:

```javascript
      const q = exportButtonState(a).canExport
        ? '<select data-quant="' + esc(a.run_id) + '">' +
          EXPORT_QUANTS.map(function (x) { return '<option>' + x + '</option>'; }).join('') +
          '</select><button class="btn" data-export="' + esc(a.run_id) + '">Export GGUF</button>' +
          '<button class="btn" data-reveal="1">Open folder</button>'
        : '';
```

and include `q` in the returned row HTML (e.g. `... + btns + ' ' + q + '</div>'`). After setting `host.innerHTML`, wire the handlers (alongside the existing `data-conv`/`data-serve` wiring):

```javascript
    host.querySelectorAll('[data-export]').forEach(function (b) {
      b.addEventListener('click', function () { exportAdapter(b.getAttribute('data-export')); });
    });
    host.querySelectorAll('[data-reveal]').forEach(function (b) {
      b.addEventListener('click', function () { api('/api/training/exports/reveal', { method: 'POST' }).catch(function () {}); });
    });
```

Add the handler function (next to `convertAdapter`/`serveAdapter`):

```javascript
async function exportAdapter(runId) {
  const host = $('training-adapters');
  const sel = host && host.querySelector('[data-quant="' + runId + '"]');
  const quant = sel ? sel.value : 'Q4_K_M';
  try {
    if (host) host.textContent = 'Exporting ' + runId + ' as ' + quant + ' (merge + convert + quantize)…';
    const r = await api('/api/training/adapters/' + encodeURIComponent(runId) + '/export', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ quant: quant }),
    });
    alert('Exported: ' + (r.gguf || '(done)'));
  } catch (e) { alert('Export failed: ' + e.message); }
  refreshAdapters();
}
```

- [ ] **Step 5: Extend the UI test**

Append to `tests/test_training_ui.py`:

```python
def test_training_js_references_export():
    src = (ROOT / "static" / "js" / "training.js").read_text(encoding="utf-8")
    assert "/export" in src and "exportCore.js" in src and "exports/reveal" in src
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_export_core_js.py tests/test_training_ui.py --import-mode=importlib -q`
Expected: PASS (the `node --check` gate in `test_training_ui.py` confirms `training.js` still parses with the new import + handlers).

- [ ] **Step 7: Commit**

```bash
git add static/js/exportCore.js static/js/training.js tests/test_export_core_js.py tests/test_training_ui.py
git commit -m "feat(export-gguf): GUI Export control (quant select + Open folder)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Finalize — docs + full suite + no-torch guard

**Files:**
- Modify: `docs/serve-adapter.md` (or create `docs/export-gguf.md`); `docs/training-gui-manual-verification.md`
- Test: (run the whole export + training suite + `import app` guard)

- [ ] **Step 1: Document the flow**

Create `docs/export-gguf.md`:

```markdown
# Exporting a Fine-Tuned Adapter as a Standalone GGUF

After training a LoRA adapter, export a self-contained GGUF to use outside Assist
(LM Studio, Ollama, sharing):

1. In the Training panel's Adapters list, pick a **quant** (Q4_K_M default, Q5_K_M,
   Q8_0, or F16) and click **Export GGUF**.
2. Assist merges the adapter into the base (in the training sidecar venv), converts
   the full model to GGUF, and quantizes it. The result lands in
   `<DATA_DIR>/training/exports/<base>-<run-id>-<quant>.gguf`.
3. Click **Open folder** to reveal it, then copy/share the file.

## API (admin, /api/training)
- `POST /adapters/{run_id}/export` (body `{quant}`) → merge + convert + quantize.
- `POST /exports/reveal` → open the exports folder.
- `GET /adapters/{run_id}` → also lists this adapter's `exports`.

## Notes
- The F16 intermediate (~2× the model) is deleted after quantizing; only the final
  GGUF is kept. Large models need proportional free disk.
- Proven end-to-end (spike): merge → convert_hf_to_gguf → llama-quantize → the
  standalone GGUF serves with no `--lora`. Owed: one real GUI export on the 6 GB GPU.

## Not yet
Auto-registering the export as an in-app Local Model (drop it into the models dir to
serve it); imatrix / per-tensor quant; GGUF splitting.
```

Add a bullet to `docs/training-gui-manual-verification.md` under the runbook: export an adapter (Q4_K_M), confirm the GGUF lands in `training/exports/`, **Open folder** reveals it, and it loads in an external tool.

- [ ] **Step 2: Run the full export + training suite**

Run: `python -m pytest tests/test_quantize_binary.py tests/test_merge_sidecar_syntax.py tests/test_export_manager.py tests/test_export_routes.py tests/test_export_core_js.py tests/test_training_ui.py tests/test_training_routes.py tests/test_serve_adapter_routes.py tests/test_convert_manager.py tests/test_training_env.py tests/test_training_runtime.py --import-mode=importlib -q`
Expected: PASS (all green).

- [ ] **Step 3: Import smoke + no-merge-stack guard**

Run: `python -c "import app, sys; bad=[m for m in ('torch','peft','transformers','sentencepiece','gguf') if m in sys.modules]; assert not bad, bad; print('OK - app free of the merge stack')"`
Expected: `OK - app free of the merge stack`.

- [ ] **Step 4: Commit**

```bash
git add docs/export-gguf.md docs/training-gui-manual-verification.md
git commit -m "docs(export-gguf): export-a-standalone-GGUF usage + manual verification

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Feasibility is already GO** (spike). The confirmed tooling is pinned in Global Constraints; the vendor source is the checkout at `C:\tmp\serve_spike\llama.cpp` (commit `c0bc859`).
- **Before packaging:** re-run `python scripts/fetch_llama_server.py` so `llama-quantize.exe` lands in `build_assets/llama/{cpu,vulkan}/` (build-time artifact, not committed). A clean frozen rebuild is needed for the sidecar `merge.py`/`convert_hf_to_gguf.py` + the quantize binary to ship.
- **The Py3.14 app never imports the merge stack.** Only `training_sidecar/merge.py` + `convert_hf_to_gguf.py` import torch/transformers/gguf; they run in the sidecar venv. Task 6 asserts none leak into `import app`.
- **Never-raises + temp cleanup** are load-bearing: a failed export must surface an error AND not strand the merged-HF/F16 temporaries.
- **Owed by the user (manual, GPU):** one real GUI export on the 6 GB GPU (train → Export → confirm the GGUF loads externally); a frozen boot-check that `resolve_quantize_binary()`/`resolve_merge_script()` resolve and the export routes register.
- **Scope:** export a standalone quantized GGUF only. NO auto in-app registration/serving (serve-adapter already covers in-app), NO imatrix/per-tensor quant, NO GGUF split, NO destination picker.
