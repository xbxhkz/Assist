# Serve a Fine-Tuned Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve a locally-trained LoRA adapter in-app — convert the peft adapter to a GGUF LoRA (in the training sidecar venv), pair it with a matching base GGUF, and serve it through the existing `llama-server` path with `--lora`, so the tuned model is selectable and chattable like any local model.

**Architecture:** Conversion runs in the reused Python-3.11 CUDA sidecar venv (the Py3.14 app never imports the conversion stack); serving is the bundled `llama-server` with one added `--lora` flag. Base↔adapter pairing is auto-matched by name with a pick/download fallback. Frontend controls are additive to the existing Training/Adapters panel.

**Tech Stack:** Python 3.14 (app) + the Py3.11 CUDA sidecar venv (adds `gguf==0.19.0` + vendored llama.cpp `convert_lora_to_gguf.py` and its `conversion/` package, commit `c0bc859`); the bundled `llama-server` (`--lora`); FastAPI; vanilla ES-module frontend.

## Global Constraints

- **Feasibility is GATED — Task 1 is a prove-on-machine GO/NO-GO** and MUST pass before Tasks 2–8 are built/merged. If NO-GO, stop and reassess (do not build on an unproven mechanism).
- **The main app (Py3.14) MUST NOT import torch/peft/transformers/gguf or the conversion scripts.** Conversion runs ONLY in the sidecar venv via `training_sidecar/convert.py` (never imported by the app).
- **Managers and routes NEVER raise** — degrade to `{"error": …}` / an error status.
- **Admin-only** — all `/api/training/*` routes are admin-gated (existing router dependency).
- **Serve-time single LoRA only** — `--lora <adapter.gguf>` at scale 1.0. No multi-adapter, no `--lora-scaled` UI.
- **Base = a local GGUF** the adapter was trained on; auto-matched by name, else user picks/downloads via the existing catalog; the user confirms before serving. A wrong base yields garbage — never auto-serve an unconfirmed non-match.
- **Reuse sub-project 1's sidecar env** (`src/training/env.py TrainingEnv`, `training_sidecar/` bundling, `resolve_sidecar_script`); add only the `gguf` pip package + the vendored convert scripts.
- **Merge-to-standalone-GGUF is a NON-GOAL here** (deferred sub-project).
- **The GPU auto-fit contract is untouched:** never add `-ngl` (the Vulkan build auto-fits GPU layers; `-ngl` disables the fitter → OOM). `build_serve_argv` adds no device flags on GPU.
- pytest `--import-mode=importlib`. Node.js on PATH for JS tests (`node --input-type=module`, `node --check`). Commit directly to `dev`; messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Stage only the files each task names — never `git add -A`; never stage `installer/Output/Assist-Setup.exe`, `assistappicon.png`, `assistlogo.png`, `build_assets/uv/uv.exe`.
- ~220 unrelated pre-existing test failures exist elsewhere — run only the test files each task names. (A pre-existing SQLAlchemy `MovedIn20Warning` from `core/database.py` at collection is known and unrelated.)

### Verified anchors (from the codebase)
- `src/localmodels/runtime.py:59 build_serve_argv(binary, model_path, port, ctx_size=16384, host="127.0.0.1", device="cpu", mmproj=None) -> list` (sets `--alias os.path.basename(model_path)`; GPU adds no flags).
- `src/localmodels/runtime.py:136 list_gguf_models(models_dir) -> list`.
- `src/localmodels/manager.py:179 LocalModelManager.start(model_path, device="cpu")`; registers via `self._register(name=os.path.basename(model_path), base_url=url)` (~:210); `get_manager()` singleton (~:322) wired with `register_local_endpoint`.
- `src/training/env.py:11 STACK = ["transformers","peft","bitsandbytes","accelerate","datasets","trl"]`.
- `src/training/manager.py list_adapters()` returns `[{"run_id","complete","base_model","path"}]`; adapters live at `<DATA_DIR>/training/adapters/<run-id>/` with `run_config.json{base_model}`.
- `routes/training_routes.py setup_training_routes()` — admin-gated `APIRouter(prefix="/api/training", dependencies=[Depends(require_admin)])`.
- `src/constants.py:60 MODELS_DIR`.

---

### Task 1: Feasibility gate (prove convert + `--lora` serve) — GO/NO-GO

**This is a machine/GPU gate, not a headless code task.** It is run by the controller/user, not a fresh implementer subagent. Nothing in Tasks 2–8 is built until this records **GO**.

**Files:**
- Create: `docs/serve-adapter-feasibility.md` (the recorded result + the confirmed invocation/flag)
- Scratch: use `C:\tmp\serve_spike\` for throwaway artifacts

**Goal:** On this machine, prove (a) llama.cpp's `convert_lora_to_gguf.py` converts a peft adapter → a GGUF LoRA in the sidecar venv, and (b) the bundled `llama-server` serves a base GGUF + that adapter via `--lora` with output that **visibly differs** from the base alone.

- [ ] **Step 1: Prepare the conversion env**

Reuse the training spike venv if present (`C:\tmp\train_spike\.venv`, has torch/transformers/peft) or build the training env. Then add the conversion deps and vendor the scripts:
```bash
# in the sidecar venv:
<venv-python> -m pip install gguf
# vendor from a pinned llama.cpp release (record the tag used):
#   convert_lora_to_gguf.py  and its dependency  convert_hf_to_gguf.py  (+ the gguf/ helpers if the script imports them)
```
Record the exact llama.cpp tag and the `gguf` version that work.

- [ ] **Step 2: Make a throwaway adapter + convert it**

Create a tiny peft LoRA adapter on a small base (e.g. `Qwen/Qwen2.5-0.5B-Instruct`) — a few init steps, no real training needed — saved to `C:\tmp\serve_spike\adapter\`. Then:
```bash
<venv-python> convert_lora_to_gguf.py --outfile C:\tmp\serve_spike\adapter.gguf C:\tmp\serve_spike\adapter
```
Record the exact working invocation (whether `--base` is required, the outfile flag name). Confirm `adapter.gguf` is written and non-trivial.

- [ ] **Step 3: Serve base + adapter and compare output**

Obtain a small base GGUF for the same base (from the catalog or `C:\tmp\serve_spike\`). Run the bundled server twice on a fixed prompt (greedy), once without and once with the adapter:
```bash
build_assets/llama/vulkan/llama-server.exe --model base.gguf --host 127.0.0.1 --port 8123 ...
build_assets/llama/vulkan/llama-server.exe --model base.gguf --lora C:\tmp\serve_spike\adapter.gguf --host 127.0.0.1 --port 8124 ...
```
Record the **exact LoRA flag** the bundled server accepts (`--lora` vs `--lora-scaled` vs `--lora-adapter`) and confirm the server loads the adapter (log shows it) and the two outputs differ.

- [ ] **Step 4: Record GO/NO-GO**

Write `docs/serve-adapter-feasibility.md` with: GO or NO-GO; the confirmed `convert_lora_to_gguf.py` invocation; the confirmed `llama-server` LoRA flag; the llama.cpp tag + `gguf` version; and any gotchas (e.g. tokenizer files needed). **If NO-GO, STOP** and surface to the human — do not proceed to Task 2.

- [ ] **Step 5: Commit the result**

```bash
git add docs/serve-adapter-feasibility.md
git commit -m "docs(serve-adapter): feasibility gate result (GO/NO-GO) + confirmed convert/serve invocation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

> **Tasks 4 (convert.py invocation) and 3 (the `--lora` flag) below use the standard llama.cpp values; if Task 1 records different working values, use the recorded ones.**

---

### Task 2: Base-GGUF resolution (pure)

**Files:**
- Create: `src/training/base_resolve.py`
- Test: `tests/test_base_resolve.py`

**Interfaces:**
- Produces: `resolve_base_gguf(base_model: str, gguf_names: list) -> dict` → `{"matched": str|None, "candidates": list}`. `gguf_names` are GGUF basenames (from `list_gguf_models`). Matches by normalized tokens; requires the parameter-size token (e.g. `0.5b`) to agree, so a different-size GGUF is never matched.

- [ ] **Step 1: Write the failing test**

Create `tests/test_base_resolve.py`:

```python
from src.training.base_resolve import resolve_base_gguf

GGUFS = ["qwen2.5-0.5b-instruct-q4_k_m.gguf", "qwen2.5-1.5b-instruct-q4_k_m.gguf",
         "llama-3.2-1b-instruct-q8_0.gguf"]


def test_matches_same_size_family():
    out = resolve_base_gguf("Qwen/Qwen2.5-0.5B-Instruct", GGUFS)
    assert out["matched"] == "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def test_wrong_size_not_matched():
    # only a 1.5B gguf present for a 0.5B base -> no match
    out = resolve_base_gguf("Qwen/Qwen2.5-0.5B-Instruct",
                            ["qwen2.5-1.5b-instruct-q4_k_m.gguf"])
    assert out["matched"] is None


def test_no_candidates_when_family_absent():
    out = resolve_base_gguf("mistralai/Mistral-7B-Instruct-v0.3", GGUFS)
    assert out["matched"] is None and out["candidates"] == []


def test_non_str_inputs_safe():
    assert resolve_base_gguf(None, GGUFS)["matched"] is None
    assert resolve_base_gguf("Qwen/Qwen2.5-0.5B", None)["candidates"] == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_base_resolve.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.training.base_resolve'`).

- [ ] **Step 3: Implement**

Create `src/training/base_resolve.py`:

```python
"""Pair a trained adapter with a local base GGUF by name. Pure (no I/O)."""
import re

_QUANT = re.compile(r"^(q\d.*|iq\d.*|f16|f32|bf16|k|m|s|l|xl|gguf|gg)$", re.I)
_SIZE = re.compile(r"^\d+(?:\.\d+)?b$", re.I)


def _tokens(s):
    if not isinstance(s, str):
        return []
    s = s.rsplit("/", 1)[-1].lower()
    s = re.sub(r"\.gguf$", "", s)
    toks = [t for t in re.split(r"[^a-z0-9.]+", s) if t]
    return [t for t in toks if not _QUANT.match(t)]


def _size(toks):
    for t in toks:
        if _SIZE.match(t):
            return t
    return None


def resolve_base_gguf(base_model, gguf_names) -> dict:
    """Best local GGUF whose name matches the adapter's base_model (same family
    + same parameter-size token). Returns {"matched","candidates"}."""
    want = _tokens(base_model)
    want_set = set(want)
    want_size = _size(want)
    names = gguf_names if isinstance(gguf_names, list) else []
    best, best_score, cands = None, 0, []
    for name in names:
        if not isinstance(name, str):
            continue
        have = set(_tokens(name))
        # a candidate must share the parameter-size token (when the base has one)
        if want_size is not None and want_size not in have:
            continue
        score = len(want_set & have)
        if score >= max(1, len(want_set) - 1):
            cands.append(name)
            if score > best_score:
                best, best_score = name, score
    return {"matched": best, "candidates": cands}
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_base_resolve.py --import-mode=importlib -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/training/base_resolve.py tests/test_base_resolve.py
git commit -m "feat(serve-adapter): base-GGUF name resolution (size-aware)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `build_serve_argv` + manager `--lora`/alias

**Files:**
- Modify: `src/localmodels/runtime.py` (`build_serve_argv`), `src/localmodels/manager.py` (`start`)
- Test: `tests/test_serve_lora_argv.py`

**Interfaces:**
- Consumes: existing `build_serve_argv`, `LocalModelManager.start`.
- Produces: `build_serve_argv(..., mmproj=None, lora=None, alias=None)` → appends `--lora <lora>` when given, `--alias (alias or basename(model_path))`; `LocalModelManager.start(model_path, device="cpu", lora=None, alias=None)` threads both through and registers the endpoint under `alias or basename(model_path)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_serve_lora_argv.py`:

```python
from src.localmodels.runtime import build_serve_argv


def test_lora_and_alias_appended():
    argv = build_serve_argv("llama-server", "C:/m/base.gguf", 8100,
                            lora="C:/m/adapter.gguf", alias="tuned-x")
    assert "--lora" in argv and argv[argv.index("--lora") + 1] == "C:/m/adapter.gguf"
    assert argv[argv.index("--alias") + 1] == "tuned-x"


def test_no_lora_by_default_and_alias_falls_back_to_basename():
    argv = build_serve_argv("llama-server", "C:/m/base.gguf", 8100)
    assert "--lora" not in argv
    assert argv[argv.index("--alias") + 1] == "base.gguf"
```

Add to `tests/test_serve_lora_argv.py` a manager test. **The injected deps mirror the existing
`tests/test_localmodels_manager.py` fixture** — `resolve_binary`/`metadata_reader`/`hardware_detect`
MUST be injected or `start()` hits the real binary resolver + GGUF reader and fails:

```python
from src.localmodels.manager import LocalModelManager


class FakeProc:
    def poll(self): return None
    def terminate(self): pass
    def kill(self): pass
    def wait(self, timeout=None): return 0
    @property
    def pid(self): return 1234


def test_manager_start_threads_lora_and_registers_alias():
    captured, reg = {}, {}
    mgr = LocalModelManager(
        spawn=lambda argv: (captured.__setitem__("argv", argv), FakeProc())[1],
        port_chooser=lambda: 8100,
        probe=lambda url: True,
        register_endpoint=lambda name, base_url: (reg.__setitem__("name", name), "eid-1")[1],
        unregister_endpoint=lambda e: None,
        resolve_binary=lambda device="cpu": "/bin/llama-server",
        metadata_reader=lambda p: {},
        hardware_detect=lambda: {"has_gpu": False, "available_ram_gb": 16},
    )
    mgr.start("/models/base.gguf", device="cpu",
              lora="/models/adapter.gguf", alias="qwen · run-1 (LoRA)")
    argv = captured["argv"]
    assert "--lora" in argv and argv[argv.index("--lora") + 1] == "/models/adapter.gguf"
    assert reg["name"] == "qwen · run-1 (LoRA)"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_serve_lora_argv.py --import-mode=importlib -q`
Expected: FAIL (`build_serve_argv() got an unexpected keyword argument 'lora'`).

- [ ] **Step 3: Implement**

In `src/localmodels/runtime.py`, change the `build_serve_argv` signature + body. Replace the signature line and the `--alias`/`mmproj` handling:

```python
def build_serve_argv(binary: str, model_path: str, port: int,
                     ctx_size: int = 16384, host: str = "127.0.0.1",
                     device: str = "cpu", mmproj: str = None,
                     lora: str = None, alias: str = None) -> list:
```

and inside, set the alias from the param (fallback to basename) and append `--lora`:

```python
    argv = [
        binary,
        "--model", model_path,
        "--alias", alias or os.path.basename(model_path),
        "--host", host,
        "--port", str(port),
        "--ctx-size", str(ctx_size),
    ]
    if mmproj:
        argv += ["--mmproj", mmproj]
    if lora:
        argv += ["--lora", lora]
```

(Leave the docstring's `--alias`/`mmproj`/no-`-ngl` notes; the `return argv` and device comment are unchanged.)

In `src/localmodels/manager.py`, change `start` to accept and thread the new params. Replace the `def start(...)` line:

```python
    def start(self, model_path: str, device: str = "cpu",
              lora: str = None, alias: str = None) -> dict:
```

At the `build_serve_argv(...)` call (~:191), pass them through — the exact current call is:

```python
            proc = self._spawn(build_serve_argv(binary, model_path, port,
```

extend that call so it forwards `lora=lora, alias=alias` (keep the existing `device=`/`ctx_size=`/`mmproj=` args it already passes). At the register call (~:210), use the alias:

```python
            endpoint_id = None
            if self._register:
                endpoint_id = self._register(name=(alias or os.path.basename(model_path)),
                                             base_url=url)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_serve_lora_argv.py --import-mode=importlib -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/localmodels/runtime.py src/localmodels/manager.py tests/test_serve_lora_argv.py
git commit -m "feat(serve-adapter): build_serve_argv --lora + alias; manager threads them

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Vendor convert script + `conversion/` package + `gguf` stack + `convert.py` sidecar

**Files:**
- Modify: `src/training/env.py` (`STACK`)
- Create: `training_sidecar/convert.py`; vendor `training_sidecar/convert_lora_to_gguf.py` + the `training_sidecar/conversion/` package from the Task-1-recorded llama.cpp commit
- Test: `tests/test_convert_sidecar_syntax.py`

**Interfaces:**
- Produces: a sidecar `python convert.py --adapter <dir> [--base <hf_or_path>]` that writes `<dir>/adapter.gguf` and emits one JSON line: `{"event":"done","adapter_gguf":<path>}` or `{"event":"error","message":<str>}`.

**IMPORTANT (from the Task-1 gate):** `convert_lora_to_gguf.py` imports `from conversion import …` — llama.cpp's model classes live in a **`conversion/` package** (~82 files, ~1.2 MB), NOT in `convert_hf_to_gguf.py`. So vendor the script **plus the whole `conversion/` package**; do **not** vendor `convert_hf_to_gguf.py` (it is only for the deferred merge-to-standalone-GGUF path). Pin **`gguf==0.19.0`** (the version must match the scripts). These import `gguf`/torch/transformers — NOT importable by the Py3.14 app or suite; the only automated check is an `ast.parse` syntax gate on `convert.py` (the vendored upstream files are third-party — do not parse-gate them). `Assist.spec` already bundles the whole `training_sidecar/` dir, so the new files ship automatically. The gate left a working checkout at `C:\tmp\serve_spike\llama.cpp` (commit `c0bc8591e8815c63cb01dd3f051a8b0df02501c9`) to vendor from.

- [ ] **Step 1: Pin `gguf` in the sidecar stack**

In `src/training/env.py`, change line 11 (pin the version the convert scripts require):

```python
STACK = ["transformers", "peft", "bitsandbytes", "accelerate", "datasets", "trl", "gguf==0.19.0"]
```

- [ ] **Step 2: Vendor the convert script + `conversion/` package**

From the Task-1 llama.cpp checkout (`C:\tmp\serve_spike\llama.cpp`, commit `c0bc859`), copy into `training_sidecar/`:
- `convert_lora_to_gguf.py`
- the entire `conversion/` directory (it must sit beside `convert_lora_to_gguf.py` so `from conversion import …` resolves when the script runs — `python <dir>/convert_lora_to_gguf.py` puts `<dir>` on `sys.path[0]`).

Do not edit the vendored files. (Do NOT copy `convert_hf_to_gguf.py` — not needed for LoRA conversion.)

- [ ] **Step 3: Write the `convert.py` wrapper**

Create `training_sidecar/convert.py` (uses the Task-1-confirmed invocation; the default below is the standard one):

```python
"""Convert a trained peft LoRA adapter to a GGUF LoRA, INSIDE the Py3.11 sidecar
venv (never imported by the Py3.14 app). Emits one JSON line to stdout:
  {"event":"done","adapter_gguf":<path>}  |  {"event":"error","message":<str>}
"""
import argparse
import json
import os
import subprocess
import sys

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
    ap.add_argument("--base", default=None)
    args = ap.parse_args()
    try:
        out_path = os.path.join(args.adapter, "adapter.gguf")
        argv = [sys.executable, os.path.join(HERE, "convert_lora_to_gguf.py"),
                "--outfile", out_path]
        if args.base:
            argv += ["--base", args.base]
        argv += [args.adapter]
        p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if p.returncode != 0 or not os.path.isfile(out_path):
            tail = ((p.stdout or "") + (p.stderr or ""))[-1500:]
            emit({"event": "error", "message": "convert_lora_to_gguf failed: " + tail})
            sys.exit(1)
        emit({"event": "done", "adapter_gguf": out_path})
    except Exception as e:  # noqa: BLE001
        try:
            emit({"event": "error", "message": f"{e}"})
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Syntax-gate `convert.py` (no import — gguf/torch absent here)**

Create `tests/test_convert_sidecar_syntax.py`:

```python
import ast
import pathlib


def test_convert_py_parses():
    p = pathlib.Path(__file__).resolve().parents[1] / "training_sidecar" / "convert.py"
    ast.parse(p.read_text(encoding="utf-8"))  # parse only; never import (needs gguf/torch)
```

Run: `python -m pytest tests/test_convert_sidecar_syntax.py --import-mode=importlib -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/training/env.py training_sidecar/convert.py training_sidecar/convert_lora_to_gguf.py training_sidecar/conversion tests/test_convert_sidecar_syntax.py
git commit -m "feat(serve-adapter): vendor llama.cpp LoRA->GGUF convert + conversion pkg + sidecar wrapper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
(Staging `training_sidecar/conversion` adds the whole package dir.)

---

### Task 5: `AdapterConverter` orchestration

**Files:**
- Create: `src/training/convert_manager.py`
- Test: `tests/test_convert_manager.py`

**Interfaces:**
- Consumes: `TrainingEnv` (env.py), `resolve_sidecar_script`-style resolution (a `training_sidecar/convert.py` path via `runtime.resolve_sidecar_script`-analogue), `TrainingEnv.venv_python()`.
- Produces: `AdapterConverter(env=None, spawn=None)` with `convert(adapter_dir) -> dict` (`{"ok":True,"adapter_gguf":path}` | `{"error":str}`; blocking, never raises) and a module singleton `get_adapter_converter()`. `spawn(argv) -> (returncode, stdout)` is injectable.

- [ ] **Step 1: Write the failing test**

Create `tests/test_convert_manager.py`:

```python
import json
import os
from src.training.convert_manager import AdapterConverter


class FakeEnv:
    def __init__(self, ready=True): self._ready = ready
    def ensure_ready(self, progress=None):
        return {"ready": self._ready, "error": None if self._ready else "no env"}
    def venv_python(self): return "venv/python"


def _adapter(tmp_path):
    d = tmp_path / "run-1"; d.mkdir()
    (d / "adapter_model.safetensors").write_text("x")
    (d / "run_config.json").write_text(json.dumps({"base_model": "x/Qwen2.5-0.5B"}))
    return str(d)


def test_convert_success(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.convert_manager.resolve_convert_script", lambda: "convert.py")
    d = _adapter(tmp_path)
    def spawn(argv):
        # simulate the sidecar writing adapter.gguf + emitting a done line
        open(os.path.join(d, "adapter.gguf"), "w").close()
        return (0, json.dumps({"event": "done", "adapter_gguf": os.path.join(d, "adapter.gguf")}))
    conv = AdapterConverter(env=FakeEnv(), spawn=spawn)
    out = conv.convert(d)
    assert out.get("ok") is True and out["adapter_gguf"].endswith("adapter.gguf")


def test_convert_env_not_ready(tmp_path):
    conv = AdapterConverter(env=FakeEnv(ready=False), spawn=lambda a: (0, ""))
    out = conv.convert(_adapter(tmp_path))
    assert "error" in out and "env" in out["error"].lower()


def test_convert_sidecar_error(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.convert_manager.resolve_convert_script", lambda: "convert.py")
    def spawn(argv): return (1, json.dumps({"event": "error", "message": "bad arch"}))
    conv = AdapterConverter(env=FakeEnv(), spawn=spawn)
    out = conv.convert(_adapter(tmp_path))
    assert "error" in out and "bad arch" in out["error"]


def test_convert_never_raises_on_spawn_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.convert_manager.resolve_convert_script", lambda: "convert.py")
    def boom(argv): raise RuntimeError("cannot spawn")
    conv = AdapterConverter(env=FakeEnv(), spawn=boom)
    out = conv.convert(_adapter(tmp_path))
    assert "error" in out
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_convert_manager.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.training.convert_manager'`).

- [ ] **Step 3: Implement**

First add a `resolve_convert_script` to `src/training/runtime.py` (mirrors `resolve_sidecar_script`, pointing at `training_sidecar/convert.py`):

```python
def resolve_convert_script(frozen_base=None, dev_base=None) -> str:
    """Path to training_sidecar/convert.py (frozen _MEIPASS / dev repo). Raises RuntimeError if missing."""
    base = _frozen_base(frozen_base)
    if base:
        cand = os.path.join(base, "training_sidecar", "convert.py")
        if os.path.isfile(cand):
            return cand
    if dev_base is None:
        dev_base = os.path.join(_REPO_ROOT, "training_sidecar")
    cand = os.path.join(dev_base, "convert.py")
    if os.path.isfile(cand):
        return cand
    raise RuntimeError("training_sidecar/convert.py not found.")
```

Create `src/training/convert_manager.py`:

```python
"""Orchestrate the LoRA->GGUF conversion sidecar from the Py3.14 app. Never
imports the conversion stack; never raises. Blocking (call via asyncio.to_thread)."""
import json
import os
import subprocess

from src.training.runtime import resolve_convert_script


def _default_spawn(argv):
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return p.returncode, (p.stdout or "")


class AdapterConverter:
    def __init__(self, env=None, spawn=None):
        if env is None:
            from src.training.env import TrainingEnv
            env = TrainingEnv()
        self._env = env
        self._spawn = spawn or _default_spawn

    def convert(self, adapter_dir, base=None) -> dict:
        try:
            if not (isinstance(adapter_dir, str) and os.path.isdir(adapter_dir)):
                return {"error": "adapter directory not found"}
            ready = self._env.ensure_ready()
            if not ready.get("ready"):
                return {"error": f"training env not ready: {ready.get('error')}"}
            argv = [self._env.venv_python(), resolve_convert_script(), "--adapter", adapter_dir]
            if base:
                argv += ["--base", base]
            rc, out = self._spawn(argv)
            ev = {}
            for line in (out or "").splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        ev = json.loads(line)
                    except Exception:
                        pass
            if ev.get("event") == "done" and ev.get("adapter_gguf"):
                return {"ok": True, "adapter_gguf": ev["adapter_gguf"]}
            if ev.get("event") == "error":
                return {"error": ev.get("message", "conversion failed")}
            gguf = os.path.join(adapter_dir, "adapter.gguf")
            if rc == 0 and os.path.isfile(gguf):
                return {"ok": True, "adapter_gguf": gguf}
            return {"error": "conversion failed: " + (out or "")[-500:]}
        except Exception as e:  # noqa: BLE001
            return {"error": f"conversion error: {e}"}


_converter = None


def get_adapter_converter():
    global _converter
    if _converter is None:
        _converter = AdapterConverter()
    return _converter
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_convert_manager.py tests/test_training_runtime.py --import-mode=importlib -q`
Expected: PASS (4 convert + the existing runtime tests).

- [ ] **Step 5: Commit**

```bash
git add src/training/runtime.py src/training/convert_manager.py tests/test_convert_manager.py
git commit -m "feat(serve-adapter): AdapterConverter orchestrates the convert sidecar (never-raises)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: API routes + serve wiring + `list_adapters` fields

**Files:**
- Modify: `src/training/manager.py` (`list_adapters` gains `converted`/`adapter_gguf`), `routes/training_routes.py` (3 endpoints)
- Test: `tests/test_serve_adapter_routes.py`

**Interfaces:**
- Consumes: `get_adapter_converter` (Task 5), `resolve_base_gguf` (Task 2), `list_gguf_models` + `LocalModelManager.get_manager().start(..., lora=, alias=)` (Task 3), `get_training_manager` (existing).
- Produces: `GET /api/training/adapters/{run_id}` → `{run_id, converted, adapter_gguf, base_model, base_match, complete}`; `POST /api/training/adapters/{run_id}/convert` → `{ok}`/400; `POST /api/training/adapters/{run_id}/serve` body `{base_gguf}` → serve result/400.

- [ ] **Step 1: Write the failing test**

Create `tests/test_serve_adapter_routes.py`:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.training_routes as tr


class FakeMgr:
    def list_adapters(self):
        return [{"run_id": "run-1", "complete": True, "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
                 "path": "p", "converted": True, "adapter_gguf": "p/adapter.gguf"}]


def _client(monkeypatch, **patches):
    monkeypatch.setattr(tr, "require_admin", lambda: None)
    monkeypatch.setattr(tr, "get_training_manager", lambda: FakeMgr())
    for k, v in patches.items():
        monkeypatch.setattr(tr, k, v)
    app = FastAPI(); app.include_router(tr.setup_training_routes())
    return TestClient(app)


def test_adapter_status(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/api/training/adapters/run-1")
    assert r.status_code == 200 and r.json()["converted"] is True


def test_convert_endpoint(monkeypatch):
    class Conv:
        def convert(self, d, base=None): return {"ok": True, "adapter_gguf": "p/adapter.gguf"}
    c = _client(monkeypatch, get_adapter_converter=lambda: Conv())
    r = c.post("/api/training/adapters/run-1/convert")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_convert_error_is_400(monkeypatch):
    class Conv:
        def convert(self, d, base=None): return {"error": "bad arch"}
    c = _client(monkeypatch, get_adapter_converter=lambda: Conv())
    r = c.post("/api/training/adapters/run-1/convert")
    assert r.status_code == 400


def test_serve_endpoint(monkeypatch):
    served = {}
    class LM:
        def start(self, path, device="cpu", lora=None, alias=None):
            served.update(path=path, lora=lora, alias=alias); return {"running": True}
    c = _client(monkeypatch, get_local_manager=lambda: LM())
    r = c.post("/api/training/adapters/run-1/serve", json={"base_gguf": "C:/m/qwen2.5-0.5b-instruct-q4_k_m.gguf"})
    assert r.status_code == 200
    assert served["lora"] == "p/adapter.gguf" and "run-1" in served["alias"]


def test_serve_requires_converted(monkeypatch):
    class Mgr2(FakeMgr):
        def list_adapters(self):
            a = super().list_adapters(); a[0]["converted"] = False; a[0]["adapter_gguf"] = None; return a
    monkeypatch.setattr(tr, "get_training_manager", lambda: Mgr2())
    monkeypatch.setattr(tr, "require_admin", lambda: None)
    app = FastAPI(); app.include_router(tr.setup_training_routes())
    r = TestClient(app).post("/api/training/adapters/run-1/serve", json={"base_gguf": "b.gguf"})
    assert r.status_code == 400
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_serve_adapter_routes.py --import-mode=importlib -q`
Expected: FAIL (routes/attributes missing).

- [ ] **Step 3: Implement `list_adapters` fields**

In `src/training/manager.py` `list_adapters`, for each adapter dir also report conversion state. Where it builds the per-adapter dict, add:

```python
            adapter_gguf = os.path.join(d, "adapter.gguf")
            converted = os.path.isfile(adapter_gguf)
            out.append({"run_id": name, "complete": has,
                        "base_model": cfg.get("base_model"), "path": d,
                        "converted": converted,
                        "adapter_gguf": adapter_gguf if converted else None})
```

- [ ] **Step 4: Implement the routes**

In `routes/training_routes.py`, add imports at the top (`import os` too — the module doesn't import it yet):

```python
import os
from src.training.convert_manager import get_adapter_converter
from src.training.base_resolve import resolve_base_gguf
from src.localmodels.manager import get_manager as get_local_manager
from src.localmodels.runtime import list_gguf_models
from src.constants import MODELS_DIR
```

`list_gguf_models(MODELS_DIR)` returns `[{"name","path","size"}]`, so `m["name"]` is the GGUF basename.

Add a helper + the three routes inside `setup_training_routes()` (after the existing `/adapters` route):

```python
    def _find_adapter(run_id):
        for a in get_training_manager().list_adapters():
            if a.get("run_id") == run_id:
                return a
        return None

    @router.get("/adapters/{run_id}")
    async def adapter_status(run_id: str):
        a = _find_adapter(run_id)
        if not a:
            raise HTTPException(404, "adapter not found")
        names = [m["name"] for m in list_gguf_models(MODELS_DIR) if isinstance(m, dict) and m.get("name")]
        match = resolve_base_gguf(a.get("base_model"), names)
        return {**a, "base_match": match}

    @router.post("/adapters/{run_id}/convert")
    async def convert_adapter(run_id: str):
        a = _find_adapter(run_id)
        if not a:
            raise HTTPException(404, "adapter not found")
        out = await asyncio.to_thread(get_adapter_converter().convert, a["path"])
        if "error" in out:
            raise HTTPException(400, out["error"])
        return out

    @router.post("/adapters/{run_id}/serve")
    async def serve_adapter(run_id: str, body: dict = Body(...)):
        a = _find_adapter(run_id)
        if not a:
            raise HTTPException(404, "adapter not found")
        if not a.get("converted") or not a.get("adapter_gguf"):
            raise HTTPException(400, "adapter is not converted yet")
        base = str(body.get("base_gguf", "")).strip()
        if not base:
            raise HTTPException(400, "base_gguf is required")
        alias = f"{os.path.basename(base)} · {run_id} (LoRA)"
        out = get_local_manager().start(base, lora=a["adapter_gguf"], alias=alias)
        if isinstance(out, dict) and out.get("error"):
            raise HTTPException(400, out["error"])
        return out
```

- [ ] **Step 5: Run the tests + import smoke**

Run: `python -m pytest tests/test_serve_adapter_routes.py tests/test_training_routes.py --import-mode=importlib -q`
Expected: PASS.
Run: `python -c "import app"`
Expected: no error.

- [ ] **Step 6: Commit**

```bash
git add src/training/manager.py routes/training_routes.py tests/test_serve_adapter_routes.py
git commit -m "feat(serve-adapter): admin convert/serve/status routes + list_adapters converted fields

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: GUI — Convert/Serve controls

**Files:**
- Create: `static/js/serveCore.js`
- Modify: `static/js/training.js` (adapter-row actions), `static/index.html` (nothing new required beyond Task-2-GUI ids; see below)
- Test: `tests/test_serve_core_js.py`, `tests/test_training_ui.py` (extend)

**Interfaces:**
- Consumes: `/api/training/adapters/{run_id}` + `/convert` + `/serve` (Task 6).
- Produces: `serveCore.js` exports `adapterActions(adapter) -> {canConvert, canServe}`; `training.js` renders per-adapter Convert/Serve buttons and calls the routes.

- [ ] **Step 1: Write the failing test (pure core)**

Create `tests/test_serve_core_js.py`:

```python
import json, subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "static" / "js" / "serveCore.js"


def _node(expr):
    s = f"import * as m from {json.dumps(CORE.as_uri())};\n{expr}"
    p = subprocess.run(["node", "--input-type=module", "-e", s],
                       capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def test_adapter_actions():
    out = _node("console.log(JSON.stringify(["
                "m.adapterActions({complete:true,converted:false}),"
                "m.adapterActions({complete:true,converted:true}),"
                "m.adapterActions({complete:false,converted:false})]));")
    a = json.loads(out)
    assert a[0] == {"canConvert": True, "canServe": False}
    assert a[1] == {"canConvert": False, "canServe": True}
    assert a[2] == {"canConvert": False, "canServe": False}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_serve_core_js.py --import-mode=importlib -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `serveCore.js`**

Create `static/js/serveCore.js`:

```javascript
// Pure helper for the adapter Convert/Serve button states — no DOM.
export function adapterActions(a) {
  const complete = !!(a && a.complete);
  const converted = !!(a && a.converted);
  return { canConvert: complete && !converted, canServe: complete && converted };
}
```

- [ ] **Step 4: Wire the buttons into `training.js`**

In `static/js/training.js`, replace `refreshAdapters`'s row rendering so each adapter shows Convert/Serve actions, and add the handlers. Replace the existing `refreshAdapters` function with:

```javascript
async function refreshAdapters() {
  try {
    const j = await api('/api/training/adapters');
    const host = $('training-adapters');
    if (!host) return;
    const rows = (j.adapters || []).map(function (a) {
      const act = adapterActions(a);
      const btns =
        (act.canConvert ? '<button class="btn" data-conv="' + esc(a.run_id) + '">Convert to GGUF</button>' : '') +
        (act.canServe ? '<button class="btn" data-serve="' + esc(a.run_id) + '">Serve</button>' : '');
      return '<div>' + (a.complete ? '✅' : '⏳') + ' ' + esc(a.run_id) + ' — ' +
             esc(a.base_model || '?') + ' ' + btns + '</div>';
    });
    host.innerHTML = rows.join('') || 'None yet.';
    host.querySelectorAll('[data-conv]').forEach(function (b) {
      b.addEventListener('click', function () { convertAdapter(b.getAttribute('data-conv')); });
    });
    host.querySelectorAll('[data-serve]').forEach(function (b) {
      b.addEventListener('click', function () { serveAdapter(b.getAttribute('data-serve')); });
    });
  } catch (e) {}
}

async function convertAdapter(runId) {
  const host = $('training-adapters');
  try {
    if (host) host.textContent = 'Converting ' + runId + ' to GGUF…';
    await api('/api/training/adapters/' + encodeURIComponent(runId) + '/convert', { method: 'POST' });
  } catch (e) { alert('Convert failed: ' + e.message); }
  refreshAdapters();
}

async function serveAdapter(runId) {
  try {
    const st = await api('/api/training/adapters/' + encodeURIComponent(runId));
    let base = st.base_match && st.base_match.matched;
    base = window.prompt('Base GGUF to serve with this adapter (must match ' +
                         (st.base_model || 'the base') + '):', base || '');
    if (!base) return;
    await api('/api/training/adapters/' + encodeURIComponent(runId) + '/serve', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_gguf: base }),
    });
    alert('Serving — pick the tuned model in the model selector.');
  } catch (e) { alert('Serve failed: ' + e.message); }
}
```

Add the import at the top of `training.js` (next to the trainingCore import):

```javascript
import { adapterActions } from '/static/js/serveCore.js';
```

(Task-2 GUI already imports from `trainingCore.js`; add this second import line.)

- [ ] **Step 5: Extend the UI test**

Append to `tests/test_training_ui.py`:

```python
def test_training_js_references_adapter_serve_routes():
    src = (ROOT / "static" / "js" / "training.js").read_text(encoding="utf-8")
    assert "/convert" in src and "/serve" in src and "serveCore.js" in src
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_serve_core_js.py tests/test_training_ui.py --import-mode=importlib -q`
Expected: PASS (the `node --check` gate in `test_training_ui.py` confirms `training.js` still parses with the new import + handlers).

- [ ] **Step 7: Commit**

```bash
git add static/js/serveCore.js static/js/training.js tests/test_serve_core_js.py tests/test_training_ui.py
git commit -m "feat(serve-adapter): GUI Convert/Serve controls in the Adapters list

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Finalize — docs + full suite + no-torch guard

**Files:**
- Modify: `docs/training-engine.md` (a "Serving a tuned adapter" section) or create `docs/serve-adapter.md`; `docs/training-gui-manual-verification.md` (add serve steps)
- Test: (run the whole serve-adapter + training suite + `import app` no-torch guard)

- [ ] **Step 1: Document the flow**

Create `docs/serve-adapter.md`:

```markdown
# Serving a Fine-Tuned Adapter

After training a LoRA adapter, serve it locally as a chat model:

1. **Convert** — in the Training panel's Adapters list, click **Convert to GGUF**.
   This runs `convert_lora_to_gguf.py` in the training sidecar venv and writes
   `adapter.gguf` next to the adapter.
2. **Serve** — click **Serve**. The app auto-matches a local base GGUF by the
   adapter's recorded base model (or prompts you to pick/download one). It then
   runs `llama-server --model <base>.gguf --lora adapter.gguf`, aliased to the
   tuned name, and registers it as a local endpoint.
3. **Chat** — pick the tuned model (shown as `<base> · <run-id> (LoRA)`) in the
   model selector and chat.

## API (admin, /api/training)
- `GET /adapters/{run_id}` → status incl. `converted`, `adapter_gguf`, `base_match`.
- `POST /adapters/{run_id}/convert` → convert to a GGUF LoRA.
- `POST /adapters/{run_id}/serve` (body `{base_gguf}`) → serve base + `--lora`.

## Owed (manual, needs the GPU)
- The Task-1 feasibility gate (convert + `--lora` serve) must be GO.
- One real end-to-end run: train → Convert → Serve → chat, verifying the tuned
  output differs from the base.

## Not yet
Merging the adapter into a standalone GGUF (a later sub-project); multi-adapter
stacking; per-adapter scale (`--lora-scaled`).
```

Add a bullet to `docs/training-gui-manual-verification.md` under the runbook: convert an adapter, serve it, confirm the tuned model appears in the selector and its output differs from the base.

- [ ] **Step 2: Run the full serve-adapter + training suite**

Run: `python -m pytest tests/test_base_resolve.py tests/test_serve_lora_argv.py tests/test_convert_sidecar_syntax.py tests/test_convert_manager.py tests/test_serve_adapter_routes.py tests/test_serve_core_js.py tests/test_training_ui.py tests/test_training_routes.py tests/test_training_manager.py --import-mode=importlib -q`
Expected: PASS (all green).

- [ ] **Step 3: Import smoke + no-torch/gguf guard**

Run: `python -c "import app, sys; assert 'torch' not in sys.modules and 'gguf' not in sys.modules, 'app must not import the conversion stack'; print('OK')"`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add docs/serve-adapter.md docs/training-gui-manual-verification.md
git commit -m "docs(serve-adapter): serve-a-tuned-adapter usage + manual verification

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Task 1 is a hard gate.** Do not build Tasks 2–8 until it records GO. Tasks 3/4 use the standard llama.cpp `--lora` flag + convert invocation; if Task 1 recorded different working values, use those.
- **The Py3.14 app never imports the conversion stack.** Only `training_sidecar/convert.py` + the vendored scripts import `gguf`/torch, and they run in the sidecar venv. Task 8 asserts `torch`/`gguf` aren't in `sys.modules` after `import app`.
- **Never-raises** on the converter/manager/routes is load-bearing; a failed conversion or a base/adapter mismatch degrades to an error, never a crash.
- **Pairing safety:** never serve an unconfirmed non-matching base — the UI shows the auto-match for confirmation and the user can override; a wrong base is surfaced by `llama-server`'s own failure/log tail.
- **Owed by the user (manual, GPU):** run Task 1; then a real train → convert → serve → chat. After a rebuild, a frozen boot-check that `resolve_convert_script()` resolves and the new routes register.
- **Scope:** serve-time single-LoRA only. NO merge-to-standalone-GGUF (deferred), NO `--lora-scaled` UI, NO multi-adapter.
