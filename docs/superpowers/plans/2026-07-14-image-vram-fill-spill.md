# Image-Model VRAM Fill-and-Spill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve local image models so they fill GPU VRAM up to a budget and stream the overflow from system RAM (`--max-vram` + `--stream-layers`), with an automatic Tier1→Tier2→Tier3 fallback so a serve never hard-fails.

**Architecture:** A new `free_vram_gb()` in the hwfit hardware module measures live free VRAM. `build_serve_argv` gains a `max_vram_gb` parameter that switches the GPU recipe between fill-and-spill (`--max-vram N --stream-layers`) and the legacy all-RAM `--offload-to-cpu`. `ImageModelManager.start` computes a budget and iterates an ordered `(device, max_vram_gb)` attempt ladder — GPU fill+spill → GPU all-RAM → CPU binary — re-resolving the binary per attempt and recording the device actually used.

**Tech Stack:** Python 3, pytest (`--import-mode=importlib`), stable-diffusion.cpp `sd-server` (bundled Vulkan + CPU builds), nvidia-smi.

## Global Constraints

- Image models only. LLM serving (llama-server) is untouched.
- No new settings, UI, or per-serve manual budget. Fully automatic.
- `VRAM_MARGIN_GB = 1.0` (GB of free VRAM left as headroom for compute buffers).
- Fallback ladder order for a GPU request: `("gpu", budget)` → `("gpu", None)` → `("cpu", None)`. Tier 1 is skipped when no budget is detected. An explicit CPU request is a single `("cpu", None)` attempt.
- `build_serve_argv(..., max_vram_gb=None)` MUST reproduce today's argv exactly (default keeps every existing test green).
- SD/SDXL checkpoints keep `use_fa = False` (flash attention crashes them on the Vulkan build); `--vae-tiling` is always present; CPU device never gets `--max-vram` or `--offload-to-cpu`.
- Run pytest with `--import-mode=importlib`.

---

### Task 1: Live free-VRAM detection

**Files:**
- Modify: `services/hwfit/hardware.py` (add `free_vram_gb()` after `_detect_nvidia`, ~line 145)
- Test: `tests/test_hwfit_free_vram.py` (create)

**Interfaces:**
- Consumes: existing module globals `_run(cmd)` (returns nvidia-smi stdout string or falsy) and `NVIDIA_PATH_CANDIDATES` (list of absolute nvidia-smi paths).
- Produces: `free_vram_gb() -> float | None` — free VRAM of the first NVIDIA GPU in GB, or `None` when unavailable/non-numeric. Never raises.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hwfit_free_vram.py`:

```python
import services.hwfit.hardware as hw


def test_free_vram_parses_first_gpu(monkeypatch):
    monkeypatch.setattr(hw, "_run", lambda cmd: "5432\n")
    got = hw.free_vram_gb()
    assert got is not None and abs(got - 5432 / 1024.0) < 0.01


def test_free_vram_multi_gpu_returns_first(monkeypatch):
    monkeypatch.setattr(hw, "_run", lambda cmd: "5432\n8000\n")
    assert abs(hw.free_vram_gb() - 5432 / 1024.0) < 0.01


def test_free_vram_none_when_no_smi(monkeypatch):
    monkeypatch.setattr(hw, "_run", lambda cmd: None)
    assert hw.free_vram_gb() is None


def test_free_vram_none_when_nonnumeric(monkeypatch):
    monkeypatch.setattr(hw, "_run", lambda cmd: "[N/A]\n")
    assert hw.free_vram_gb() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_hwfit_free_vram.py --import-mode=importlib -q`
Expected: FAIL with `AttributeError: module 'services.hwfit.hardware' has no attribute 'free_vram_gb'`.

- [ ] **Step 3: Implement `free_vram_gb`**

In `services/hwfit/hardware.py`, add after the `_detect_nvidia()` function:

```python
def free_vram_gb():
    """Live free VRAM of the first NVIDIA GPU in GB, or None when there is no
    usable NVIDIA GPU / nvidia-smi is unavailable / the value is non-numeric
    (unified-memory parts report '[N/A]'). Never raises. Mirrors _detect_nvidia's
    _run + absolute-path fallback."""
    q = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    out = _run(q)
    if not out:
        for _p in NVIDIA_PATH_CANDIDATES:
            out = _run([_p, "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
            if out:
                break
    if not out:
        return None
    first = out.strip().split("\n")[0].strip()
    try:
        return float(first) / 1024.0
    except ValueError:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_hwfit_free_vram.py --import-mode=importlib -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add services/hwfit/hardware.py tests/test_hwfit_free_vram.py
git commit -m "feat(hwfit): free_vram_gb() live VRAM probe"
```

---

### Task 2: `build_serve_argv` VRAM-budget parameter

**Files:**
- Modify: `src/imagemodels/runtime.py` (add `_fmt_gb` before `build_serve_argv`; add `max_vram_gb` param at line 20-21; rewrite the `device == "gpu"` block at lines 77-88)
- Test: `tests/test_imagemodels_runtime.py` (add cases)

**Interfaces:**
- Consumes: nothing new.
- Produces: `build_serve_argv(binary, files, port, device="cpu", host="127.0.0.1", threads=0, steps=None, max_vram_gb=None)`. When `device=="gpu"` and `max_vram_gb` is truthy → `--max-vram <gb> --stream-layers`; else → `--offload-to-cpu`. `_fmt_gb(gb) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_imagemodels_runtime.py`:

```python
def test_fmt_gb_integer_and_fractional():
    assert rt._fmt_gb(5) == "5"
    assert rt._fmt_gb(5.0) == "5"
    assert rt._fmt_gb(4.5) == "4.5"


def test_build_argv_gpu_max_vram_fills_and_streams():
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/c.safetensors", "vae": "/m/v.safetensors"}
    argv = rt.build_serve_argv("/x/sd", files, 8200, device="gpu", max_vram_gb=5)
    assert argv[argv.index("--max-vram") + 1] == "5"
    assert "--stream-layers" in argv
    assert "--offload-to-cpu" not in argv
    assert "--diffusion-fa" in argv  # FLUX.1 keeps flash attention


def test_build_argv_gpu_no_budget_keeps_offload():
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/c.safetensors", "vae": "/m/v.safetensors"}
    argv = rt.build_serve_argv("/x/sd", files, 8200, device="gpu", max_vram_gb=None)
    assert "--offload-to-cpu" in argv and "--max-vram" not in argv


def test_build_argv_checkpoint_max_vram_no_flash_attn():
    files = {"checkpoint": "/m/juggernaut-xl-v9-Q8_0.gguf"}
    argv = rt.build_serve_argv("/x/sd", files, 8200, device="gpu", max_vram_gb=5)
    assert "--max-vram" in argv and "--stream-layers" in argv
    assert "--diffusion-fa" not in argv


def test_build_argv_cpu_ignores_max_vram():
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/c.safetensors", "vae": "/m/v.safetensors"}
    argv = rt.build_serve_argv("/x/sd", files, 8200, device="cpu", threads=8, max_vram_gb=5)
    assert "--max-vram" not in argv and "--offload-to-cpu" not in argv
    assert argv[argv.index("-t") + 1] == "8"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_imagemodels_runtime.py -k "max_vram or fmt_gb or no_budget" --import-mode=importlib -q`
Expected: FAIL (`AttributeError: ... _fmt_gb` and/or `--max-vram` not found).

- [ ] **Step 3: Add `_fmt_gb` and the `max_vram_gb` parameter**

In `src/imagemodels/runtime.py`, add just before `def build_serve_argv`:

```python
def _fmt_gb(gb) -> str:
    """Render a VRAM budget for sd-server's --max-vram (integer when whole)."""
    gb = float(gb)
    return str(int(gb)) if gb == int(gb) else f"{gb:.1f}"
```

Change the signature (line 20-21) to add the keyword:

```python
def build_serve_argv(binary, files, port, device="cpu", host="127.0.0.1",
                     threads=0, steps=None, max_vram_gb=None):
```

- [ ] **Step 4: Rewrite the GPU device block**

Replace the current `if device == "gpu":` block (lines 77-88) with:

```python
    if device == "gpu":
        if max_vram_gb:
            # Fill VRAM up to the budget and stream the overflow from RAM
            # (graph-cut segmented execution): small models stay resident, big
            # ones fit instead of the blanket all-RAM offload.
            argv += ["--max-vram", _fmt_gb(max_vram_gb), "--stream-layers"]
        else:
            # No budget (VRAM undetectable): the proven all-RAM recipe — weights
            # live in RAM and stream into VRAM per use.
            argv += ["--offload-to-cpu"]
        if use_fa:
            argv += ["--diffusion-fa"]
    elif threads:
        argv += ["-t", str(threads)]
    return argv
```

- [ ] **Step 5: Run the full runtime suite (new + regression)**

Run: `python -m pytest tests/test_imagemodels_runtime.py --import-mode=importlib -q`
Expected: PASS. The existing `test_build_argv_gpu_low_vram_layout` still passes because `max_vram_gb` defaults to `None` → `--offload-to-cpu`.

- [ ] **Step 6: Commit**

```bash
git add src/imagemodels/runtime.py tests/test_imagemodels_runtime.py
git commit -m "feat(imagemodels): build_serve_argv max_vram_gb fill-and-spill option"
```

---

### Task 3: Manager budget + Tier1→Tier2→Tier3 fallback ladder

**Files:**
- Modify: `src/imagemodels/manager.py` (add `VRAM_MARGIN_GB` + `_default_vram_probe` near the top; add `vram_probe` to `__init__` at lines 74-95; rewrite `start` at lines 130-164)
- Test: `tests/test_imagemodels_manager.py` (update `make_manager`, add fallback cases)

**Interfaces:**
- Consumes: `free_vram_gb()` (Task 1, via `services.hwfit.hardware`); `build_serve_argv(..., max_vram_gb=...)` (Task 2); existing `resolve_sd_binary(device)`.
- Produces: `ImageModelManager(..., vram_probe=None)`; `start` records `state["device"]` = the device of the attempt that succeeded.

- [ ] **Step 1: Update the test harness and write the failing fallback tests**

In `tests/test_imagemodels_manager.py`, replace `make_manager` with a version that injects `vram_probe` and a per-attempt readiness gate, and returns device-tagged binaries:

```python
def make_manager(ready=True, spawned=None, registered=None, unregistered=None,
                 proc_exit_code=None, vram=6.0, ready_after=None):
    spawned = spawned if spawned is not None else []
    registered = registered if registered is not None else []
    unregistered = unregistered if unregistered is not None else []
    if ready_after is None:
        ready_after = 0 if ready else 999  # 0 = first attempt ready; 999 = never

    def spawn(argv):
        p = FakeProc(exit_code=proc_exit_code)
        spawned.append((argv, p))
        return p

    def register(name, base_url):
        eid = f"img-local-{len(registered)}"
        registered.append({"name": name, "base_url": base_url, "id": eid})
        return eid

    def probe(url):
        # The latest spawn is the current attempt (0-based index).
        return (len(spawned) - 1) >= ready_after

    clock = itertools.count(0, 10)
    mgr = ImageModelManager(
        spawn=spawn, port_chooser=lambda: 8200, probe=probe,
        register_endpoint=register, unregister_endpoint=lambda eid: unregistered.append(eid),
        resolve_binary=lambda device: f"/bin/sd-{device}",
        log_path="/nonexistent/sd-server.log",
        sleep=lambda _s: None, now=lambda: next(clock), ready_timeout=45.0,
        vram_probe=lambda: vram)
    return mgr, spawned, registered, unregistered
```

Then add these tests:

```python
def test_start_gpu_tier1_fills_vram():
    mgr, spawned, registered, _ = make_manager(vram=6.0)
    st = mgr.start(FILES, device="gpu")
    assert st["running"] is True and st["device"] == "gpu"
    argv = spawned[0][0]
    assert argv[argv.index("--max-vram") + 1] == "5"  # 6.0 - 1.0 margin
    assert "--stream-layers" in argv and "--offload-to-cpu" not in argv
    assert len(spawned) == 1


def test_start_gpu_falls_back_to_offload():
    mgr, spawned, registered, _ = make_manager(vram=6.0, ready_after=1)
    st = mgr.start(FILES, device="gpu")
    assert st["running"] is True and st["device"] == "gpu"
    assert "--max-vram" in spawned[0][0]           # Tier 1 attempted
    assert "--offload-to-cpu" in spawned[1][0]     # Tier 2 succeeded
    assert spawned[0][0][0] == "/bin/sd-gpu" and spawned[1][0][0] == "/bin/sd-gpu"
    assert len(spawned) == 2


def test_start_gpu_falls_back_to_cpu():
    mgr, spawned, registered, _ = make_manager(vram=6.0, ready_after=2)
    st = mgr.start(FILES, device="gpu")
    assert st["running"] is True and st["device"] == "cpu"
    assert spawned[2][0][0] == "/bin/sd-cpu"       # Tier 3 uses the CPU binary
    assert "--max-vram" not in spawned[2][0] and "--offload-to-cpu" not in spawned[2][0]
    assert len(spawned) == 3


def test_start_gpu_no_budget_skips_tier1():
    mgr, spawned, registered, _ = make_manager(vram=None)
    st = mgr.start(FILES, device="gpu")
    assert st["running"] is True
    assert "--offload-to-cpu" in spawned[0][0] and "--max-vram" not in spawned[0][0]
    assert len(spawned) == 1


def test_start_cpu_does_not_probe_vram():
    probed = []
    clock = itertools.count(0, 10)
    spawned = []
    mgr = ImageModelManager(
        spawn=lambda argv: (spawned.append((argv, FakeProc())) or spawned[-1][1]),
        port_chooser=lambda: 8200, probe=lambda url: True,
        register_endpoint=lambda name, base_url: "eid",
        unregister_endpoint=lambda eid: None,
        resolve_binary=lambda device: f"/bin/sd-{device}",
        log_path="/nonexistent/sd-server.log", sleep=lambda _s: None,
        now=lambda: next(clock),
        vram_probe=lambda: probed.append(1))
    st = mgr.start(FILES, device="cpu")
    assert st["running"] is True and st["device"] == "cpu"
    assert probed == []            # VRAM probe not consulted for a CPU request
    assert spawned[0][0][0] == "/bin/sd-cpu"


def test_start_gpu_all_tiers_fail_raises():
    import pytest
    mgr, spawned, registered, _ = make_manager(vram=6.0, ready_after=999)
    with pytest.raises(RuntimeError, match="did not become ready"):
        mgr.start(FILES, device="gpu")
    assert registered == [] and len(spawned) == 3  # tried all three tiers
    assert mgr.status()["running"] is False
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `python -m pytest tests/test_imagemodels_manager.py -k "tier or budget or does_not_probe or all_tiers" --import-mode=importlib -q`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'vram_probe'`).

- [ ] **Step 3: Add the module constant and default probe**

In `src/imagemodels/manager.py`, after the imports block (after line 22), add:

```python
VRAM_MARGIN_GB = 1.0  # free VRAM left as headroom for sd-server compute buffers


def _default_vram_probe():
    """Live free VRAM (GB) or None. Lazy import keeps hwfit off the hot path."""
    try:
        from services.hwfit.hardware import free_vram_gb
        return free_vram_gb()
    except Exception:
        return None
```

- [ ] **Step 4: Add the `vram_probe` dependency to `__init__`**

Change the `__init__` signature (append the keyword) and store it. The signature becomes:

```python
    def __init__(self, spawn=None, port_chooser=None, probe=None,
                 register_endpoint=None, unregister_endpoint=None,
                 resolve_binary=resolve_sd_binary, log_path=None,
                 sleep=None, now=None, ready_timeout=45.0, probe_interval=0.5,
                 sec_per_gb=12.0, force_kill=None, vram_probe=None):
```

Add near the other assignments (e.g. right after `self._force_kill = ...`):

```python
        self._vram_probe = vram_probe or _default_vram_probe
```

- [ ] **Step 5: Rewrite `start` as the attempt ladder**

Replace the entire `start` method (lines 130-164) with:

```python
    def start(self, files: dict, device: str = "cpu", steps=None) -> dict:
        with self._lock:
            if self._proc is not None:
                self._stop_locked()
            # A bare diffusion model uses "diffusion_model"; an all-in-one
            # SD/SDXL checkpoint uses "checkpoint" — the primary file is either.
            model_path = files.get("diffusion_model") or files.get("checkpoint")
            threads = os.cpu_count() or 4
            # Ordered attempt ladder. A GPU request fills VRAM (Tier 1), then
            # falls back to all-RAM offload (Tier 2), then a pure-CPU serve
            # (Tier 3). No GPU tiers for an explicit CPU request.
            if device == "gpu":
                free = self._vram_probe()
                budget = max(1.0, free - VRAM_MARGIN_GB) if free else None
                attempts = ([("gpu", budget)] if budget else []) + \
                    [("gpu", None), ("cpu", None)]
            else:
                attempts = [("cpu", None)]

            last_msg = "sd-server did not start."
            for attempt_device, max_vram_gb in attempts:
                binary = self._resolve_binary(attempt_device)
                port = self._port_chooser()
                proc = self._spawn(build_serve_argv(
                    binary, files, port, device=attempt_device,
                    threads=threads, steps=steps, max_vram_gb=max_vram_gb))
                url = local_image_endpoint_url(port)
                timeout = self._ready_timeout_for(model_path)
                if self._await_ready(url + "/models", proc, timeout):
                    endpoint_id = None
                    if self._register:
                        endpoint_id = self._register(
                            name=os.path.basename(model_path), base_url=url)
                    self._proc = proc
                    self._state = {"model_path": model_path, "port": port,
                                   "endpoint_id": endpoint_id,
                                   "pid": getattr(proc, "pid", None),
                                   "device": attempt_device}
                    return self.status()
                exited = _poll(proc) is not None
                tail = _read_log_tail(self._log_path)
                self._terminate(proc)
                self._close_log()
                reason = ("exited on startup (the bundled sd.cpp may not support "
                          "this model's architecture)"
                          if exited else "did not become ready in time")
                last_msg = f"sd-server {reason}."
                if tail:
                    last_msg += "\n\n--- sd-server output (tail) ---\n" + tail
            raise RuntimeError(last_msg)
```

- [ ] **Step 6: Run the full manager suite (new + regression)**

Run: `python -m pytest tests/test_imagemodels_manager.py --import-mode=importlib -q`
Expected: PASS. Existing tests still hold: `test_start_launches_and_registers` and `test_stop_terminates_and_unregisters` use `device="cpu"` (single attempt); `test_start_readiness_failure_kills_and_raises` (ready=False → `ready_after=999`, single CPU attempt) matches "did not become ready"; `test_start_fails_fast_on_early_exit` (proc_exit_code=1, CPU) matches "exited on startup".

- [ ] **Step 7: Commit**

```bash
git add src/imagemodels/manager.py tests/test_imagemodels_manager.py
git commit -m "feat(imagemodels): VRAM-budget fallback ladder (fill->offload->cpu)"
```

---

### Task 4: Package and live-verify on the 6 GB card

**Files:**
- Modify: `installer/Output/Assist-Setup.exe` (rebuilt artifact; force-added like prior build commits)

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: a frozen build whose image serve fills VRAM then falls back.

- [ ] **Step 1: Full affected-suite run**

Run: `python -m pytest tests/test_hwfit_free_vram.py tests/test_imagemodels_runtime.py tests/test_imagemodels_manager.py tests/test_imagemodels_routes.py tests/test_imagemodels_encoders.py tests/test_gguf_meta.py --import-mode=importlib -q`
Expected: PASS (all green).

- [ ] **Step 2: Build the installer**

Run: `powershell -ExecutionPolicy Bypass -File .\build-installer.ps1 -Fast`
Expected: ends with `Installer built: ...\installer\Output\Assist-Setup.exe`.

- [ ] **Step 3: Frozen-verify the wiring (no real GPU spawn)**

Write a scratch script `verify.py` and run it through the frozen exe:

```python
import services.hwfit.hardware as hw
from src.imagemodels.runtime import build_serve_argv
# argv wiring
a = build_serve_argv("sd", {"diffusion_model": "/m/f.gguf", "t5xxl": "/m/t", "clip_l": "/m/c", "vae": "/m/v"}, 8200, device="gpu", max_vram_gb=5)
print("ARGV_OK" if ("--max-vram" in a and "--stream-layers" in a and "--offload-to-cpu" not in a) else "ARGV_FAIL")
# probe returns a float or None without raising
v = hw.free_vram_gb()
print("PROBE_OK", isinstance(v, float) or v is None, "value=", v)
```

Run: `./dist/Assist/Assist.exe --run-py "<scratchpad>/verify.py"`
Expected: `ARGV_OK` and `PROBE_OK True value= <free GB on this 6GB card, ~4-5>`.

- [ ] **Step 4: Live-verify serve behavior (manual, GPU free)**

Ensure the app is closed (no VRAM held). For each check, start the bundled sd-server through the manager with a no-op register (as in prior sessions' `verify_manager_checkpoint.py`, but with real spawn/binary and `device="gpu"`), or serve via the running app's Image models card and read `~/.assist/data/logs/sd-server.log`:
- Small model (`flux-2-klein-4b`): serves on GPU; `sd-server.log` shows `--max-vram`/`--stream-layers` and **non-zero resident VRAM** (not `VRAM 0.00MB`); generates a 512px image.
- Large model (`flux-2-klein-9b`, historically OOM): serves via Tier 1 streaming, or `status.device` reports a fallback to `gpu` (offload) / `cpu`; a generation completes with no unhandled 500.
- Confirm `sd/cpu/sd-server.exe` exists in `dist/Assist/_internal/sd/cpu/` (Tier 3 binary present).

Kill any leftover `sd-server.exe` after testing (`taskkill //IM sd-server.exe //F`).

- [ ] **Step 5: Commit the installer**

```bash
git add -f installer/Output/Assist-Setup.exe
git commit -m "build: Assist-Setup.exe with image VRAM fill-and-spill"
```

---

## Notes for the executor

- Every pytest invocation uses `--import-mode=importlib`.
- Tasks 1-3 are pure-Python + fakes; only Task 4 touches the GPU/binary.
- The single most important regression guard: `build_serve_argv(..., max_vram_gb=None)` must be byte-identical to today's GPU argv (`--offload-to-cpu [--diffusion-fa]`), so no existing image test changes behavior.
