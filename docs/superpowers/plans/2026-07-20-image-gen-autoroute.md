# Image-Gen Auto-Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a text/agent model is asked to generate an image, auto-serve the configured local default image model on demand (mirroring `ensure_vision_served`) so `do_generate_image` stops failing with "No endpoint found", and make `generate_image` reachable when a default image model is configured.

**Architecture:** Extract the route's inline arch-detect + encoder-resolution into a shared `resolve_image_files`. Add `ensure_image_served` (returns the running image model, or serves the configured local default). Wire it into `do_generate_image` via a small testable helper. Relax the tool-availability gate to also pass when a default image model is set.

**Tech Stack:** Python 3.14, existing sd-server / stable-diffusion.cpp image stack, `src/imagemodels/*`, FastAPI. No new dependencies.

## Global Constraints

- **No new dependencies.** No change to sd-server, `manager.start`'s serve ladder, or the external-API (gpt-image/dall-e) path.
- **Behavior-preserving route refactor** — extracting `resolve_image_files` must not change what `POST /api/imagemodels/serve` does, including the per-architecture "missing encoder" guidance messages and the payload encoder overrides (`llm`/`vae`/`t5xxl`/`clip_l`).
- **`ensure_image_served` NEVER raises** — returns `{"model": <id>|None, "error": <str>|None}`. A serve/resolve failure is caught and returned as `{"error": …}`.
- **Non-disruptive:** if a local image model is already serving, reuse it (no mid-session swap); serve the configured default only when nothing is running. Matches `ensure_vision_served`.
- **Auto-serve only when the model wasn't explicitly named in the tool call** — an explicit `model` line in the `generate_image` args is honored as-is.
- **Gate:** `generate_image` is available when `get_setting("image_gen_enabled", False)` is truthy **OR** `get_setting("image_model", "")` is a non-empty string.
- Anchors (verified): route inline logic at `routes/imagemodels_routes.py::serve()` (lines ~44-98); `ensure_vision_served` at `src/localmodels/manager.py:363`; image-manager singleton `get_manager()` at `src/imagemodels/manager.py:259`; `do_generate_image` at `src/ai_interaction.py:919`; gate at `src/agent_loop.py:1980`; `get_setting(k,d)=load_settings().get(k,d)` (`src/settings.py:263`); resolvers + `MissingEncoderError(.missing)` in `src/imagemodels/encoders.py`; `looks_like_flux2`/`looks_like_chroma` in `src/imagemodels/runtime.py`; `gguf_is_full_checkpoint`/`read_gguf_architecture` in `src/gguf_meta.py`.
- pytest `--import-mode=importlib`. Commit directly to `dev`. Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Stage only the files each task names — never `git add -A`; never stage `installer/Output/Assist-Setup.exe`, `assistappicon.png`, `assistlogo.png`.
- ~220 unrelated pre-existing test failures exist elsewhere — run only the test files each task names.

---

### Task 1: Extract `resolve_image_files` + refactor the route

**Files:**
- Create: `src/imagemodels/serve_resolve.py`
- Modify: `routes/imagemodels_routes.py` (imports + the `serve()` files-building block)
- Test: `tests/test_resolve_image_files.py`

**Interfaces:**
- Consumes: `gguf_is_full_checkpoint`, `read_gguf_architecture` (`src/gguf_meta.py`); `looks_like_flux2`, `looks_like_chroma` (`src/imagemodels/runtime.py`); `resolve_flux_files`, `resolve_flux2_files`, `resolve_chroma_files`, `resolve_zimage_files`, `MissingEncoderError` (`src/imagemodels/encoders.py`).
- Produces: `resolve_image_files(model_path, *, llm=None, vae=None, t5xxl=None, clip_l=None) -> dict` — returns the sd-server `files` dict; raises `MissingEncoderError` (with a `.hint` attribute carrying the arch-specific guidance string).

- [ ] **Step 1: Write the failing test**

Create `tests/test_resolve_image_files.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_resolve_image_files.py --import-mode=importlib -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.imagemodels.serve_resolve'`).

- [ ] **Step 3: Create `serve_resolve.py`**

Create `src/imagemodels/serve_resolve.py` (the four hint strings are copied verbatim from the current route so the messages don't change):

```python
"""Resolve the sd-server `files` dict for a local image GGUF: detect the model
family and gather its encoders/VAE. Extracted from the imagemodels serve route so
BOTH the route and the on-demand auto-serve path share one implementation."""
import os

from src.gguf_meta import gguf_is_full_checkpoint, read_gguf_architecture
from src.imagemodels.runtime import looks_like_flux2, looks_like_chroma
from src.imagemodels.encoders import (
    resolve_flux_files, resolve_flux2_files, resolve_zimage_files,
    resolve_chroma_files, MissingEncoderError,
)

_MISSING_HINTS = {
    "Z-Image": ("Missing Z-Image files: {missing}. Z-Image needs a Qwen3-4B text "
                "encoder (llm, e.g. Qwen3-4B-Q4_K_M.gguf) and the FLUX.1 VAE (vae, "
                "ae.safetensors) next to the model or in the shared encoders folder."),
    "FLUX.2": ("Missing FLUX.2 files: {missing}. klein needs a Qwen3 text encoder "
               "(llm, e.g. Qwen3-4B-Q4_K_M.gguf) and the FLUX.2 VAE (vae, "
               "flux2_ae.safetensors) next to the model or in the shared encoders folder."),
    "Chroma": ("Missing Chroma files: {missing}. Chroma needs a T5-XXL text encoder "
               "(t5xxl, e.g. t5xxl_q8_0.gguf) and the FLUX.1 VAE (vae, ae.safetensors) "
               "next to the model or in the shared encoders folder."),
    "FLUX": ("Missing FLUX files: {missing}. Put t5xxl / clip_l / vae next to the "
             "model, or download the FLUX encoders."),
}


def resolve_image_files(model_path, *, llm=None, vae=None, t5xxl=None, clip_l=None) -> dict:
    """Return the sd-server `files` dict for `model_path`. Raises MissingEncoderError
    (with a `.hint` string) when a required encoder/VAE can't be found."""
    real = os.path.realpath(model_path)
    base = os.path.basename(real).lower()

    # All-in-one SD/SDXL checkpoint (embedded encoders+VAE): self-contained.
    if gguf_is_full_checkpoint(real):
        return {"checkpoint": real}

    arch = read_gguf_architecture(real)
    if arch == "lumina2" or "z-image" in base or "z_image" in base:
        kind, call = "Z-Image", lambda: resolve_zimage_files(real, llm=llm, vae=vae)
    elif looks_like_flux2(real):
        kind, call = "FLUX.2", lambda: resolve_flux2_files(real, llm=llm, vae=vae)
    elif arch == "chroma" or looks_like_chroma(real):
        kind, call = "Chroma", lambda: resolve_chroma_files(real, t5xxl=t5xxl, vae=vae)
    else:
        kind, call = "FLUX", lambda: resolve_flux_files(real, t5xxl=t5xxl, clip_l=clip_l, vae=vae)

    try:
        return call()
    except MissingEncoderError as e:
        e.hint = _MISSING_HINTS[kind].format(missing=", ".join(e.missing))
        raise
```

- [ ] **Step 4: Refactor the route to use it (behavior-preserving)**

In `routes/imagemodels_routes.py`, replace the import block (lines 14-20) so the route no longer imports the moved helpers, keeping only what it still uses:

```python
from src.imagemodels.manager import get_manager
from src.imagemodels.serve_resolve import resolve_image_files
from src.imagemodels.encoders import MissingEncoderError
```

Then replace the entire files-building `if/elif` chain in `serve()` (the block from `base = os.path.basename(real).lower()` through the final FLUX `else:` — current lines 44-98) with:

```python
        try:
            files = resolve_image_files(
                real, llm=payload.get("llm"), vae=payload.get("vae"),
                t5xxl=payload.get("t5xxl"), clip_l=payload.get("clip_l"))
        except MissingEncoderError as e:
            raise HTTPException(400, getattr(e, "hint", str(e)))
```

Leave everything after it (the `steps` clamp, the `fast_decode`/`find_taesd` block, and the `await asyncio.to_thread(get_manager().start, files, device, steps)` call) unchanged. `find_taesd` is still imported inline at its use site, so no import change is needed for it.

- [ ] **Step 5: Run the resolver tests + the route's existing tests**

Run: `python -m pytest tests/test_resolve_image_files.py --import-mode=importlib -q`
Expected: PASS (7 passed).

Run the existing imagemodels route/manager tests to confirm the refactor didn't change behavior:
Run: `python -m pytest tests/test_imagemodels_runtime.py tests/test_imagemodels_manager.py --import-mode=importlib -q`
Expected: PASS (unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/imagemodels/serve_resolve.py routes/imagemodels_routes.py tests/test_resolve_image_files.py
git commit -m "refactor(imagemodels): extract resolve_image_files (shared by route + auto-serve)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `ensure_image_served`

**Files:**
- Modify: `src/imagemodels/manager.py` (add the function)
- Test: `tests/test_ensure_image_served.py`

**Interfaces:**
- Consumes (Task 1): `src.imagemodels.serve_resolve.resolve_image_files`; `MissingEncoderError`; `get_manager()`; `load_settings()` (`src/settings.py`).
- Produces: `ensure_image_served(owner=None, *, settings=None, manager=None, resolver=None, lister=None) -> dict` returning `{"model": <served id>|None, "error": <str>|None}`. Never raises.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ensure_image_served.py`:

```python
import src.imagemodels.manager as im
from src.imagemodels.encoders import MissingEncoderError


class SpyManager:
    def __init__(self, running=None):
        self._running = running          # a model id string, or None
        self.started = []
    def status(self):
        if self._running:
            return {"running": True, "model": self._running}
        return {"running": False, "model": None}
    def list_models(self):
        return [{"name": "flux.gguf", "path": "C:/img/flux.gguf", "size": 1}]
    def start(self, files, device="cpu", steps=None):
        self.started.append((files, device))
        return {"running": True, "model": "flux.gguf"}


def _call(**kw):
    kw.setdefault("resolver", lambda p: {"diffusion_model": p})
    return im.ensure_image_served("admin", **kw)


def test_reuses_already_running_without_start():
    mgr = SpyManager(running="already.gguf")
    out = _call(manager=mgr, settings={"image_model": "flux.gguf"})
    assert out == {"model": "already.gguf", "error": None}
    assert mgr.started == []          # no swap


def test_serves_local_default_when_idle():
    mgr = SpyManager(running=None)
    out = _call(manager=mgr, settings={"image_model": "flux.gguf"},
                lister=mgr.list_models)
    assert out["model"] == "flux.gguf" and out["error"] is None
    assert len(mgr.started) == 1 and mgr.started[0][1] == "gpu"


def test_external_default_is_noop():
    mgr = SpyManager(running=None)
    out = _call(manager=mgr, settings={"image_model": "gpt-image-1.5"},
                lister=mgr.list_models)
    assert out == {"model": None, "error": None}
    assert mgr.started == []


def test_unset_default_is_noop():
    mgr = SpyManager(running=None)
    out = _call(manager=mgr, settings={}, lister=mgr.list_models)
    assert out == {"model": None, "error": None}


def test_start_failure_returns_error_never_raises():
    mgr = SpyManager(running=None)
    def boom(files, device="cpu", steps=None):
        raise RuntimeError("sd-server did not start")
    mgr.start = boom
    out = _call(manager=mgr, settings={"image_model": "flux.gguf"},
                lister=mgr.list_models)
    assert out["model"] is None and "did not start" in out["error"]


def test_missing_encoder_returns_hint():
    mgr = SpyManager(running=None)
    def boom(p):
        e = MissingEncoderError(["t5xxl"])
        e.hint = "Missing FLUX files: t5xxl. Put t5xxl / clip_l / vae ..."
        raise e
    out = _call(manager=mgr, settings={"image_model": "flux.gguf"},
                lister=mgr.list_models, resolver=boom)
    assert out["model"] is None and "t5xxl" in out["error"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_ensure_image_served.py --import-mode=importlib -q`
Expected: FAIL (`AttributeError: module 'src.imagemodels.manager' has no attribute 'ensure_image_served'`).

- [ ] **Step 3: Implement `ensure_image_served`**

Add to `src/imagemodels/manager.py` (below `get_manager`):

```python
def ensure_image_served(owner=None, *, settings=None, manager=None,
                        resolver=None, lister=None) -> dict:
    """Ensure a local image model is serving so do_generate_image can resolve one.

    Mirrors ensure_vision_served. Returns {"model": <served id>|None,
    "error": <str>|None} and NEVER raises. Behavior:
      - if the manager is already serving a local image model, reuse it (no swap);
      - else, if the configured default `image_model` names a LOCAL model, resolve
        its encoders and serve it on the GPU (falling back to CPU via start());
      - else (external/unset/unmatched default), return {"model": None} so the
        existing external/auto-detect path in do_generate_image runs unchanged.
    """
    import os
    from src.imagemodels.serve_resolve import resolve_image_files
    from src.imagemodels.encoders import MissingEncoderError

    mgr = manager or get_manager()
    try:
        st = mgr.status()
        if st.get("running"):
            return {"model": st.get("model"), "error": None}
    except Exception:
        pass  # fall through and try to serve the default

    try:
        if settings is None:
            from src.settings import load_settings
            settings = load_settings()
        default = str(settings.get("image_model") or "").strip()
    except Exception:
        default = ""
    if not default:
        return {"model": None, "error": None}

    try:
        models = (lister or mgr.list_models)()
    except Exception:
        models = []
    match = None
    dl = default.lower()
    for m in models:
        name = str(m.get("name") or "").lower()
        pbase = os.path.basename(str(m.get("path") or "")).lower()
        if dl in (name, pbase):
            match = m
            break
    if not match:
        return {"model": None, "error": None}  # external / not a local model

    try:
        files = (resolver or resolve_image_files)(match["path"])
    except MissingEncoderError as e:
        return {"model": None, "error": getattr(e, "hint", str(e))}
    except Exception as e:
        return {"model": None, "error": f"could not resolve image model files: {e}"}

    try:
        status = mgr.start(files, device="gpu")
        return {"model": status.get("model"), "error": None}
    except Exception as e:
        return {"model": None, "error": f"could not serve image model: {e}"}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ensure_image_served.py --import-mode=importlib -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/imagemodels/manager.py tests/test_ensure_image_served.py
git commit -m "feat(imagemodels): ensure_image_served — auto-serve the local default image model

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Wire auto-serve into `do_generate_image`

**Files:**
- Modify: `src/ai_interaction.py` (add `_apply_image_autoserve` helper; call it in `do_generate_image`)
- Test: `tests/test_apply_image_autoserve.py`

**Interfaces:**
- Consumes (Task 2): `src.imagemodels.manager.ensure_image_served`.
- Produces: `async _apply_image_autoserve(model_spec, explicit, owner, *, ensure=None) -> tuple[str, str | None]` returning `(model_spec_to_use, error_or_None)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_apply_image_autoserve.py`:

```python
import asyncio
import src.ai_interaction as ai


def _run(coro):
    return asyncio.run(coro)


def test_explicit_model_skips_autoserve():
    calls = []
    def ensure(owner):
        calls.append(owner)
        return {"model": "should-not-be-used", "error": None}
    spec, err = _run(ai._apply_image_autoserve("gpt-image-1.5", True, "u", ensure=ensure))
    assert spec == "gpt-image-1.5" and err is None
    assert calls == []                      # ensure not called when explicit


def test_local_default_uses_served_id():
    def ensure(owner):
        return {"model": "flux.gguf", "error": None}
    spec, err = _run(ai._apply_image_autoserve("", False, "u", ensure=ensure))
    assert spec == "flux.gguf" and err is None


def test_serve_error_is_returned():
    def ensure(owner):
        return {"model": None, "error": "sd-server did not start"}
    spec, err = _run(ai._apply_image_autoserve("", False, "u", ensure=ensure))
    assert err == "sd-server did not start"


def test_external_default_leaves_spec_untouched():
    def ensure(owner):
        return {"model": None, "error": None}
    spec, err = _run(ai._apply_image_autoserve("gpt-image-1.5", False, "u", ensure=ensure))
    assert spec == "gpt-image-1.5" and err is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_apply_image_autoserve.py --import-mode=importlib -q`
Expected: FAIL (`AttributeError: module 'src.ai_interaction' has no attribute '_apply_image_autoserve'`).

- [ ] **Step 3: Add the helper + call it in `do_generate_image`**

In `src/ai_interaction.py`, add the helper (near `do_generate_image`):

```python
async def _apply_image_autoserve(model_spec, explicit, owner, *, ensure=None):
    """When the caller did not name a model explicitly, auto-serve the configured
    local default image model and return its id to use. Returns
    (model_spec_to_use, error_or_None). Never raises (ensure_image_served doesn't)."""
    if explicit:
        return model_spec, None
    if ensure is None:
        from src.imagemodels.manager import ensure_image_served as ensure
    served = await asyncio.to_thread(ensure, owner)
    if served.get("error"):
        return model_spec, served["error"]
    if served.get("model"):
        return served["model"], None
    return model_spec, None
```

Then in `do_generate_image`, capture whether the model was explicit and apply auto-serve. The model line is parsed at `model_spec = lines[1].strip() ...` (line ~936) and the admin default is applied at `if not model_spec: model_spec = _settings.get("image_model", "")` (line ~954). Immediately AFTER that admin-default block and the quality block (i.e. right before the `# Auto-detect best available image model` block at line ~958), insert:

```python
        # Auto-serve the configured local default image model on demand (mirrors
        # ensure_vision_served) so a text-model chat can generate an image without
        # the user manually serving the model first. Only when no explicit model
        # was named in the tool call.
        _explicit_model = bool(lines[1].strip()) if len(lines) > 1 else False
        model_spec, _autoserve_err = await _apply_image_autoserve(
            model_spec, _explicit_model, owner)
        if _autoserve_err:
            return {"error": _autoserve_err}
```

Everything after (the auto-detect loop, `_resolve_model`, the httpx generation) stays unchanged — it now sees `model_spec` set to the freshly-served local model id when applicable.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_apply_image_autoserve.py --import-mode=importlib -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ai_interaction.py tests/test_apply_image_autoserve.py
git commit -m "feat(imagegen): auto-serve local default image model in do_generate_image

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Make `generate_image` reachable when a default image model is set

**Files:**
- Modify: `src/agent_loop.py` (the gate at line ~1980)
- Test: `tests/test_generate_image_gate.py`

**Interfaces:**
- Consumes: `get_setting` (already imported in `agent_loop.py`).
- Produces: `src.agent_loop._generate_image_hidden() -> bool` (True when `generate_image` should be dropped from the toolset).

- [ ] **Step 1: Write the failing test**

Create `tests/test_generate_image_gate.py`:

```python
import src.agent_loop as al


def _patch(monkeypatch, enabled, image_model):
    def fake_get_setting(key, default=None):
        if key == "image_gen_enabled":
            return enabled
        if key == "image_model":
            return image_model
        return default
    monkeypatch.setattr(al, "get_setting", fake_get_setting)


def test_hidden_when_disabled_and_no_default(monkeypatch):
    _patch(monkeypatch, False, "")
    assert al._generate_image_hidden() is True


def test_available_when_default_image_model_set(monkeypatch):
    _patch(monkeypatch, False, "flux.gguf")
    assert al._generate_image_hidden() is False


def test_available_when_enabled(monkeypatch):
    _patch(monkeypatch, True, "")
    assert al._generate_image_hidden() is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_generate_image_gate.py --import-mode=importlib -q`
Expected: FAIL (`AttributeError: module 'src.agent_loop' has no attribute '_generate_image_hidden'`).

- [ ] **Step 3: Add the helper + use it at the gate**

In `src/agent_loop.py`, add the helper (module level, near the other tool-gating helpers):

```python
def _generate_image_hidden() -> bool:
    """generate_image is hidden only when image gen is disabled AND no default
    image model is configured. A configured default is a clear signal the user
    wants image generation, so the tool stays available for a text-model chat."""
    if get_setting("image_gen_enabled", False):
        return False
    return not str(get_setting("image_model", "") or "").strip()
```

Then replace the current gate (lines 1980-1981):

```python
    if not get_setting("image_gen_enabled", False):
        disabled.add("generate_image")
```

with:

```python
    if _generate_image_hidden():
        disabled.add("generate_image")
```

- [ ] **Step 4: Run the tests + import smoke**

Run: `python -m pytest tests/test_generate_image_gate.py --import-mode=importlib -q`
Expected: PASS (3 passed).
Run: `python -c "import app"`
Expected: no error.

- [ ] **Step 5: Run the whole feature suite (no regression)**

Run: `python -m pytest tests/test_resolve_image_files.py tests/test_ensure_image_served.py tests/test_apply_image_autoserve.py tests/test_generate_image_gate.py tests/test_imagemodels_runtime.py tests/test_imagemodels_manager.py --import-mode=importlib -q`
Expected: PASS (the 4 new files + the two existing imagemodels suites green).

- [ ] **Step 6: Commit**

```bash
git add src/agent_loop.py tests/test_generate_image_gate.py
git commit -m "feat(imagegen): expose generate_image when a default image model is configured

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **All TDD, all headless.** No GPU, no real sd-server, no real generation. Task 1 injects the arch-detectors + resolvers; Task 2 injects a spy manager + fakes; Task 3 tests the pure decision helper (no httpx); Task 4 injects `get_setting`.
- **Behavior-preserving route refactor (Task 1)** — the per-arch "missing encoder" messages and the payload encoder overrides must be identical to before. The route's existing tests must stay green.
- **`ensure_image_served` never raises** — every external call (`status`, `list_models`, resolve, `start`) is wrapped; a failure becomes `{"error": …}`. Do not let it throw into `do_generate_image`.
- **Non-disruptive + explicit-wins** — reuse a running image model rather than swapping; skip auto-serve entirely when the tool call names a model.
- **No new dependencies, no `Assist.spec` change.** A frozen boot-check after a rebuild confirms the new module imports; expected clean.
- **Owed by the user (manual, hardware):** the real end-to-end path — chat with a text model, ask for an image, confirm the configured local default auto-serves and returns an image on the 6GB GPU. The automated tests prove the serve-decision + routing plumbing, not real image output.
- **Scope:** just auto-serve the local default + reach the tool. No sd-server change, no multi-model concurrency, no new UI, no change to the external-API path, no auto-download of missing models/encoders.
