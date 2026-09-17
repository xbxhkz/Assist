# Unslothed Capability Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the model ask which tools, installed software or model abilities can accomplish a task, which is best, and whether it works right now — and see what is missing.

**Architecture:** A new `core/inference/capability_map/` package holds a curated vocabulary of 22 capabilities, each with providers in preference order. Every provider declares requirements (a tool, a Python module, a program on PATH, a model ability, or a cached model file); each requirement resolves to piece 3a's `Readiness`, and fixed rules roll those up into provider and capability status. A new `find_capability` tool renders the result in master spec §38's format.

**Tech Stack:** Python 3.12, pytest. No new dependencies.

**Spec:** `C:\Users\Admin\odysseus\docs\superpowers\specs\2026-09-17-unslothed-capability-map-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Repo:** `C:\Users\Admin\unsloth`. Create and work on branch `capability-map` off `unslothed-release` (currently `6b552aba`). `origin` is UPSTREAM `unslothai/unsloth` and **must never be pushed to**; the fork remote is `fork`. Commit locally only.
- **`studio/backend/core/inference/tools.py` must stay PURELY ADDITIVE against upstream: zero deletions.** It is at **88 insertions / 0 deletions**; this plan takes it to **93 / 0**. A line the FORK inserted may have its content amended (it still counts as one insertion); a line UPSTREAM owns must never be edited. Verify with:
  `git diff --numstat $(git merge-base origin/main HEAD) -- studio/backend/core/inference/tools.py`
- **`studio/backend/routes/inference.py`** is at **58 insertions / 0 deletions**; this plan takes it to **59 / 0**. Same rule: only fork-inserted lines may be amended. Verify the same way.
- **Do-not-edit:** `core/inference/llama_cpp.py`, `pyproject.toml`, `studio/backend/main.py` (stays 4 / 0).
- **NEVER run the full backend test suite.** Upstream fixtures fabricate GGUF files up to 40 GB and filled the disk once. Run only the test files each task names.
- **Test runner**, from `C:\Users\Admin\unsloth\studio\backend`:
  `C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest <files> -v -p no:cacheprovider`
- **Probes must never create or import what they ask about.** Never call `get_inference_backend()`, `get_llama_cpp_backend()`, `get_diffusion_backend()`, `get_sd_cpp_backend()` or `get_active_diffusion_engine()` — each constructs an instance. Never import `routes.inference`. Read already-loaded state through `sys.modules`.
- **Never-raises guards use a bare `except BaseException`**, not `except Exception` (this project found `OverflowError` and unhashable types slipping past the narrower form).
- **Every negative control is demonstrated to fail.** Ten inert controls have appeared in this project, several authored by the plan's author. **Run mutation controls AFTER committing the task**, restore each mutation immediately with `git checkout -- <file>`, and finish with `git status --porcelain` showing no modified tracked files. A verification agent killed mid-mutation once left a Critical fix silently reverted in the working tree.
- **Style:** keyword arguments take spaces around `=` (`refresh = True`).
- **Commit trailer:** end every commit message with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## File Structure

**Created:**

| Path | Responsibility |
|---|---|
| `studio/backend/core/inference/capability_map/__init__.py` | data model, matching, status rules; Task 5 adds output + `execute()` |
| `studio/backend/core/inference/capability_map/providers.py` | `check_requirement()` for every requirement kind |
| `studio/backend/core/inference/capability_map/vocabulary.py` | the 22 curated capabilities |
| `studio/backend/core/inference/capability_map/schemas.py` | `find_capability` schema, `CAPABILITY_TOOLS`, `CAPABILITY_TOOL_NAMES` |
| `studio/backend/tests/test_capability_map_core.py` | Task 1 |
| `studio/backend/tests/test_capability_map_providers.py` | Task 2 |
| `studio/backend/tests/test_capability_map_vocabulary.py` | Task 4 |
| `studio/backend/tests/test_capability_map_tool.py` | Task 5 |
| `studio/backend/tests/test_capability_map_seam.py` | Task 6 |

**Modified:**

| Path | Change |
|---|---|
| `studio/backend/core/inference/tool_readiness/probes.py` | Task 3: probes for `remove_background`, `detect_shapes`, `edit_image_prompt`. Task 6: `find_capability` joins `ALWAYS_READY_TOOLS` |
| `studio/backend/tests/test_tool_readiness_probes.py` | Task 3: new probe tests + rewrite of a test the new probes invalidate. Task 6: `find_capability` in the always-ready assertion |
| `studio/backend/core/inference/tools.py` | Task 6: +5 lines, one fork line amended |
| `studio/backend/routes/inference.py` | Task 6: +1 line, two fork lines amended |
| `studio/backend/tests/test_tool_audit_seam.py` | Task 6: ceiling 90 → 100 |
| `studio/backend/tests/test_tool_readiness_seam.py` | Task 6: ceiling 90 → 100 |

## Facts verified against live code while writing this plan

Implementers should not need to re-derive these, but reviewers may check them.

- `tool_readiness` exports `Readiness(state, detail, missing = None, remedy = None)`, `READY`/`MISSING`/`UNKNOWN`, `register`, `register_default`, `resolve(name, *, refresh = False)`, `reset_for_tests`, `_all_tool_names`, `execute`. `resolve` caches for 60 s and never raises. `probes.install_default_probes()` uses `register_default`, so an explicit `register` always wins.
- `probes._module_present(name)` wraps `importlib.util.find_spec` and never raises.
- `routes/inference.py:5113` — `_llama_cpp_backend = LlamaCppBackend()`. Its `is_loaded` is `self._process is not None and self._healthy`, `is_vision` is `self._is_vision and self._mmproj_accepts_image`, `model_identifier` returns `self._model_identifier` — all attribute reads.
- `core/inference/orchestrator.py:2465` — `peek_inference_backend()`: *"The orchestrator if one exists, else None. Never constructs one."* A loaded model's entry is `self.models[self.active_model_name]` with keys including `"is_vision"` and `"display_name"`.
- `core/inference/diffusion_engine_router.py:65` — `_active_engine_name`, with `ENGINE_DIFFUSERS = "diffusers"` and `ENGINE_SD_CPP = "sd_cpp"` imported from `sd_cpp_engine`. The router is imported lazily, not at startup. `core/inference/diffusion.py:6301` — `_diffusion_backend: Optional[DiffusionBackend] = None`; its `is_loaded` is `self._state is not None`. The native sd.cpp engine rejects image-to-image (`sd_cpp_backend.py:2153`).
- `assist_vision/bg_removal.py:42` — `_model_path()` returns `$UNSLOTH_U2NET_PATH` if set (`_PATH_ENV`, line 37), else `model_path("u2net.onnx")`; `onnxruntime` is imported lazily.
- **Line numbers in this plan refer to each file BEFORE that task's own edits.** Every edit also gives the exact text to find, which is authoritative if a line number has shifted.
- Mask R-CNN weight: `maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth` under `torch.hub.get_dir()/checkpoints/`. `torch` is imported at module level by `core/inference/inference.py`, so it is in `sys.modules` in a running backend.
- Whisper: `load_model` builds `default = os.path.join(os.path.expanduser("~"), ".cache")` then `download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")`; files are named `<model>.pt`.
- `tests/test_tool_readiness_probes.py:224` asserts `detect_shapes`, `edit_image_prompt` and `remove_background` report `unknown`. **Task 3 makes that false** and rewrites it.
- `tests/test_tool_readiness_seam.py::_route_addable()` resolves each name in the route's single-line `_addable = A | B | C` via `getattr(core.inference.tools, name)`. **The amended `_addable` line must stay on ONE line.**
- `tests/` is a package, so `tests.test_tool_readiness_seam._anthropic_gate_genexp` is importable.

---

### Task 1: Data model, matching and status rules

**Files:**
- Create: `studio/backend/core/inference/capability_map/__init__.py`
- Create: `studio/backend/core/inference/capability_map/providers.py`
- Test: `studio/backend/tests/test_capability_map_core.py`

**Interfaces:**
- Consumes: `Readiness`, `READY`, `MISSING`, `UNKNOWN`, `register`, `resolve`, `reset_for_tests` from `core.inference.tool_readiness`; `probes._module_present`, `probes.install_default_probes`
- Produces (in `core.inference.capability_map`):
  - `Requirement(kind: str, name: str)` — frozen dataclass
  - `Provider(kind: str, name: str, via: Optional[str], requires: tuple[Requirement, ...], reason: str)`
  - `Capability(name: str, title: str, aliases: tuple[str, ...], providers: tuple[Provider, ...], acquire: Optional[str])`
  - `ProviderResult(provider: Provider, state: str, detail: str)`
  - `CapabilityResult(capability: Capability, state: str, best: Optional[ProviderResult], providers: tuple[ProviderResult, ...])`
  - `normalize(text) -> str`, `find(query, capabilities) -> Optional[Capability]`, `requirements_for(provider) -> tuple[Requirement, ...]`, `resolve_provider(provider) -> ProviderResult`, `resolve_capability(capability) -> CapabilityResult`
- Produces (in `core.inference.capability_map.providers`): `check_requirement(req) -> Readiness` (never raises), `_check_tool`, `_check_module`, `_check_binary`, `_which`, `_CHECK_NAMES`

- [ ] **Step 1: Create the branch**

```bash
cd C:/Users/Admin/unsloth
git checkout unslothed-release
git checkout -b capability-map
```

- [ ] **Step 2: Write the failing test**

Create `studio/backend/tests/test_capability_map_core.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Data model, matching and status rules for the capability map.

The rule everything rests on is carried up from tool readiness: an `unknown`
never makes anything `ready`. A capability is ready only when some provider was
actually checked and found ready.
"""

from __future__ import annotations

import pytest

from core.inference import capability_map as cm
from core.inference import tool_readiness as tr
from core.inference.capability_map import providers
from core.inference.tool_readiness import MISSING, READY, UNKNOWN, Readiness, probes


@pytest.fixture(autouse = True)
def _clean():
    tr.reset_for_tests()
    yield
    tr.reset_for_tests()


def _req(kind, name):
    return cm.Requirement(kind, name)


def _fake_checks(monkeypatch, states):
    """Every requirement resolves to states[req.name]."""

    def fake(req):
        return Readiness(states[req.name], f"{req.name} is {states[req.name]}")

    monkeypatch.setattr(providers, "check_requirement", fake)


def _provider(name, *module_names, via = None):
    return cm.Provider(
        "software" if via else "tool",
        name,
        via,
        tuple(_req("module", n) for n in module_names),
        f"reason for {name}",
    )


def _cap(*provs, acquire = None):
    return cm.Capability("c", "C", (), tuple(provs), acquire)


# --- matching ---------------------------------------------------------------


def test_normalize_ignores_case_and_separators():
    assert cm.normalize("  Video_Editing ") == "video editing"
    assert cm.normalize("speech-to-text") == "speech to text"


def test_find_matches_a_name_or_an_alias():
    caps = (cm.Capability("ocr", "Read text", ("text recognition",), (), None),)
    assert cm.find("OCR", caps) is caps[0]
    assert cm.find("Text-Recognition", caps) is caps[0]


def test_find_returns_none_for_no_match_or_an_empty_query():
    caps = (cm.Capability("ocr", "Read text", (), (), None),)
    assert cm.find("video", caps) is None
    assert cm.find("   ", caps) is None


# --- provider status --------------------------------------------------------


def test_a_provider_is_ready_only_when_every_requirement_is(monkeypatch):
    _fake_checks(monkeypatch, {"a": READY, "b": READY})
    assert cm.resolve_provider(_provider("p", "a", "b")).state == READY


def test_one_missing_requirement_makes_the_provider_missing_even_beside_an_unknown(monkeypatch):
    _fake_checks(monkeypatch, {"a": UNKNOWN, "b": MISSING})
    result = cm.resolve_provider(_provider("p", "a", "b"))
    assert result.state == MISSING
    assert "b is missing" in result.detail


def test_an_unknown_requirement_without_a_missing_one_is_unknown(monkeypatch):
    _fake_checks(monkeypatch, {"a": READY, "b": UNKNOWN})
    assert cm.resolve_provider(_provider("p", "a", "b")).state == UNKNOWN


def test_a_provider_with_no_requirements_is_unknown_not_vacuously_ready():
    """all() over nothing is True. Reporting 'ready' for a provider nobody
    declared any requirement for is the optimistic lie in its purest form."""
    assert cm.resolve_provider(_provider("p")).state == UNKNOWN


def test_a_software_provider_needs_its_via_tool(monkeypatch):
    _fake_checks(monkeypatch, {"python": MISSING, "fitz": READY})
    assert cm.resolve_provider(_provider("PyMuPDF", "fitz", via = "python")).state == MISSING


def test_the_via_tool_being_ready_lets_the_software_provider_be_ready(monkeypatch):
    """Control for the test above."""
    _fake_checks(monkeypatch, {"python": READY, "fitz": READY})
    result = cm.resolve_provider(_provider("PyMuPDF", "fitz", via = "python"))
    assert result.state == READY
    assert result.detail == "fitz is ready", (
        "a ready provider reports its OWN requirements, not its via tool's"
    )


# --- capability status ------------------------------------------------------


def test_best_option_is_the_first_READY_provider_not_the_first_listed(monkeypatch):
    _fake_checks(monkeypatch, {"a": UNKNOWN, "b": READY, "c": READY})
    result = cm.resolve_capability(_cap(_provider("A", "a"), _provider("B", "b"), _provider("C", "c")))
    assert result.state == READY
    assert result.best.provider.name == "B"


def test_an_unknown_provider_never_makes_a_capability_ready(monkeypatch):
    _fake_checks(monkeypatch, {"a": UNKNOWN, "b": MISSING})
    result = cm.resolve_capability(_cap(_provider("A", "a"), _provider("B", "b")))
    assert result.state == UNKNOWN
    assert result.best is None


def test_every_provider_missing_makes_the_capability_missing(monkeypatch):
    _fake_checks(monkeypatch, {"a": MISSING, "b": MISSING})
    result = cm.resolve_capability(_cap(_provider("A", "a"), _provider("B", "b")))
    assert result.state == MISSING


def test_a_capability_with_no_providers_is_missing():
    assert cm.resolve_capability(_cap()).state == MISSING


# --- requirement checks -----------------------------------------------------


def test_a_module_requirement_delegates_to_the_readiness_module_check(monkeypatch):
    monkeypatch.setattr(probes, "_module_present", lambda name: name == "fitz")
    assert providers.check_requirement(_req("module", "fitz")).state == READY
    assert providers.check_requirement(_req("module", "easyocr")).state == MISSING


def test_a_binary_requirement_reads_PATH_and_explains_a_restart(monkeypatch):
    monkeypatch.setattr(
        providers, "_which", lambda name: "C:/bin/ffmpeg.exe" if name == "ffmpeg" else None
    )
    assert providers.check_requirement(_req("binary", "ffmpeg")).state == READY
    result = providers.check_requirement(_req("binary", "tesseract"))
    assert result.state == MISSING
    assert "restart" in result.detail, (
        "a program installed after startup is invisible until restart; say so"
    )


def test_a_tool_requirement_uses_tool_readiness_and_an_explicit_probe_wins():
    """install_default_probes() runs inside the check; register_default must not
    clobber this explicit registration (3a's Ruling 1)."""
    tr.register("web_search", lambda: Readiness(MISSING, "ddgs gone", remedy = "pip install ddgs"))
    result = providers.check_requirement(_req("tool", "web_search"))
    assert result.state == MISSING
    assert "ddgs gone" in result.detail


def test_a_tool_requirements_remedy_is_not_carried_into_the_map():
    """§4 boundary. 3a's remedies include install commands; the map never repeats
    them, so a model reading the map is never handed one."""
    tr.register("web_search", lambda: Readiness(MISSING, "ddgs gone", remedy = "pip install ddgs"))
    result = providers.check_requirement(_req("tool", "web_search"))
    assert result.remedy is None
    assert "pip install" not in result.detail


def test_an_unrecognised_requirement_kind_is_unknown():
    assert providers.check_requirement(_req("telepathy", "x")).state == UNKNOWN


def test_a_raising_check_reports_unknown_and_never_raises(monkeypatch):
    """Also proves dispatch looks checks up BY NAME at call time: if the dispatch
    table held function objects captured at import, this monkeypatch would not
    take effect and the real PATH lookup would run instead."""

    def boom(name):
        raise OverflowError("nope")

    monkeypatch.setattr(providers, "_check_binary", boom)
    result = providers.check_requirement(_req("binary", "ffmpeg"))
    assert result.state == UNKNOWN
    assert "nope" in result.detail
```

- [ ] **Step 3: Run it and watch it fail**

From `C:\Users\Admin\unsloth\studio\backend`:
```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_capability_map_core.py -v -p no:cacheprovider
```
Expected: collection error — `cannot import name 'capability_map' from 'core.inference'` (or `ModuleNotFoundError`).

- [ ] **Step 4: Create the package**

Create `studio/backend/core/inference/capability_map/__init__.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which tools, installed software or model abilities can do something, which is
best, and whether it works right now (master spec §37/§38).

A CAPABILITY ("ocr") lists PROVIDERS in curated preference order. A provider is a
tool, software reached through the python/terminal tools, or an ability of the
loaded model, and it declares REQUIREMENTS. Every requirement resolves to tool
readiness's three states, and these rules roll them up:

  provider   -- ready: every requirement ready (and there is at least one)
               missing: any requirement missing
               unknown: otherwise
  capability -- ready: some provider ready; BEST = the first ready one in order
               missing: every provider missing, or there are none
               unknown: otherwise

An `unknown` never makes anything `ready`, at either level. That is readiness's
central claim carried up: the model must never read "nobody checked" as
"verified working".

This package depends on tool_readiness; tool_readiness must never depend on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from core.inference.tool_readiness import MISSING, READY, UNKNOWN


@dataclass(frozen = True)
class Requirement:
    kind: str  # "tool" | "module" | "binary" | "model" | "cached_model"
    name: str


@dataclass(frozen = True)
class Provider:
    kind: str  # "tool" | "software" | "model" -- for display
    name: str
    via: Optional[str]  # "python" / "terminal" for software; None otherwise
    requires: tuple[Requirement, ...]
    reason: str


@dataclass(frozen = True)
class Capability:
    name: str
    title: str
    aliases: tuple[str, ...]
    providers: tuple[Provider, ...]  # curated preference order; may be empty
    acquire: Optional[str]  # §4 hook -- descriptive text, never a runnable command


@dataclass(frozen = True)
class ProviderResult:
    provider: Provider
    state: str
    detail: str


@dataclass(frozen = True)
class CapabilityResult:
    capability: Capability
    state: str
    best: Optional[ProviderResult]
    providers: tuple[ProviderResult, ...]


_SEPARATORS = re.compile(r"[\s_\-]+")


def normalize(text) -> str:
    """Case-insensitive, and spaces, underscores and hyphens are the same thing,
    so "video editing", "video_editing" and "Video-Editing" all match. Still an
    exact match after that -- no fuzzy or model-based guessing."""
    return _SEPARATORS.sub(" ", str(text)).strip().lower()


def find(query, capabilities: Sequence[Capability]) -> Optional[Capability]:
    key = normalize(query)
    if not key:
        return None
    for capability in capabilities:
        if key == normalize(capability.name):
            return capability
        if any(key == normalize(alias) for alias in capability.aliases):
            return capability
    return None


def requirements_for(provider: Provider) -> tuple[Requirement, ...]:
    """A software provider's `via` tool is an implicit first requirement:
    PyMuPDF is only usable if the python tool is."""
    implicit = (Requirement("tool", provider.via),) if provider.via else ()
    return implicit + tuple(provider.requires)


def resolve_provider(provider: Provider) -> ProviderResult:
    # Looked up at call time, so tests can substitute check_requirement.
    from core.inference.capability_map import providers as _providers

    requirements = requirements_for(provider)
    if not requirements:
        return ProviderResult(provider, UNKNOWN, "no requirement is declared for this provider")
    checked = [(req, _providers.check_requirement(req)) for req in requirements]
    for _req, readiness in checked:
        if readiness.state == MISSING:
            return ProviderResult(provider, MISSING, readiness.detail)
    for _req, readiness in checked:
        if readiness.state != READY:
            return ProviderResult(provider, UNKNOWN, readiness.detail)
    own = [readiness.detail for req, readiness in checked if req in provider.requires]
    return ProviderResult(provider, READY, "; ".join(own) if own else checked[-1][1].detail)


def resolve_capability(capability: Capability) -> CapabilityResult:
    results = tuple(resolve_provider(p) for p in capability.providers)
    for result in results:
        if result.state == READY:
            return CapabilityResult(capability, READY, result, results)
    # all() over no providers is True: a capability nothing provides is missing.
    if all(result.state == MISSING for result in results):
        return CapabilityResult(capability, MISSING, None, results)
    return CapabilityResult(capability, UNKNOWN, None, results)
```

- [ ] **Step 5: Create the requirement checks**

Create `studio/backend/core/inference/capability_map/providers.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Requirement checks for the capability map.

Every check is CHEAP -- an import-spec lookup, a PATH lookup, a file stat, or a
read of state already in memory -- and never imports, loads or creates what it
asks about.

Dispatch is BY NAME, looked up at call time (_CHECK_NAMES -> globals()). A table
of function objects captured at import would make monkeypatching a check
silently ineffective, which is the exact shape of this project's inert controls.
"""

from __future__ import annotations

import shutil

from core.inference.tool_readiness import MISSING, READY, UNKNOWN, Readiness

_CHECK_NAMES = {
    "tool": "_check_tool",
    "module": "_check_module",
    "binary": "_check_binary",
}


def _which(name: str):
    return shutil.which(name)


def _check_tool(name: str) -> Readiness:
    from core.inference import tool_readiness
    from core.inference.tool_readiness.probes import install_default_probes

    # Cheap and idempotent (register_default never clobbers). Without it the
    # registry can be empty and every tool provider would read "unknown".
    install_default_probes()
    readiness = tool_readiness.resolve(name)
    # Detail only. Readiness remedies can be install commands ("pip install
    # ddgs"); the map is a descriptive surface and must not hand those on.
    return Readiness(readiness.state, f"tool {name}: {readiness.detail}")


def _check_module(name: str) -> Readiness:
    from core.inference.tool_readiness import probes

    if probes._module_present(name):
        return Readiness(READY, f"{name} importable")
    return Readiness(MISSING, f"{name} not installed")


def _check_binary(name: str) -> Readiness:
    if _which(name):
        return Readiness(READY, f"{name} on PATH")
    return Readiness(
        MISSING,
        f"{name} not on PATH (a program installed after Unslothed started "
        "is not seen until Unslothed restarts)",
    )


def check_requirement(req) -> Readiness:
    """Readiness for one requirement. Never raises."""
    try:
        function_name = _CHECK_NAMES.get(getattr(req, "kind", None))
        check = globals().get(function_name) if function_name else None
        if check is None:
            return Readiness(UNKNOWN, f"no check for requirement kind {getattr(req, 'kind', None)!r}")
        result = check(req.name)
        if not isinstance(result, Readiness):
            return Readiness(UNKNOWN, f"check returned {type(result).__name__}, not Readiness")
        return result
    except BaseException as exc:  # noqa: BLE001 - a capability query must never break a turn
        return Readiness(UNKNOWN, f"check failed: {exc}")
```

- [ ] **Step 6: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_capability_map_core.py -v -p no:cacheprovider
```
Expected: 19 passed.

- [ ] **Step 7: Commit**

```bash
git add studio/backend/core/inference/capability_map/__init__.py studio/backend/core/inference/capability_map/providers.py studio/backend/tests/test_capability_map_core.py
git commit -m "feat(capabilities): data model, matching and status rules

A capability lists providers in curated preference order; each provider
declares requirements that resolve to tool readiness's three states. A provider
is ready only when every requirement is -- and a provider with no requirements
is unknown, never vacuously ready. A capability's best option is the first
READY provider, not the first listed, and an unknown never promotes anything.

Tool requirements drop readiness remedies: those can be install commands, and
the map must stay a descriptive surface (master spec §4).

Checks dispatch by name at call time so substituting one in a test actually
takes effect.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 8: Demonstrate the controls (after committing)**

For each mutation: make it, run `tests/test_capability_map_core.py`, read the failure, then restore with `git checkout -- <file>`.

1. In `capability_map/__init__.py` `resolve_provider`, change `return ProviderResult(provider, UNKNOWN, "no requirement is declared for this provider")` to return `READY`. Expect `test_a_provider_with_no_requirements_is_unknown_not_vacuously_ready` to FAIL.
2. In `resolve_capability`, return the FIRST provider as best regardless of state (`if results: return CapabilityResult(capability, READY, results[0], results)` before the loop). Expect `test_best_option_is_the_first_READY_provider_not_the_first_listed` and `test_an_unknown_provider_never_makes_a_capability_ready` to FAIL.
3. In `providers.py`, replace the by-name lookup with a table of function objects (`_CHECKS = {"binary": _check_binary, ...}` defined after the functions, and `check = _CHECKS.get(req.kind)`). Expect `test_a_raising_check_reports_unknown_and_never_raises` to FAIL.
4. In `_check_tool`, append the remedy to the detail (`f"tool {name}: {readiness.detail} -> {readiness.remedy}"`). Expect `test_a_tool_requirements_remedy_is_not_carried_into_the_map` to FAIL.

Finish with `git status --porcelain` — no modified tracked files.

---

### Task 2: Model abilities and cached models

**Files:**
- Modify: `studio/backend/core/inference/capability_map/providers.py`
- Test: `studio/backend/tests/test_capability_map_providers.py`

**Interfaces:**
- Consumes: `Requirement` (Task 1), `check_requirement`, `_CHECK_NAMES` (Task 1)
- Produces: `check_requirement` handles kinds `"model"` (name `"vision"`) and `"cached_model"` (name `"whisper"`); helpers `_llama_backend()`, `_orchestrator()`, `_loaded_models() -> Optional[list[tuple[str, bool]]]`, `_whisper_cache_dir() -> str`, `_check_model`, `_check_cached_model`

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_capability_map_providers.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model-ability and cached-model checks.

Both read state that is ALREADY in memory or on disk. The backend getters in this
codebase construct what they are asked about, so a check that called one would
start the subsystem it was describing.

WHY THE TRIPWIRES RECORD INSTEAD OF RAISING: check_requirement swallows every
exception into an "unknown" answer. A tripwire getter that raised would be caught
by that guard, the test would see "unknown", and nothing would fail -- an inert
control. So each fake getter appends to a list, and the tests assert the list
stays empty.
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest

from core.inference import tool_readiness as tr
from core.inference.capability_map import Requirement, providers
from core.inference.tool_readiness import MISSING, READY, UNKNOWN

_ROUTE = "routes.inference"
_ORCH = "core.inference.orchestrator"


@pytest.fixture(autouse = True)
def _no_backends_in_memory(monkeypatch):
    """Every test starts with NEITHER backend module loaded and states exactly
    what is in memory. monkeypatch restores the real modules afterwards."""
    tr.reset_for_tests()
    monkeypatch.delitem(sys.modules, _ROUTE, raising = False)
    monkeypatch.delitem(sys.modules, _ORCH, raising = False)
    yield
    tr.reset_for_tests()


def _vision():
    return providers.check_requirement(Requirement("model", "vision"))


def _route_module(monkeypatch, backend, calls):
    module = types.ModuleType(_ROUTE)
    module._llama_cpp_backend = backend

    def get_llama_cpp_backend():
        calls.append("get_llama_cpp_backend")
        return backend

    module.get_llama_cpp_backend = get_llama_cpp_backend
    monkeypatch.setitem(sys.modules, _ROUTE, module)


def _orchestrator_module(monkeypatch, orchestrator, calls):
    module = types.ModuleType(_ORCH)
    module.peek_inference_backend = lambda: orchestrator

    def get_inference_backend():
        calls.append("get_inference_backend")
        return orchestrator

    module.get_inference_backend = get_inference_backend
    monkeypatch.setitem(sys.modules, _ORCH, module)


def _llama(*, loaded, vision, name = "Qwen2.5-VL-7B"):
    return types.SimpleNamespace(is_loaded = loaded, is_vision = vision, model_identifier = name)


def _orch(active, *, vision, display = None):
    models = {active: {"is_vision": vision, "display_name": display or active}} if active else {}
    return types.SimpleNamespace(active_model_name = active, models = models)


# --- vision ---------------------------------------------------------------


def test_vision_is_unknown_when_no_backend_module_is_in_memory():
    result = _vision()
    assert result.state == UNKNOWN
    assert "backend" in result.detail


def test_vision_is_ready_and_names_a_loaded_gguf_vision_model(monkeypatch):
    _route_module(monkeypatch, _llama(loaded = True, vision = True), [])
    result = _vision()
    assert result.state == READY
    assert "Qwen2.5-VL-7B" in result.detail


def test_vision_is_missing_and_names_a_loaded_model_without_vision(monkeypatch):
    _route_module(monkeypatch, _llama(loaded = True, vision = False, name = "Llama-3.1-8B"), [])
    result = _vision()
    assert result.state == MISSING
    assert "Llama-3.1-8B" in result.detail


def test_vision_is_unknown_when_the_backend_exists_but_nothing_is_loaded(monkeypatch):
    """An external or API model may be serving, with abilities this process
    cannot see -- so this is 'unknown', not 'missing'."""
    _route_module(monkeypatch, _llama(loaded = False, vision = False), [])
    assert _vision().state == UNKNOWN


def test_vision_is_ready_from_a_transformers_model(monkeypatch):
    _orchestrator_module(monkeypatch, _orch("qwen-vl", vision = True, display = "Qwen VL"), [])
    result = _vision()
    assert result.state == READY
    assert "Qwen VL" in result.detail


def test_either_backend_having_vision_is_enough(monkeypatch):
    _route_module(monkeypatch, _llama(loaded = True, vision = False, name = "text-model"), [])
    _orchestrator_module(monkeypatch, _orch("qwen-vl", vision = True), [])
    assert _vision().state == READY


def test_the_checks_never_call_a_creating_getter(monkeypatch):
    """THE side-effect guard. Exercised on a ready path AND an unknown path."""
    calls = []
    _route_module(monkeypatch, _llama(loaded = True, vision = True), calls)
    _orchestrator_module(monkeypatch, _orch(None, vision = False), calls)
    assert _vision().state == READY
    _route_module(monkeypatch, _llama(loaded = False, vision = False), calls)
    assert _vision().state == UNKNOWN
    assert calls == [], f"a check constructed a backend: {calls}"


def test_an_ability_other_than_vision_is_unknown(monkeypatch):
    _route_module(monkeypatch, _llama(loaded = True, vision = True), [])
    assert providers.check_requirement(Requirement("model", "audio")).state == UNKNOWN


# --- cached models ----------------------------------------------------------


def test_whisper_is_missing_until_a_model_is_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    result = providers.check_requirement(Requirement("cached_model", "whisper"))
    assert result.state == MISSING
    assert "first use" in result.detail


def test_whisper_is_ready_once_a_pt_file_is_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (tmp_path / "whisper").mkdir()
    (tmp_path / "whisper" / "tiny.en.pt").write_bytes(b"x")
    result = providers.check_requirement(Requirement("cached_model", "whisper"))
    assert result.state == READY
    assert "tiny.en.pt" in result.detail


def test_a_non_model_file_in_the_whisper_cache_does_not_count(tmp_path, monkeypatch):
    """Control: without this, a check that accepted any file would pass above."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (tmp_path / "whisper").mkdir()
    (tmp_path / "whisper" / "notes.txt").write_text("x")
    assert providers.check_requirement(Requirement("cached_model", "whisper")).state == MISSING


def test_the_whisper_cache_dir_honours_XDG_CACHE_HOME(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert providers._whisper_cache_dir() == str(tmp_path / "whisper")


def test_the_pinned_whisper_cache_dir_matches_whisper_itself():
    """Drift guard. The path is pinned because importing whisper loads torch; if
    whisper ever builds download_root differently, this fails instead of the
    check silently looking in the wrong place."""
    whisper = pytest.importorskip("whisper")
    source = inspect.getsource(whisper.load_model)
    assert 'default = os.path.join(os.path.expanduser("~"), ".cache")' in source
    assert 'download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")' in source


def test_an_unrecognised_cached_model_is_unknown():
    assert providers.check_requirement(Requirement("cached_model", "llama")).state == UNKNOWN
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_capability_map_providers.py -v -p no:cacheprovider
```
Expected: most tests FAIL — `check_requirement` answers `unknown` ("no check for requirement kind 'model'") and `providers._whisper_cache_dir` does not exist.

- [ ] **Step 3: Add the checks**

In `studio/backend/core/inference/capability_map/providers.py`:

(a) Replace the import block

```python
from __future__ import annotations

import shutil

from core.inference.tool_readiness import MISSING, READY, UNKNOWN, Readiness
```

with

```python
from __future__ import annotations

import os
import shutil
import sys
from typing import Optional

from core.inference.tool_readiness import MISSING, READY, UNKNOWN, Readiness
```

(b) Replace `_CHECK_NAMES` with

```python
_CHECK_NAMES = {
    "tool": "_check_tool",
    "module": "_check_module",
    "binary": "_check_binary",
    "model": "_check_model",
    "cached_model": "_check_cached_model",
}
```

(c) Add, directly above `def check_requirement`:

```python
def _llama_backend():
    """routes.inference's resident llama.cpp backend WITHOUT importing the route
    module (a ~20,000-line router core code must not depend on) and without
    get_llama_cpp_backend(). None when the route is not loaded."""
    module = sys.modules.get("routes.inference")
    return getattr(module, "_llama_cpp_backend", None) if module is not None else None


def _orchestrator():
    """The transformers orchestrator through peek_inference_backend(), which the
    orchestrator documents as never constructing one. get_inference_backend()
    DOES construct one, blocking on the torch import."""
    module = sys.modules.get("core.inference.orchestrator")
    if module is None:
        return None
    peek = getattr(module, "peek_inference_backend", None)
    return peek() if callable(peek) else None


def _loaded_models() -> Optional[list[tuple[str, bool]]]:
    """[(model name, has vision)] for locally loaded models, or None when no
    backend module is in memory at all."""
    if "routes.inference" not in sys.modules and "core.inference.orchestrator" not in sys.modules:
        return None
    loaded: list[tuple[str, bool]] = []
    llama = _llama_backend()
    if llama is not None and llama.is_loaded:
        loaded.append((llama.model_identifier or "a GGUF model", bool(llama.is_vision)))
    orchestrator = _orchestrator()
    if orchestrator is not None:
        active = getattr(orchestrator, "active_model_name", None)
        entry = (getattr(orchestrator, "models", None) or {}).get(active) if active else None
        if entry is not None:
            loaded.append((entry.get("display_name") or active, bool(entry.get("is_vision"))))
    return loaded


def _check_model(name: str) -> Readiness:
    if name != "vision":
        # Audio is deliberately absent: llama.cpp's only audio flag routes TTS
        # OUTPUT, and reading it as audio understanding would be a false ready.
        return Readiness(UNKNOWN, f"no check for model ability {name!r}")
    loaded = _loaded_models()
    if loaded is None:
        return Readiness(UNKNOWN, "no local model backend is running in this process")
    if not loaded:
        return Readiness(UNKNOWN, "no local model is loaded; an external model may be serving")
    with_vision = [model for model, has_vision in loaded if has_vision]
    if with_vision:
        return Readiness(READY, f"{with_vision[0]} is loaded")
    names = ", ".join(model for model, _ in loaded)
    return Readiness(MISSING, f"{names} is loaded, without vision")


def _whisper_cache_dir() -> str:
    """Where whisper.load_model downloads models. PINNED rather than asked, because
    importing whisper loads torch; a test asserts whisper still builds it this way."""
    default = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")


def _check_cached_model(name: str) -> Readiness:
    """A library that downloads its model on first use is MISSING until one is
    cached -- the same convention readiness applies to webcam_look's weight."""
    if name != "whisper":
        return Readiness(UNKNOWN, f"no check for cached model {name!r}")
    try:
        cached = sorted(f for f in os.listdir(_whisper_cache_dir()) if f.endswith(".pt"))
    except OSError:
        cached = []
    if cached:
        return Readiness(READY, f"Whisper model cached ({', '.join(cached)})")
    return Readiness(
        MISSING, "no Whisper model downloaded yet; Whisper fetches one on first use"
    )
```

- [ ] **Step 4: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_capability_map_providers.py tests/test_capability_map_core.py -v -p no:cacheprovider
```
Expected: 33 passed (14 + 19).

- [ ] **Step 5: Commit**

```bash
git add studio/backend/core/inference/capability_map/providers.py studio/backend/tests/test_capability_map_providers.py
git commit -m "feat(capabilities): model vision and cached-model checks

Vision reads already-running backends only: the llama.cpp backend through
sys.modules (never importing routes.inference, never get_llama_cpp_backend) and
the orchestrator through peek_inference_backend, which never constructs one.
A loaded model without vision is missing; no loaded local model is unknown,
because an external model may be serving.

Whisper is missing until a .pt model is cached, matching readiness's
webcam_look convention. The cache path is pinned (importing whisper loads torch)
with a test reading whisper.load_model's own source for drift.

The creating-getter tripwires RECORD rather than raise: check_requirement
swallows exceptions into 'unknown', so a raising tripwire would be inert.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Demonstrate the controls (after committing)**

Restore each with `git checkout -- studio/backend/core/inference/capability_map/providers.py`.

1. In `_orchestrator`, replace `peek = getattr(module, "peek_inference_backend", None)` with `peek = getattr(module, "get_inference_backend", None)`. Expect `test_the_checks_never_call_a_creating_getter` to FAIL on its `calls == []` assertion.
2. In `_check_model`, change the `if not loaded:` branch to return `MISSING`. Expect `test_vision_is_unknown_when_the_backend_exists_but_nothing_is_loaded` to FAIL.
3. In `_check_cached_model`, drop the `.endswith(".pt")` filter. Expect `test_a_non_model_file_in_the_whisper_cache_does_not_count` to FAIL.

Finish with `git status --porcelain` — no modified tracked files.

---

### Task 3: Probes for the three tools readiness left unknown

**Files:**
- Modify: `studio/backend/core/inference/tool_readiness/probes.py`
- Modify: `studio/backend/tests/test_tool_readiness_probes.py`

**Interfaces:**
- Consumes: `Readiness`, `READY`/`MISSING`/`UNKNOWN`, `register_default`, `_module_present` (all existing in `probes.py`)
- Produces: probes registered for `remove_background`, `detect_shapes`, `edit_image_prompt`; helpers `_bg_removal_weight_present() -> bool`, `_torch_hub_dir() -> Optional[str]`, `_maskrcnn_weight_present() -> Optional[bool]`, `_diffusion_state() -> str` (one of `"unknown"`, `"sd_cpp"`, `"not_loaded"`, `"loaded"`); constant `_MASKRCNN_WEIGHT`

This task does not touch `capability_map/`. It shares `tool_readiness`'s registry with Tasks 1 and 5, which is why every new probe goes through `register_default`.

- [ ] **Step 1: Rewrite the test these probes invalidate**

In `studio/backend/tests/test_tool_readiness_probes.py`, replace the whole of `test_unprobed_tools_are_reported_as_unknown_rows_not_omitted` (currently lines 224–233) with:

```python
def test_unprobed_tools_are_reported_as_unknown_rows_not_omitted(monkeypatch):
    """The row must actually say 'unknown'; merely appearing is not enough.

    This used to name detect_shapes, edit_image_prompt and remove_background, the
    three tools piece 3a left unprobed. Piece 3b gave all three probes, so no real
    built-in tool is unprobed any more -- the invariant is exercised instead with a
    name that has no probe by construction, added to the catalogue the report
    walks. What is being guarded is unchanged."""
    real = tr._all_tool_names()
    monkeypatch.setattr(tr, "_all_tool_names", lambda: real + ["zz_unprobed_tool"])
    report = tr.execute("check_tool_readiness", {})
    rows = {
        line.split()[0]: line
        for line in report.splitlines()
        if not line.startswith(" ")
    }
    assert "unknown" in rows["zz_unprobed_tool"], rows["zz_unprobed_tool"]
```

- [ ] **Step 2: Write the failing tests**

Change the import block at the top of `studio/backend/tests/test_tool_readiness_probes.py` from

```python
from __future__ import annotations

import pytest
```

to

```python
from __future__ import annotations

import os
import sys
import types

import pytest
```

Then append to the end of the file:

```python
# --- piece 3b: the three tools 3a left unprobed -----------------------------


def test_the_three_previously_unprobed_tools_now_have_probes():
    for name in ("remove_background", "detect_shapes", "edit_image_prompt"):
        assert name in tr.registered_names(), name


# remove_background


def test_remove_background_missing_without_onnxruntime(monkeypatch):
    monkeypatch.setattr(probes, "_module_present", lambda name: name != "onnxruntime")
    monkeypatch.setattr(probes, "_bg_removal_weight_present", lambda: True)
    r = tr.resolve("remove_background", refresh = True)
    assert r.state == tr.MISSING
    assert r.missing == "onnxruntime"


def test_remove_background_missing_without_its_weight(monkeypatch):
    monkeypatch.setattr(probes, "_module_present", lambda name: True)
    monkeypatch.setattr(probes, "_bg_removal_weight_present", lambda: False)
    r = tr.resolve("remove_background", refresh = True)
    assert r.state == tr.MISSING
    assert r.missing == "u2net.onnx"


def test_remove_background_ready_with_package_and_weight(monkeypatch):
    """Control for the two tests above."""
    monkeypatch.setattr(probes, "_module_present", lambda name: True)
    monkeypatch.setattr(probes, "_bg_removal_weight_present", lambda: True)
    assert tr.resolve("remove_background", refresh = True).state == tr.READY


def test_the_bg_weight_check_delegates_to_bg_removal_itself(tmp_path, monkeypatch):
    """bg_removal._model_path owns the override env var and the cache root."""
    from core.inference.assist_vision import bg_removal

    weight = tmp_path / "u2net.onnx"
    monkeypatch.setattr(bg_removal, "_model_path", lambda: str(weight))
    assert probes._bg_removal_weight_present() is False
    weight.write_bytes(b"x")
    assert probes._bg_removal_weight_present() is True


# detect_shapes


def test_detect_shapes_missing_without_torch_or_torchvision(monkeypatch):
    monkeypatch.setattr(probes, "_maskrcnn_weight_present", lambda: True)
    for absent in ("torch", "torchvision"):
        monkeypatch.setattr(probes, "_module_present", lambda name, a = absent: name != a)
        r = tr.resolve("detect_shapes", refresh = True)
        assert r.state == tr.MISSING, absent
        assert r.missing == absent


def test_detect_shapes_unknown_when_the_weight_location_is_unknown(monkeypatch):
    monkeypatch.setattr(probes, "_module_present", lambda name: True)
    monkeypatch.setattr(probes, "_maskrcnn_weight_present", lambda: None)
    assert tr.resolve("detect_shapes", refresh = True).state == tr.UNKNOWN


def test_detect_shapes_missing_without_its_weight(monkeypatch):
    monkeypatch.setattr(probes, "_module_present", lambda name: True)
    monkeypatch.setattr(probes, "_maskrcnn_weight_present", lambda: False)
    r = tr.resolve("detect_shapes", refresh = True)
    assert r.state == tr.MISSING
    assert "first use" in (r.remedy or "")


def test_detect_shapes_ready_with_packages_and_weight(monkeypatch):
    """Control for the three tests above."""
    monkeypatch.setattr(probes, "_module_present", lambda name: True)
    monkeypatch.setattr(probes, "_maskrcnn_weight_present", lambda: True)
    assert tr.resolve("detect_shapes", refresh = True).state == tr.READY


def test_the_torch_hub_dir_is_read_only_from_an_already_imported_torch(tmp_path, monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.hub = types.SimpleNamespace(get_dir = lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert probes._torch_hub_dir() == str(tmp_path)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    assert probes._maskrcnn_weight_present() is False
    (checkpoints / probes._MASKRCNN_WEIGHT).write_bytes(b"x")
    assert probes._maskrcnn_weight_present() is True


def test_no_torch_in_memory_means_the_weight_location_is_unknown(monkeypatch):
    # None in sys.modules reads as "not imported" without evicting the real torch.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert probes._torch_hub_dir() is None
    assert probes._maskrcnn_weight_present() is None


def test_the_pinned_maskrcnn_filename_matches_torchvision():
    """Drift guard: reading the filename at runtime would import torchvision's
    detection stack, so it is pinned and checked here."""
    detection = pytest.importorskip("torchvision.models.detection")
    url = detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT.url
    assert os.path.basename(url) == probes._MASKRCNN_WEIGHT


# edit_image_prompt

_ROUTER = "core.inference.diffusion_engine_router"
_DIFFUSION = "core.inference.diffusion"


@pytest.fixture
def _diffusion_modules(monkeypatch):
    """Clears both modules and returns a helper to install fakes. The fakes'
    creating getters RECORD calls; resolve() swallows exceptions, so a raising
    tripwire would be inert."""
    monkeypatch.delitem(sys.modules, _ROUTER, raising = False)
    monkeypatch.delitem(sys.modules, _DIFFUSION, raising = False)
    calls = []

    def install(*, active = None, backend = "absent"):
        if active is not None:
            router = types.ModuleType(_ROUTER)
            router.ENGINE_DIFFUSERS = "diffusers"
            router.ENGINE_SD_CPP = "sd_cpp"
            router._active_engine_name = active
            router.get_active_diffusion_engine = lambda: calls.append("get_active_diffusion_engine")
            monkeypatch.setitem(sys.modules, _ROUTER, router)
        if backend != "absent":
            diffusion = types.ModuleType(_DIFFUSION)
            diffusion._diffusion_backend = backend
            diffusion.get_diffusion_backend = lambda: calls.append("get_diffusion_backend")
            monkeypatch.setitem(sys.modules, _DIFFUSION, diffusion)

    return install, calls


def _engine(loaded):
    return types.SimpleNamespace(is_loaded = loaded)


def test_edit_image_prompt_unknown_when_no_router_is_in_memory(_diffusion_modules):
    assert tr.resolve("edit_image_prompt", refresh = True).state == tr.UNKNOWN


def test_edit_image_prompt_unknown_when_the_engine_module_is_not_in_memory(_diffusion_modules):
    install, _calls = _diffusion_modules
    install(active = "diffusers")
    assert tr.resolve("edit_image_prompt", refresh = True).state == tr.UNKNOWN


def test_edit_image_prompt_missing_on_the_native_sd_cpp_engine(_diffusion_modules):
    install, _calls = _diffusion_modules
    install(active = "sd_cpp", backend = _engine(True))
    r = tr.resolve("edit_image_prompt", refresh = True)
    assert r.state == tr.MISSING
    assert "image-to-image" in r.detail


def test_edit_image_prompt_missing_when_no_image_model_is_loaded(_diffusion_modules):
    install, _calls = _diffusion_modules
    install(active = "diffusers", backend = None)
    assert tr.resolve("edit_image_prompt", refresh = True).state == tr.MISSING
    install(active = "diffusers", backend = _engine(False))
    assert tr.resolve("edit_image_prompt", refresh = True).state == tr.MISSING


def test_edit_image_prompt_ready_with_a_diffusers_model_loaded(_diffusion_modules):
    """Control for the missing cases above."""
    install, _calls = _diffusion_modules
    install(active = "diffusers", backend = _engine(True))
    assert tr.resolve("edit_image_prompt", refresh = True).state == tr.READY


def test_the_diffusion_probe_never_calls_a_creating_getter(_diffusion_modules):
    install, calls = _diffusion_modules
    install(active = "diffusers", backend = _engine(True))
    assert tr.resolve("edit_image_prompt", refresh = True).state == tr.READY
    install(active = "sd_cpp", backend = None)
    tr.resolve("edit_image_prompt", refresh = True)
    assert calls == [], f"the probe constructed an engine: {calls}"
```

- [ ] **Step 3: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_tool_readiness_probes.py -v -p no:cacheprovider
```
Expected: the new tests FAIL (no probes registered, helpers absent). The rewritten `test_unprobed_tools_are_reported_as_unknown_rows_not_omitted` already passes.

- [ ] **Step 4: Add the probes**

In `studio/backend/core/inference/tool_readiness/probes.py`:

(a) Change the import block from

```python
import importlib.util
import os
```

to

```python
import importlib.util
import os
import sys
from typing import Optional
```

(b) Directly after the `_CODE_TOOL_NAMES = (...)` tuple, add:

```python
# Pinned: reading MaskRCNN_ResNet50_FPN_Weights.DEFAULT.url at runtime imports
# torchvision's detection stack. A test asserts it still matches torchvision.
_MASKRCNN_WEIGHT = "maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
```

(c) Directly above `def _probe_always_ready`, add:

```python
def _bg_removal_weight_present() -> bool:
    """Delegates to bg_removal._model_path(), which owns the override env var and
    the shared model cache root."""
    from core.inference.assist_vision import bg_removal

    return os.path.isfile(bg_removal._model_path())


def _torch_hub_dir() -> Optional[str]:
    """torch.hub.get_dir() -- but only from a torch that is ALREADY imported. The
    running backend imports torch at startup; importing it here would cost
    seconds and break the cheap-probe contract."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    hub = getattr(torch, "hub", None)
    return hub.get_dir() if hub is not None else None


def _maskrcnn_weight_present() -> Optional[bool]:
    """None when the cache location cannot be known without importing torch."""
    hub_dir = _torch_hub_dir()
    if hub_dir is None:
        return None
    return os.path.isfile(os.path.join(hub_dir, "checkpoints", _MASKRCNN_WEIGHT))


def _diffusion_state() -> str:
    """'unknown' | 'sd_cpp' | 'not_loaded' | 'loaded', read from modules already
    in memory. Never get_active_diffusion_engine() / get_diffusion_backend():
    both construct the engine they return."""
    router = sys.modules.get("core.inference.diffusion_engine_router")
    if router is None:
        return "unknown"
    if getattr(router, "_active_engine_name", None) == getattr(router, "ENGINE_SD_CPP", "sd_cpp"):
        return "sd_cpp"
    diffusion = sys.modules.get("core.inference.diffusion")
    if diffusion is None:
        return "unknown"
    backend = getattr(diffusion, "_diffusion_backend", None)
    if backend is None or not backend.is_loaded:
        return "not_loaded"
    return "loaded"
```

(d) Directly above `def _probe_code_tool`, add:

```python
def _probe_remove_background() -> Readiness:
    if not _module_present("onnxruntime"):
        return Readiness(
            MISSING,
            "onnxruntime not importable in this build",
            missing = "onnxruntime",
            remedy = "pip install onnxruntime (or declare it in the frozen build)",
        )
    if _bg_removal_weight_present():
        return Readiness(READY, "u2net.onnx present; onnxruntime importable")
    return Readiness(
        MISSING,
        "u2net.onnx not in the model cache",
        missing = "u2net.onnx",
        remedy = "it is downloaded on first use",
    )


def _probe_detect_shapes() -> Readiness:
    absent = [name for name in ("torch", "torchvision") if not _module_present(name)]
    if absent:
        return Readiness(
            MISSING,
            f"{', '.join(absent)} not importable in this build",
            missing = ", ".join(absent),
            remedy = "pip install torch torchvision (or declare them in the frozen build)",
        )
    present = _maskrcnn_weight_present()
    if present is None:
        return Readiness(
            UNKNOWN,
            "torch and torchvision importable; the weight cache cannot be located without loading torch",
        )
    if present:
        return Readiness(READY, "Mask R-CNN weights present; torch and torchvision importable")
    return Readiness(
        MISSING,
        "Mask R-CNN weights not downloaded",
        missing = _MASKRCNN_WEIGHT,
        remedy = "torchvision downloads them (~170 MB) on first use",
    )


def _probe_edit_image_prompt() -> Readiness:
    state = _diffusion_state()
    if state == "sd_cpp":
        return Readiness(
            MISSING,
            "the native sd.cpp engine is active, and it does not support image-to-image",
            missing = "an image model on the diffusers engine",
            remedy = "load the image model in Studio on the diffusers engine",
        )
    if state == "not_loaded":
        return Readiness(
            MISSING,
            "no image model loaded in Studio",
            missing = "a loaded image model",
            remedy = "load an image model in Studio",
        )
    if state == "loaded":
        return Readiness(READY, "an image model is loaded on the diffusers engine")
    return Readiness(
        UNKNOWN,
        "image generation has not been used this session, so no engine is in memory to check",
    )
```

(e) In `install_default_probes`, directly after `register_default("face_swap", _probe_face_swap)`, add:

```python
    register_default("remove_background", _probe_remove_background)
    register_default("detect_shapes", _probe_detect_shapes)
    register_default("edit_image_prompt", _probe_edit_image_prompt)
```

- [ ] **Step 5: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_tool_readiness_probes.py tests/test_tool_readiness_registry.py tests/test_tool_readiness_seam.py tests/test_tool_readiness_enrichment.py -v -p no:cacheprovider
```
Expected: all pass. `test_tool_readiness_probes.py` grows by 18 tests.

- [ ] **Step 6: Commit**

```bash
git add studio/backend/core/inference/tool_readiness/probes.py studio/backend/tests/test_tool_readiness_probes.py
git commit -m "feat(readiness): probe remove_background, detect_shapes, edit_image_prompt

The three tools piece 3a left 'unknown' now have probes; each is also a provider
in the capability map.

- remove_background: onnxruntime importable + u2net.onnx present, via
  bg_removal._model_path() so its override env var is honoured.
- detect_shapes: torch + torchvision importable + the Mask R-CNN weight in the
  torch hub cache. The hub dir is read only from an already-imported torch
  (else unknown); the filename is pinned with a drift test against torchvision.
- edit_image_prompt: missing on the native sd.cpp engine (it rejects
  image-to-image, sd_cpp_backend.py:2153) or with no model loaded; unknown until
  image generation has put an engine in memory. Never calls
  get_active_diffusion_engine / get_diffusion_backend, which construct engines.

Rewrites test_unprobed_tools_are_reported_as_unknown_rows_not_omitted: it named
these three tools, so the invariant is now exercised with a fake unprobed name.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Demonstrate the controls (after committing)**

Restore each with `git checkout -- <file>`.

1. In `probes.py` `_probe_edit_image_prompt`, make the `sd_cpp` branch return `READY`. Expect `test_edit_image_prompt_missing_on_the_native_sd_cpp_engine` to FAIL.
2. In `_diffusion_state`, replace `backend = getattr(diffusion, "_diffusion_backend", None)` with `backend = diffusion.get_diffusion_backend()`. Expect `test_the_diffusion_probe_never_calls_a_creating_getter` to FAIL.
3. In `_probe_detect_shapes`, change `if present is None:` to return `READY`. Expect `test_detect_shapes_unknown_when_the_weight_location_is_unknown` to FAIL.
4. In `probes.py`, change `_MASKRCNN_WEIGHT` to `"maskrcnn_wrong.pth"`. Expect `test_the_pinned_maskrcnn_filename_matches_torchvision` to FAIL.
5. In `tool_readiness/__init__.py` `execute`, delete the `for name in _all_tool_names(): ...` loop. Expect the rewritten `test_unprobed_tools_are_reported_as_unknown_rows_not_omitted` to FAIL.

Finish with `git status --porcelain` — no modified tracked files.

---

### Task 4: The capability vocabulary

**Files:**
- Create: `studio/backend/core/inference/capability_map/vocabulary.py`
- Test: `studio/backend/tests/test_capability_map_vocabulary.py`

**Interfaces:**
- Consumes: `Capability`, `Provider`, `Requirement`, `find`, `normalize` (Task 1)
- Produces: `CAPABILITIES: tuple[Capability, ...]` — exactly 22, in the order below

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_capability_map_vocabulary.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The curated vocabulary must stay honest as the codebase grows.

A new tool without a capability fails here -- the intended forcing function. An
acquire hint that turns into an install command fails here too, because master
spec §4 says downloads must be identified, verified and presented to the user,
and a paste-ready command would let the model install through `terminal`.
"""

from __future__ import annotations

import re

from core.inference import capability_map as cm
from core.inference.capability_map.vocabulary import CAPABILITIES

_DISCOVERY_TOOLS = {"check_tool_readiness", "find_capability"}

_INSTALL_COMMAND = re.compile(
    r"pip3?\s+install|\bwinget\b|\bchoco\b|\bnpm\s+i(nstall)?\b|\bapt(-get)?\s+install|"
    r"\bapt-get\b|\bbrew\s+install|\bconda\s+install|\bcurl\b|dotnet\s+tool\s+install",
    re.IGNORECASE,
)

_CHECKABLE = {
    "tool": None,  # any real tool name
    "module": None,  # any module name
    "binary": None,  # any program name
    "model": {"vision"},
    "cached_model": {"whisper"},
}


def _tool_names_required():
    for capability in CAPABILITIES:
        for provider in capability.providers:
            for requirement in cm.requirements_for(provider):
                if requirement.kind == "tool":
                    yield requirement.name


def test_the_vocabulary_is_the_22_specified_capabilities_in_order():
    assert [c.name for c in CAPABILITIES] == [
        "internet_search", "run_python", "run_commands", "filesystem", "pdf_analysis",
        "ocr", "image_understanding", "object_detection", "camera", "image_processing",
        "image_editing", "background_removal", "face_swap", "image_generation",
        "video_editing", "speech_to_text", "word_documents", "document_search",
        "conversation_recall", "code_intelligence", "visual_output", "computer_control",
    ]


def test_every_tool_requirement_names_a_real_tool():
    from core.inference.tools import ALL_TOOLS

    real = {t["function"]["name"] for t in ALL_TOOLS}
    for name in _tool_names_required():
        assert name in real, f"the vocabulary requires tool {name!r}, which does not exist"


def test_every_current_tool_is_findable_by_capability():
    """The vocabulary's promise: every tool maps to some capability."""
    from core.inference.tools import ALL_TOOLS

    required = set(_tool_names_required())
    for tool in ALL_TOOLS:
        name = tool["function"]["name"]
        if name in _DISCOVERY_TOOLS:
            continue
        assert name in required, f"tool {name!r} is in no capability"


def test_no_name_or_alias_belongs_to_two_capabilities():
    seen = {}
    for capability in CAPABILITIES:
        for key in (capability.name, *capability.aliases):
            normalized = cm.normalize(key)
            assert normalized not in seen or seen[normalized] == capability.name, (
                f"{key!r} matches both {seen.get(normalized)!r} and {capability.name!r}"
            )
            seen[normalized] = capability.name


def test_no_acquire_hint_contains_an_install_command():
    for capability in CAPABILITIES:
        if capability.acquire:
            assert not _INSTALL_COMMAND.search(capability.acquire), (
                f"{capability.name}: acquire hint is a runnable command: {capability.acquire!r}"
            )


def test_a_capability_nothing_provides_says_what_would_provide_it():
    for capability in CAPABILITIES:
        if not capability.providers:
            assert capability.acquire, f"{capability.name} has no providers and no acquire hint"


def test_every_provider_has_a_reason_and_something_to_check():
    for capability in CAPABILITIES:
        for provider in capability.providers:
            assert provider.reason.strip(), f"{capability.name}/{provider.name} has no reason"
            assert cm.requirements_for(provider), f"{capability.name}/{provider.name} checks nothing"


def test_every_requirement_is_one_the_checks_can_answer():
    """A requirement no check handles is 'unknown' forever -- silently."""
    for capability in CAPABILITIES:
        for provider in capability.providers:
            for requirement in provider.requires:
                assert requirement.kind in _CHECKABLE, (capability.name, requirement)
                allowed = _CHECKABLE[requirement.kind]
                if allowed is not None:
                    assert requirement.name in allowed, (capability.name, requirement)


def test_speech_to_text_requires_whisper_ffmpeg_and_a_cached_model():
    """Whisper imports cleanly without ffmpeg, then fails on every file."""
    capability = cm.find("speech_to_text", CAPABILITIES)
    requires = {(r.kind, r.name) for p in capability.providers for r in p.requires}
    assert {("module", "whisper"), ("binary", "ffmpeg"), ("cached_model", "whisper")} <= requires


def test_the_master_spec_s37_questions_find_a_capability():
    for question, expected in (
        ("manipulate images", "image_processing"),
        ("edit videos", "video_editing"),
        ("access the filesystem", "filesystem"),
        ("execute python", "run_python"),
        ("search the internet", "internet_search"),
        ("control the computer", "computer_control"),
        ("generate images", "image_generation"),
        ("analyze pdfs", "pdf_analysis"),
        ("OCR", "ocr"),
    ):
        found = cm.find(question, CAPABILITIES)
        assert found is not None and found.name == expected, question
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_capability_map_vocabulary.py -v -p no:cacheprovider
```
Expected: collection error — `No module named 'core.inference.capability_map.vocabulary'`.

- [ ] **Step 3: Create the vocabulary**

Create `studio/backend/core/inference/capability_map/vocabulary.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The curated capability vocabulary.

Providers are in PREFERENCE ORDER; the capability's best option is the first one
that is actually ready, and its `reason` says why it ranks there. The order is a
judgement, not a measurement -- the reasons exist so the model can see the
trade-off and pick another listed provider when the task calls for it.

Acquire hints are DESCRIPTIVE. Never a runnable command: master spec §4 requires
downloads to be identified, verified and presented to the user, and acquisition
is a later piece with its own security design. A test enforces this.

MCP tools are not here: their names only exist at runtime.
"""

from __future__ import annotations

from core.inference.capability_map import Capability, Provider, Requirement


def _tool(name: str, reason: str) -> Provider:
    return Provider("tool", name, None, (Requirement("tool", name),), reason)


def _software(name: str, via: str, reason: str, *requires: Requirement) -> Provider:
    return Provider("software", name, via, tuple(requires), reason)


def _vision(reason: str) -> Provider:
    return Provider("model", "vision model", None, (Requirement("model", "vision"),), reason)


_CODE_TOOLS = ("code_definition", "code_references", "code_hover", "code_symbols", "code_diagnostics")

CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        "internet_search",
        "Search the web and read pages",
        ("web search", "search the internet", "search online", "browse the web"),
        (_tool("web_search", "searches and fetches pages in one tool, with no API key"),),
        "Needs the ddgs Python package.",
    ),
    Capability(
        "run_python",
        "Run Python code",
        ("python", "execute python", "run code"),
        (_tool("python", "runs in the Studio environment, so its installed packages are available"),),
        None,
    ),
    Capability(
        "run_commands",
        "Run terminal commands",
        ("terminal", "shell", "command line", "run a command"),
        (_tool("terminal", "runs the platform shell with the installed programs on PATH"),),
        None,
    ),
    Capability(
        "filesystem",
        "Read, write and edit files",
        ("files", "file access", "access the filesystem", "edit files", "read files", "write files"),
        (
            _tool("edit_file", "exact, reviewable edits to a single file"),
            _tool("terminal", "listing, moving and reading many files"),
            _tool("python", "bulk or structured file work"),
        ),
        None,
    ),
    Capability(
        "pdf_analysis",
        "Extract text and tables from PDFs",
        ("pdf", "pdfs", "read pdf", "analyze pdfs", "analyse pdfs", "pdf text"),
        (
            _software(
                "pymupdf4llm", "python",
                "keeps headings, tables and reading order as Markdown",
                Requirement("module", "pymupdf4llm"),
            ),
            _software(
                "PyMuPDF", "python",
                "page-level text, images and metadata",
                Requirement("module", "fitz"),
            ),
        ),
        "Needs the PyMuPDF Python package.",
    ),
    Capability(
        "ocr",
        "Read text in images",
        ("text recognition", "read text from image", "read text in images", "optical character recognition"),
        (
            _vision("understands layout; nothing to install"),
            _software(
                "Tesseract", "terminal",
                "fast on clean scans and large batches",
                Requirement("binary", "tesseract"),
            ),
            _software(
                "EasyOCR", "python",
                "handles photographed text and many languages",
                Requirement("module", "easyocr"),
            ),
        ),
        "Load a vision-capable model, or install Tesseract OCR (a free program).",
    ),
    Capability(
        "image_understanding",
        "Describe or answer questions about an image",
        ("describe image", "describe an image", "what is in this image", "image question", "vision"),
        (
            _vision("answers open questions about what an image shows"),
            _tool("detect_shapes", "labels the objects it finds, and nothing more"),
        ),
        "Load a vision-capable model.",
    ),
    Capability(
        "object_detection",
        "Find and label objects in a photo",
        ("detect objects", "find objects", "label objects", "object recognition"),
        (
            _tool("detect_shapes", "boxes and a confidence for each object"),
            _vision("describes the objects, without boxes"),
        ),
        "Needs torch, torchvision and the Mask R-CNN weights.",
    ),
    Capability(
        "camera",
        "See through the webcam",
        ("webcam", "take a photo", "see through the camera"),
        (_tool("webcam_look", "captures a frame and identifies what is in view"),),
        "Needs a webcam, ultralytics, OpenCV and the YOLO weights.",
    ),
    Capability(
        "image_processing",
        "Resize, crop, convert or adjust images precisely",
        ("manipulate images", "image manipulation", "resize image", "crop image", "convert image"),
        (
            _software(
                "Pillow", "python",
                "simple, exact resizing, cropping and conversion",
                Requirement("module", "PIL"),
            ),
            _software(
                "OpenCV", "python",
                "advanced filters and geometry",
                Requirement("module", "cv2"),
            ),
        ),
        "Needs the Pillow Python package.",
    ),
    Capability(
        "image_editing",
        "Change an image by describing the edit",
        ("edit image", "edit an image", "change an image", "photo editing"),
        (_tool("edit_image_prompt", "applies a described change with the loaded image model"),),
        "Load an image model in Studio on the diffusers engine.",
    ),
    Capability(
        "background_removal",
        "Remove an image's background",
        ("remove background", "transparent background", "cut out subject"),
        (_tool("remove_background", "returns a transparent PNG of the subject"),),
        "Needs onnxruntime and the u2net model.",
    ),
    Capability(
        "face_swap",
        "Swap a face between two images",
        ("swap faces", "face replacement"),
        (_tool("face_swap", "swaps a face from one image into another"),),
        "Needs the InsightFace licence accepted and its models downloaded.",
    ),
    Capability(
        "image_generation",
        "Create an image from a text description",
        ("generate images", "generate an image", "create image", "text to image", "draw a picture"),
        (),
        "Studio can generate images on its Images page, but no agent tool exposes text-to-image yet.",
    ),
    Capability(
        "video_editing",
        "Cut, convert or combine video",
        ("edit videos", "edit video", "cut video", "trim video", "convert video", "combine video"),
        (
            _software(
                "FFmpeg", "terminal",
                "cuts, converts and combines almost any format",
                Requirement("binary", "ffmpeg"),
            ),
        ),
        "Needs FFmpeg, a free command-line program.",
    ),
    Capability(
        "speech_to_text",
        "Transcribe audio or video speech",
        ("transcribe", "transcription", "speech recognition", "audio to text"),
        (
            _software(
                "Whisper", "python",
                "accurate local transcription in many languages",
                Requirement("module", "whisper"),
                Requirement("binary", "ffmpeg"),
                Requirement("cached_model", "whisper"),
            ),
        ),
        (
            "Needs the Whisper Python package, FFmpeg (a free command-line program), and a "
            "Whisper model, which Whisper downloads on first use (the smallest is about 75 MB)."
        ),
    ),
    Capability(
        "word_documents",
        "Read and write Word documents",
        ("word", "docx", "word document"),
        (
            _software(
                "python-docx", "python",
                "reads and writes .docx paragraphs, tables and styles",
                Requirement("module", "docx"),
            ),
        ),
        "Needs the python-docx Python package.",
    ),
    Capability(
        "document_search",
        "Search the user's uploaded documents",
        ("search documents", "knowledge base", "uploaded documents"),
        (_tool("search_knowledge_base", "searches the documents the user uploaded"),),
        "Upload documents to a knowledge base.",
    ),
    Capability(
        "conversation_recall",
        "Recall earlier parts of this conversation",
        ("recall conversation", "earlier in this conversation", "search conversation"),
        (_tool("search_conversation", "finds turns trimmed from this conversation's context"),),
        "Needs the conversation archive enabled with RAG available.",
    ),
    Capability(
        "code_intelligence",
        "Definitions, references, types and errors in code",
        ("go to definition", "find references", "type errors", "code navigation"),
        (
            Provider(
                "tool",
                "code tools",
                None,
                tuple(Requirement("tool", name) for name in _CODE_TOOLS),
                "language-server answers without running a build",
            ),
        ),
        "Needs a language server for the project's language.",
    ),
    Capability(
        "visual_output",
        "Show the user an interactive HTML canvas or chart",
        ("render html", "chart", "canvas", "show a visualization"),
        (_tool("render_html", "shows an interactive canvas or chart to the user"),),
        None,
    ),
    Capability(
        "computer_control",
        "Control the mouse, keyboard and applications",
        ("control the computer", "mouse and keyboard", "desktop control", "automate applications"),
        (),
        "No provider yet; desktop control is planned for a later piece.",
    ),
)
```

- [ ] **Step 4: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_capability_map_vocabulary.py tests/test_capability_map_core.py -v -p no:cacheprovider
```
Expected: 29 passed (10 + 19).

- [ ] **Step 5: Commit**

```bash
git add studio/backend/core/inference/capability_map/vocabulary.py studio/backend/tests/test_capability_map_vocabulary.py
git commit -m "feat(capabilities): the 22-capability vocabulary

Master spec §37/§38's named examples plus a capability for everything today's
tools do, each with providers in curated preference order and a reason for each
rank. Speech-to-text requires whisper, ffmpeg AND a cached model -- whisper
imports cleanly without ffmpeg and then fails on every file.

Integrity tests make the vocabulary's promises enforceable: every tool maps to
a capability, every tool requirement names a real tool, no alias is ambiguous,
every requirement is one a check can answer, the §37 questions resolve, and no
acquire hint is a runnable install command (§4).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Demonstrate the controls (after committing)**

Restore each with `git checkout -- studio/backend/core/inference/capability_map/vocabulary.py`.

1. Append ` Run: pip install easyocr` to the `ocr` capability's acquire hint. Expect `test_no_acquire_hint_contains_an_install_command` to FAIL.
2. Delete the whole `face_swap` `Capability(...)` entry. Expect `test_every_current_tool_is_findable_by_capability` to FAIL (and the 22-names test).
3. Add the alias `"transcribe"` to `video_editing`. Expect `test_no_name_or_alias_belongs_to_two_capabilities` to FAIL.
4. Delete `Requirement("binary", "ffmpeg"),` from the Whisper provider. Expect `test_speech_to_text_requires_whisper_ffmpeg_and_a_cached_model` to FAIL.

Finish with `git status --porcelain` — no modified tracked files.

---

### Task 5: Output, `execute()` and the schema

**Files:**
- Modify: `studio/backend/core/inference/capability_map/__init__.py`
- Create: `studio/backend/core/inference/capability_map/schemas.py`
- Test: `studio/backend/tests/test_capability_map_tool.py`

**Interfaces:**
- Consumes: everything from Tasks 1, 2 and 4
- Produces:
  - in `capability_map`: `format_capability(result: CapabilityResult) -> str`, `format_map(results: Sequence[CapabilityResult]) -> str`, `format_unmatched(query: str, capabilities: Sequence[Capability]) -> str`, `execute(name: str, arguments) -> str` (never raises)
  - in `capability_map.schemas`: `FIND_CAPABILITY_TOOL`, `CAPABILITY_TOOLS: list[dict]`, `CAPABILITY_TOOL_NAMES: frozenset[str]`

The tool is not registered with the dispatcher until Task 6.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_capability_map_tool.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What find_capability says to the model."""

from __future__ import annotations

import re

import pytest

from core.inference import capability_map as cm
from core.inference import tool_readiness as tr
from core.inference.capability_map import providers
from core.inference.capability_map.vocabulary import CAPABILITIES
from core.inference.tool_readiness import MISSING, READY, UNKNOWN, Readiness


@pytest.fixture(autouse = True)
def _clean():
    tr.reset_for_tests()
    yield
    tr.reset_for_tests()


def _by_requirement(monkeypatch, states, default):
    """states keys are 'kind:name'; anything else resolves to `default`."""

    def fake(req):
        state = states.get(f"{req.kind}:{req.name}", default)
        return Readiness(state, f"{req.kind}:{req.name} is {state}")

    monkeypatch.setattr(providers, "check_requirement", fake)


def _row_index(report, capability_name):
    for index, line in enumerate(report.splitlines()):
        parts = line.split()
        if parts and parts[0] == capability_name:
            return index
    raise AssertionError(f"{capability_name} not in report")


def test_one_capability_renders_in_the_master_spec_format(monkeypatch):
    _by_requirement(monkeypatch, {}, READY)
    out = cm.execute("find_capability", {"capability": "ocr"})
    assert "CAPABILITY: ocr — Read text in images" in out
    assert "Status:      ready" in out
    assert "Best option: vision model" in out
    assert "Reason:      understands layout; nothing to install" in out
    assert "  1. vision model" in out
    assert "  2. Tesseract via terminal" in out


def test_the_best_option_skips_an_unknown_first_provider(monkeypatch):
    _by_requirement(
        monkeypatch,
        {"model:vision": UNKNOWN, "tool:terminal": READY, "binary:tesseract": READY},
        MISSING,
    )
    out = cm.execute("find_capability", {"capability": "ocr"})
    assert "Best option: Tesseract via terminal" in out


def test_a_missing_capability_says_how_to_get_it(monkeypatch):
    _by_requirement(monkeypatch, {}, MISSING)
    out = cm.execute("find_capability", {"capability": "video editing"})
    assert "Status:      missing" in out
    assert "To get it:   Needs FFmpeg" in out


def test_an_alias_finds_its_capability(monkeypatch):
    _by_requirement(monkeypatch, {}, MISSING)
    assert "CAPABILITY: speech_to_text" in cm.execute("find_capability", {"capability": "transcribe"})


def test_a_capability_with_no_providers_says_so(monkeypatch):
    _by_requirement(monkeypatch, {}, READY)
    out = cm.execute("find_capability", {"capability": "computer_control"})
    assert "Status:      missing" in out
    assert "Providers:   none yet" in out


def test_the_whole_map_lists_ready_then_unknown_then_missing(monkeypatch):
    _by_requirement(
        monkeypatch,
        {"tool:terminal": READY, "binary:ffmpeg": READY, "model:vision": UNKNOWN},
        MISSING,
    )
    out = cm.execute("find_capability", {})
    ready = _row_index(out, "video_editing")
    unknown = _row_index(out, "image_understanding")
    missing = _row_index(out, "computer_control")
    assert ready < unknown < missing, out


def test_the_whole_map_covers_every_capability_and_notes_mcp(monkeypatch):
    _by_requirement(monkeypatch, {}, MISSING)
    out = cm.execute("find_capability", {})
    for capability in CAPABILITIES:
        _row_index(out, capability.name)
    assert "MCP tools are not in this map" in out


def test_an_unmatched_query_lists_the_capabilities_instead_of_failing(monkeypatch):
    _by_requirement(monkeypatch, {}, MISSING)
    out = cm.execute("find_capability", {"capability": "teleportation"})
    assert "No capability matches 'teleportation'" in out
    for capability in CAPABILITIES:
        assert capability.name in out


def test_execute_never_raises_on_bad_arguments(monkeypatch):
    _by_requirement(monkeypatch, {}, MISSING)

    class Unprintable:
        def __str__(self):
            raise RuntimeError("cannot render")

    for arguments in (None, [], "ocr", {"capability": ["ocr"]}, {"capability": 5},
                      {"capability": {"x": 1}}, {"capability": Unprintable()}):
        out = cm.execute("find_capability", arguments)
        assert isinstance(out, str) and out, arguments


def test_output_never_carries_an_install_command_even_when_readiness_does():
    """§4 boundary, end to end: a real readiness probe whose remedy IS an install
    command must not leak into anything the map renders."""
    tr.register("web_search", lambda: Readiness(MISSING, "ddgs gone", remedy = "pip install ddgs"))
    single = cm.execute("find_capability", {"capability": "internet_search"})
    whole = cm.execute("find_capability", {})
    for out in (single, whole):
        assert not re.search(r"pip3?\s+install", out), out


def test_the_schema_matches_what_execute_accepts():
    from core.inference.capability_map.schemas import (
        CAPABILITY_TOOL_NAMES,
        CAPABILITY_TOOLS,
        FIND_CAPABILITY_TOOL,
    )

    function = FIND_CAPABILITY_TOOL["function"]
    assert function["name"] == "find_capability"
    assert set(function["parameters"]["properties"]) == {"capability"}
    assert function["parameters"]["required"] == []
    assert CAPABILITY_TOOLS == [FIND_CAPABILITY_TOOL]
    assert CAPABILITY_TOOL_NAMES == frozenset({"find_capability"})
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_capability_map_tool.py -v -p no:cacheprovider
```
Expected: FAIL — `module 'core.inference.capability_map' has no attribute 'execute'`, and the schema import fails.

- [ ] **Step 3: Add the output and `execute()`**

Append to `studio/backend/core/inference/capability_map/__init__.py`:

```python
_MCP_NOTE = "MCP tools are not in this map; their descriptions are in your tool list."

_ORDER = {READY: 0, UNKNOWN: 1, MISSING: 2}


def _label(result: ProviderResult) -> str:
    """Tool providers are phrased 'via tool X' so the model can match them against
    the tools actually offered in THIS request, which this code cannot see."""
    provider = result.provider
    if provider.kind == "software":
        return f"{provider.name} via {provider.via}"
    if provider.kind == "tool":
        tools = [r.name for r in provider.requires if r.kind == "tool"]
        if len(tools) == 1:
            return f"via tool {tools[0]}"
        return f"{provider.name} (via tools {', '.join(tools)})"
    return provider.name


def format_capability(result: CapabilityResult) -> str:
    capability = result.capability
    lines = [
        f"CAPABILITY: {capability.name} — {capability.title}",
        f"Status:      {result.state}",
    ]
    if result.best is not None:
        lines.append(f"Best option: {_label(result.best)} ({result.best.detail})")
        lines.append(f"Reason:      {result.best.provider.reason}")
    elif result.state == MISSING and capability.acquire:
        lines.append(f"To get it:   {capability.acquire}")
    if result.providers:
        lines.append("Providers:")
        for index, provider in enumerate(result.providers, 1):
            lines.append(f"  {index}. {_label(provider):<34} {provider.state:<8} {provider.detail}")
    else:
        lines.append("Providers:   none yet")
    return "\n".join(lines)


def format_map(results: Sequence[CapabilityResult]) -> str:
    ranked = sorted(enumerate(results), key = lambda pair: (_ORDER.get(pair[1].state, 1), pair[0]))
    lines = ["Capability map -- call find_capability with a capability for its providers and reasons:"]
    for _index, result in ranked:
        capability = result.capability
        line = f"  {capability.name:<20} {result.state:<8} {capability.title}"
        if result.best is not None:
            line += f" -- best: {_label(result.best)}"
        lines.append(line)
        if result.state == MISSING and capability.acquire:
            lines.append(f"  {'':<29}-> {capability.acquire}")
    lines.append(_MCP_NOTE)
    return "\n".join(lines)


def format_unmatched(query: str, capabilities: Sequence[Capability]) -> str:
    lines = [f"No capability matches {query!r}. Known capabilities:"]
    lines.extend(f"  {c.name:<20} {c.title}" for c in capabilities)
    return "\n".join(lines)


def execute(name: str, arguments) -> str:
    """Handler for the find_capability tool. Never raises."""
    try:
        from core.inference.capability_map.vocabulary import CAPABILITIES

        args = arguments if isinstance(arguments, dict) else {}
        requested = args.get("capability")
        if requested is None or (isinstance(requested, str) and not requested.strip()):
            return format_map([resolve_capability(c) for c in CAPABILITIES])
        query = requested if isinstance(requested, str) else str(requested)
        capability = find(query, CAPABILITIES)
        if capability is None:
            return format_unmatched(query, CAPABILITIES)
        return format_capability(resolve_capability(capability))
    except BaseException as exc:  # noqa: BLE001 - a capability query must never break a turn
        return f"Capability map unavailable: {exc}"
```

- [ ] **Step 4: Create the schema**

Create `studio/backend/core/inference/capability_map/schemas.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI-style schema for the capability map tool."""

FIND_CAPABILITY_TOOL = {
    "type": "function",
    "function": {
        "name": "find_capability",
        "description": (
            "Find which tools, installed software or model abilities can do something "
            "(read text in images, edit video, analyse PDFs), which is best, and whether it "
            "works right now. Omit 'capability' to see the whole map, including what is "
            "missing. Read-only and cheap: it checks what is installed and loaded, not "
            "network reachability, and it does not cover MCP tools."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "capability": {
                    "type": "string",
                    "description": (
                        "A capability name or alias, e.g. 'ocr', 'video editing', "
                        "'transcribe'. An unrecognised one returns the list of capabilities. "
                        "Omit for the whole map."
                    ),
                },
            },
            "required": [],
        },
    },
}

CAPABILITY_TOOLS = [FIND_CAPABILITY_TOOL]
CAPABILITY_TOOL_NAMES = frozenset({"find_capability"})
```

- [ ] **Step 5: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_capability_map_tool.py tests/test_capability_map_core.py tests/test_capability_map_providers.py tests/test_capability_map_vocabulary.py -v -p no:cacheprovider
```
Expected: 54 passed (11 + 19 + 14 + 10).

- [ ] **Step 6: Commit**

```bash
git add studio/backend/core/inference/capability_map/__init__.py studio/backend/core/inference/capability_map/schemas.py studio/backend/tests/test_capability_map_tool.py
git commit -m "feat(capabilities): find_capability output and schema

One capability renders in master spec §38's format -- status, best option with
its reason, then every provider with its state. With no argument, the whole map
lists ready capabilities first, then unknown, then missing with a descriptive
acquire hint, and notes that MCP tools are not covered. An unrecognised query
returns the capability list rather than an error.

execute() never raises, including on non-dict arguments and values whose str()
raises. An end-to-end test proves a readiness remedy that IS an install command
never reaches anything the map renders.

Not yet registered with the dispatcher.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Demonstrate the controls (after committing)**

Restore each with `git checkout -- <file>`.

1. In `capability_map/__init__.py` `format_map`, remove `key = ...` from `sorted` (keep vocabulary order). Expect `test_the_whole_map_lists_ready_then_unknown_then_missing` to FAIL.
2. In `capability_map/providers.py` `_check_tool`, change the detail to `f"tool {name}: {readiness.detail} ({readiness.remedy})"`. Expect `test_output_never_carries_an_install_command_even_when_readiness_does` to FAIL.
3. In `execute`, replace `return f"Capability map unavailable: {exc}"` with `raise`. Expect `test_execute_never_raises_on_bad_arguments` to FAIL.

Finish with `git status --porcelain` — no modified tracked files.

---

### Task 6: Registration and the seam

**Files:**
- Modify: `studio/backend/core/inference/tools.py` (lines 4541, 9842, 9854, 10087)
- Modify: `studio/backend/routes/inference.py` (lines 3674, 3676, 20049)
- Modify: `studio/backend/core/inference/tool_readiness/probes.py` (`ALWAYS_READY_TOOLS`)
- Modify: `studio/backend/tests/test_tool_readiness_probes.py` (always-ready assertion)
- Modify: `studio/backend/tests/test_tool_audit_seam.py` (line 102)
- Modify: `studio/backend/tests/test_tool_readiness_seam.py` (line 299)
- Test: `studio/backend/tests/test_capability_map_seam.py`

**Interfaces:**
- Consumes: `CAPABILITY_TOOLS`, `CAPABILITY_TOOL_NAMES` (Task 5), `capability_map.execute` (Task 5), `tests.test_tool_readiness_seam._anthropic_gate_genexp`
- Produces: `find_capability` reachable by the model, dispatched, safe on both channels, and `ready` in the readiness report

A new built-in tool needs FIVE registration points. The `_addable` re-add in `routes/inference.py` has been missed three times; a test that only checks `ALL_TOOLS` passed the entire time one tool was unreachable.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_capability_map_seam.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""find_capability's five registration points, each asserted by BEHAVIOUR.

Reachability drives the REAL _select_request_tools with a Studio-shaped payload.
Being in ALL_TOOLS proves nothing: check_tool_readiness was in ALL_TOOLS and
unreachable in every Studio chat, and a test replicating the route's filter was
inert under mutation.
"""

from __future__ import annotations

import asyncio
import types
from pathlib import Path

import pytest

from core.inference import tool_audit
from core.inference import tool_readiness as tr
from core.inference import tools
from storage import tool_audit_db
from tests.test_tool_readiness_seam import _anthropic_gate_genexp

_ROUTE_FILE = Path(tools.__file__).parents[2] / "routes" / "inference.py"
_STUDIO_ALLOWLIST = ["web_search", "python", "terminal", "edit_file"]


@pytest.fixture(autouse = True)
def _isolated(tmp_path, monkeypatch):
    """execute_tool is the audit shadow; without this, calling it writes audit rows
    into the user's real studio.db -- silently, because the recorder never raises."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    tool_audit_db.reset_for_tests()
    tool_audit.reset_degraded_for_tests()
    tr.reset_for_tests()
    yield
    tr.reset_for_tests()


def _payload(**over):
    base = dict(
        enabled_tools = list(_STUDIO_ALLOWLIST),
        rag_scope = None,
        thread_id = None,
        bypass_permissions = False,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _select(payload, **kwargs):
    import routes.inference as routes_mod

    kwargs.setdefault("tools_on", True)
    kwargs.setdefault("mcp_allowed", False)
    return [t["function"]["name"] for t in asyncio.run(routes_mod._select_request_tools(payload, **kwargs))]


# --- 1. reachability (the _addable re-add) ----------------------------------


def test_find_capability_reaches_the_model_through_the_real_route():
    assert "find_capability" in _select(_payload())


def test_the_narrowest_studio_allowlist_still_carries_it():
    assert "find_capability" in _select(_payload(enabled_tools = ["web_search"]))


def test_an_empty_selection_stays_empty_through_the_real_route():
    """The `tools_on and tools` gate is load-bearing; this tool must not break it."""
    assert _select(_payload(enabled_tools = [])) == []


# --- 2 and 3. ALL_TOOLS + dispatch ------------------------------------------


def test_execute_tool_dispatches_to_the_capability_map():
    out = tools.execute_tool("find_capability", {})
    assert out.startswith("Capability map"), out[:200]


def test_execute_tool_answers_one_capability():
    out = tools.execute_tool("find_capability", {"capability": "ocr"})
    assert "CAPABILITY: ocr" in out


# --- 4. _ALWAYS_SAFE_TOOLS ---------------------------------------------------


def test_find_capability_is_not_treated_as_high_risk():
    assert tools.is_high_risk_tool_call("find_capability", {}) is False


def test_an_unknown_tool_is_still_high_risk():
    """Control: the assertion above is not passing because everything is safe."""
    assert tools.is_high_risk_tool_call("zz_not_a_real_tool", {}) is True


# --- 5. the Anthropic Messages channel --------------------------------------


@pytest.fixture
def _inference_routes():
    return pytest.importorskip("routes.inference", reason = "inference stack not installed")


def test_find_capability_is_unprompted_on_the_anthropic_channel(_inference_routes):
    inf = _inference_routes
    run = types.FunctionType(_anthropic_gate_genexp(inf), vars(inf))
    assert any(run(iter([{"function": {"name": "find_capability"}}]))) is False


def test_terminal_still_trips_the_anthropic_gate(_inference_routes):
    """Control: the genexp is not answering False for everything."""
    inf = _inference_routes
    run = types.FunctionType(_anthropic_gate_genexp(inf), vars(inf))
    assert any(run(iter([{"function": {"name": "terminal"}}]))) is True


def test_the_upstream_anthropic_literal_was_not_edited(_inference_routes):
    source = _ROUTE_FILE.read_text(encoding = "utf-8")
    literal = source.split("_ANTHROPIC_UNPROMPTED_SAFE_TOOLS = frozenset(", 1)[1].split(")", 1)[0]
    assert "find_capability" not in literal, "upstream's literal was edited"


# --- readiness integration ---------------------------------------------------


def test_the_readiness_report_calls_find_capability_ready_not_unknown():
    """Without this, adding the tool makes check_tool_readiness's full report say
    'unknown -- nobody checked' about a discovery tool that plainly works."""
    out = tr.execute("check_tool_readiness", {"tool": "find_capability"})
    assert out.split()[1] == tr.READY, out
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_capability_map_seam.py -v -p no:cacheprovider
```
Expected: the reachability, dispatch, high-risk, Anthropic and readiness tests FAIL. `test_an_empty_selection_stays_empty_through_the_real_route`, `test_an_unknown_tool_is_still_high_risk`, `test_terminal_still_trips_the_anthropic_gate` and `test_the_upstream_anthropic_literal_was_not_edited` already pass.

- [ ] **Step 3: Register in `tools.py`**

Four edits. Only the first changes an existing line, and that line is fork-inserted. **Do not reformat anything else.**

(a) Line 4541 — replace

```python
_ALWAYS_SAFE_TOOLS = _ALWAYS_SAFE_TOOLS | frozenset({"check_tool_readiness"})
```

with

```python
_ALWAYS_SAFE_TOOLS = _ALWAYS_SAFE_TOOLS | frozenset({"check_tool_readiness", "find_capability"})
```

(b) Directly after line 9842 (`from core.inference.tool_readiness.schemas import READINESS_TOOLS, READINESS_TOOL_NAMES`), add:

```python
from core.inference.capability_map.schemas import CAPABILITY_TOOLS, CAPABILITY_TOOL_NAMES
```

(c) In `ALL_TOOLS`, directly after `    *READINESS_TOOLS,`, add:

```python
    *CAPABILITY_TOOLS,
```

(d) In `execute_tool`, directly after the readiness dispatch block

```python
    if name in READINESS_TOOL_NAMES:
        from core.inference import tool_readiness
        return tool_readiness.execute(name, arguments)
```

add:

```python
    if name in CAPABILITY_TOOL_NAMES:
        from core.inference import capability_map
        return capability_map.execute(name, arguments)
```

- [ ] **Step 4: Register in `routes/inference.py`**

All three lines below are fork-inserted.

(a) In the import at line 3671–3675, directly after `            READINESS_TOOL_NAMES,`, add:

```python
            CAPABILITY_TOOL_NAMES,
```

(b) Line 3676 — replace

```python
        _addable = ASSIST_VISION_TOOL_NAMES | ASSIST_CODE_TOOL_NAMES | READINESS_TOOL_NAMES
```

with (**one line** — `tests/test_tool_readiness_seam.py::_route_addable()` only parses a single-line `_addable = A | B | C`):

```python
        _addable = ASSIST_VISION_TOOL_NAMES | ASSIST_CODE_TOOL_NAMES | READINESS_TOOL_NAMES | CAPABILITY_TOOL_NAMES
```

(c) Line 20049 — replace

```python
    {"check_tool_readiness"}
```

with

```python
    {"check_tool_readiness", "find_capability"}
```

- [ ] **Step 5: Mark the tool always-ready in readiness**

In `studio/backend/core/inference/tool_readiness/probes.py`, in `ALWAYS_READY_TOOLS`, directly after `    "check_tool_readiness",`, add:

```python
    # Same reason: it is a discovery tool reading the same probes. Without this,
    # the full readiness report would call it "unknown -- nobody checked".
    "find_capability",
```

In `studio/backend/tests/test_tool_readiness_probes.py`, change the tuple in `test_trivially_ready_tools_are_ready` from

```python
    for name in ("terminal", "python", "edit_file", "render_html", "check_tool_readiness"):
```

to

```python
    for name in ("terminal", "python", "edit_file", "render_html", "check_tool_readiness", "find_capability"):
```

- [ ] **Step 6: Raise both seam ceilings**

`tools.py` is now 93 insertions, past both hard-coded ceilings of 90.

In `studio/backend/tests/test_tool_audit_seam.py`, replace

```python
    assert insertions <= 90, f"seam grew to {insertions} insertions; budget is ~88"
```

with

```python
    assert insertions <= 100, f"seam grew to {insertions} insertions; budget is ~93"
```

In `studio/backend/tests/test_tool_readiness_seam.py`, replace

```python
    assert insertions <= 90, f"seam grew to {insertions}; budget is ~88"
```

with

```python
    assert insertions <= 100, f"seam grew to {insertions}; budget is ~93"
```

Leave both `deletions == 0` assertions exactly as they are.

- [ ] **Step 7: Verify the seams**

From `C:\Users\Admin\unsloth`:
```
git diff --numstat $(git merge-base origin/main HEAD) -- studio/backend/core/inference/tools.py
git diff --numstat $(git merge-base origin/main HEAD) -- studio/backend/routes/inference.py
git diff --numstat $(git merge-base origin/main HEAD) -- studio/backend/main.py
```
Expected: `93  0`, `59  0`, `4  0`. **The second column must be 0 on every line.**

- [ ] **Step 8: Run the tests and watch them pass**

From `C:\Users\Admin\unsloth\studio\backend`:
```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_capability_map_seam.py tests/test_capability_map_core.py tests/test_capability_map_providers.py tests/test_capability_map_vocabulary.py tests/test_capability_map_tool.py tests/test_tool_readiness_registry.py tests/test_tool_readiness_probes.py tests/test_tool_readiness_seam.py tests/test_tool_readiness_enrichment.py tests/test_tool_audit_seam.py tests/test_tool_audit_recorder.py tests/test_tool_audit_db.py tests/test_assist_vision_request_tools.py tests/test_assist_code_registration.py tests/test_run_tools_locally_discriminator.py -p no:cacheprovider -q
```
Expected: 229 passed (the pre-existing 146, plus 18 from Task 3, plus 54 from Tasks 1/2/4/5, plus 11 here). If anything in the three real-route suites fails, stop and report it — adding a schema to the re-add must not change their behaviour.

- [ ] **Step 9: Commit**

```bash
git add studio/backend/core/inference/tools.py studio/backend/routes/inference.py studio/backend/core/inference/tool_readiness/probes.py studio/backend/tests/test_tool_readiness_probes.py studio/backend/tests/test_tool_audit_seam.py studio/backend/tests/test_tool_readiness_seam.py studio/backend/tests/test_capability_map_seam.py
git commit -m "feat(capabilities): register find_capability at all five points

ALL_TOOLS, execute_tool dispatch, the _addable re-add in routes/inference.py,
_ALWAYS_SAFE_TOOLS, and _ANTHROPIC_UNPROMPTED_SAFE_TOOLS. Every existing line
touched is fork-inserted, so the safe-set rebinds and the re-add simply gain a
name: tools.py 88 -> 93 insertions, routes/inference.py 58 -> 59, both still
zero deletions. The _addable union stays on one line because the readiness
seam test parses it.

Reachability is asserted through the REAL _select_request_tools with a
Studio-shaped payload -- the _addable re-add has been missed three times, and a
test checking only ALL_TOOLS passed while a tool was unreachable.

find_capability joins readiness's ALWAYS_READY_TOOLS so the full readiness
report does not call a working discovery tool 'unknown'. Both hard-coded tools.py
seam ceilings rise from 90 to 100.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 10: Demonstrate the controls (after committing)**

Restore each with `git checkout -- <file>`, and run `tests/test_capability_map_seam.py` for each.

1. In `routes/inference.py`, remove ` | CAPABILITY_TOOL_NAMES` from the `_addable` line. Expect `test_find_capability_reaches_the_model_through_the_real_route` and `test_the_narrowest_studio_allowlist_still_carries_it` to FAIL.
2. In `tools.py` line 4541, remove `, "find_capability"`. Expect `test_find_capability_is_not_treated_as_high_risk` to FAIL.
3. In `routes/inference.py` line 20049, remove `, "find_capability"`. Expect `test_find_capability_is_unprompted_on_the_anthropic_channel` to FAIL.
4. In `probes.py`, remove `"find_capability",` from `ALWAYS_READY_TOOLS`. Expect `test_the_readiness_report_calls_find_capability_ready_not_unknown` to FAIL.

Finish with `git status --porcelain` — no modified tracked files — and re-run Step 7's numstats.

---

## Self-Review

**Spec coverage.** Package layout and data model → Task 1. Status rules (provider and capability, `unknown` never promotes, zero providers `missing`) → Task 1. Matching → Task 1. Requirement kinds `tool`/`module`/`binary` → Task 1; `model` and `cached_model` → Task 2. The three new tool probes and their state tables → Task 3. The 22-capability vocabulary, descriptive acquire hints and every integrity test → Task 4. Output formats, unmatched query, never-raises → Task 5. Registration table, seam budgets, reachability through the real route, both safe sets, ceiling bumps → Task 6. Every "pinned fact with a drift test" in the spec's Testing section: Mask R-CNN filename (Task 3), Whisper cache (Task 2), sd.cpp `missing` (Task 3).

**Deliberate deviations from the spec, stated rather than glossed.**
1. **Matching also treats spaces, underscores and hyphens as equal.** The spec says case-insensitive after trimming. Without this, "video editing" would not find `video_editing` unless every name were duplicated as an alias. Still exact after normalizing, never fuzzy.
2. **A provider with no requirements is `unknown`.** The spec's rule "ready when every requirement is ready" is vacuously true for none; that would be the optimistic lie.
3. **Tool requirements drop readiness remedies.** The spec keeps acquire hints descriptive; this extends the same §4 boundary to everything the map renders, because 3a's remedies include `pip install` commands.
4. **The orchestrator is read through `peek_inference_backend()`**, not its `_inference_backend` global as the spec says. It is the orchestrator's own documented never-constructs accessor, so this is delegation rather than reaching into a private.
5. **`find_capability` joins readiness's `ALWAYS_READY_TOOLS`.** Not in the spec; without it 3a's full report would call the new tool `unknown`.
6. **One piece-3a test is rewritten**, because it asserted the exact behaviour Task 3 changes. Its invariant is kept.

**Cross-task shared state, traced.** Tasks 1, 3, 5 and 6 all touch `tool_readiness`'s registry. New probes use `register_default`, so tests that `register` explicitly still win; `_check_tool` calls `install_default_probes()` so the registry is never read empty; every test file that resolves anything resets the registry around each test.

**Placeholders.** None — every code step carries the code.

**Type consistency.** `Requirement(kind, name)`, `Provider(kind, name, via, requires, reason)`, `Capability(name, title, aliases, providers, acquire)`, `ProviderResult(provider, state, detail)`, `CapabilityResult(capability, state, best, providers)` are defined in Task 1 and used unchanged in Tasks 2, 4, 5 and 6. `check_requirement(req) -> Readiness` keeps one signature. `CAPABILITY_TOOLS` / `CAPABILITY_TOOL_NAMES` are defined in Task 5 and consumed in Task 6.

**Ordering.** Task 2 depends on 1; Task 4 on 1; Task 5 on 1, 2 and 4; Task 6 on 3 and 5. Task 3 is independent of 1, 2, 4 and 5 but must precede Task 6. Execute in order 1 → 6.
