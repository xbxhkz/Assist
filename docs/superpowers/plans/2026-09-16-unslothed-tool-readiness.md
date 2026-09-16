# Unslothed Tool Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the model ask whether a tool will actually work before calling it, and tell it what was missing when a call fails for a missing dependency.

**Architecture:** A fork-owned registry maps tool name → a cheap probe that delegates to helpers already in the codebase. A new `check_tool_readiness` tool exposes it to the model; the audit log's existing `around()` wrapper appends a one-line explanation when a call fails and its probe reports something missing.

**Tech Stack:** Python 3.12, pytest. No new dependencies.

**Spec:** `C:\Users\Admin\odysseus\docs\superpowers\specs\2026-09-16-unslothed-tool-readiness-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Repo:** `C:\Users\Admin\unsloth`. Create and work on branch `tool-readiness` off `unslothed-release`. `origin` is UPSTREAM `unslothai/unsloth` and **must never be pushed to**; the fork remote is `fork`.
- **`studio/backend/core/inference/tools.py` must stay PURELY ADDITIVE.** It is at **77 insertions / 0 deletions** against upstream; this plan adds ~7, so ~84 / **0**. Zero deletions, zero re-indentation. Verify with:
  `git diff --numstat $(git merge-base origin/main HEAD) -- studio/backend/core/inference/tools.py`
- **Do-not-edit files:** `core/inference/llama_cpp.py`, `routes/inference.py`, `pyproject.toml`.
- `studio/backend/main.py` is additive-only and stands at 4 / 0. **This plan does not touch it** — there is no HTTP surface in this piece.
- **NEVER run the full backend test suite.** Upstream fixtures fabricate GGUF files up to 40 GB and filled the disk once. Run only the test files each task names.
- **Test runner**, from `C:\Users\Admin\unsloth\studio\backend`:
  `C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest <file> -v -p no:cacheprovider`
- **Every guard gets a negative control demonstrated to fail.** Nine inert controls have appeared in this project — one written by the plan's own author in the previous piece. Where a step says "watch it fail", run it and read the failure.
- **Guards use a bare `except BaseException`**, not `except Exception`: this project established that `OverflowError` and unhashable types slip past the narrower form.
- **Style:** keyword arguments take spaces around `=` (`session_id = session_id`).
- **Commit trailer:** end every commit message with
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## Scope of v1 probes

Probes are written only where the check is cheap AND an existing helper backs it. Everything else
reports `unknown`, which is the honest answer and what the three-state design exists for.

| Tool(s) | Probe | Backed by |
|---|---|---|
| the 5 code tools | language server on PATH | `assist_code.servers._SERVERS` + `_which` |
| `webcam_look` | yolov8n weight on disk | `assist_vision.yolo.weights_path()` |
| `face_swap` | InsightFace licence accepted | `assist_vision.face_swap.licence_accepted()` |
| `web_search` | the `ddgs` package importable | `importlib.util.find_spec` |
| `terminal`, `python`, `edit_file`, `render_html`, `search_conversation` | always ready — no external dependency | — |
| everything else | `unknown` | — |

`web_search` is **not** in the always-ready set. It lazily imports `ddgs` at `tools.py:11845`, and a
frozen build missing a lazily-imported module is the single most common defect class in this project.
`find_spec("ddgs")` costs ~2 ms and imports nothing. Network reachability is deliberately not checked
— the spec forbids network calls — so `ready` here means "the package is present", which `detail`
says out loud.

MCP tools are covered by the registry's `unknown` default in v1 rather than by a registered probe:
`mcp__*` names are created at runtime, so there is no static name to register against. The honest
`unknown` is what the model sees, and 3b — which enumerates capabilities — is the piece that would
add dynamic resolution.

**`search_knowledge_base` reports `unknown` in v1, deliberately.** The spec lists it as needing an
index, but `storage/` exposes no cheap knowledge-base count helper, and inventing a query here would
be guessing at a schema this piece does not own. `unknown` is accurate; a probe can be added when
3b needs it.

## File Structure

**Created:**

| Path | Responsibility |
|---|---|
| `studio/backend/core/inference/tool_readiness/__init__.py` | `Readiness`, the registry, the TTL cache, `resolve()` / `resolve_all()` |
| `studio/backend/core/inference/tool_readiness/probes.py` | the concrete probes; every one delegates to an existing helper |
| `studio/backend/core/inference/tool_readiness/schemas.py` | the `check_tool_readiness` OpenAI schema + `READINESS_TOOLS` / `READINESS_TOOL_NAMES` |
| `studio/backend/tests/test_tool_readiness_registry.py` | registry, states, caching |
| `studio/backend/tests/test_tool_readiness_probes.py` | each probe, with its control |
| `studio/backend/tests/test_tool_readiness_seam.py` | tool registration, dispatch, the `_ALWAYS_SAFE_TOOLS` rebind, additive seam |
| `studio/backend/tests/test_tool_readiness_enrichment.py` | failure enrichment and its controls |

**Modified:**

| Path | Change |
|---|---|
| `studio/backend/core/inference/tools.py` | +7 lines: `*READINESS_TOOLS,` in `ALL_TOOLS`, a dispatch block, the `_ALWAYS_SAFE_TOOLS` rebind |
| `studio/backend/core/inference/tool_audit/__init__.py` | enrichment before `return result` |

---

### Task 1: The registry

**Files:**
- Create: `studio/backend/core/inference/tool_readiness/__init__.py`
- Test: `studio/backend/tests/test_tool_readiness_registry.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `Readiness` — frozen dataclass with `state: str`, `detail: str`, `missing: str | None`, `remedy: str | None`
  - `register(tool_name: str, probe: Callable[[], Readiness]) -> None`
  - `register_default(tool_name: str, probe: Callable[[], Readiness]) -> None` — registers only if absent
  - `resolve(tool_name: str, *, refresh: bool = False) -> Readiness`
  - `resolve_all(*, refresh: bool = False) -> dict[str, Readiness]`
  - `reset_for_tests() -> None`
  - `CACHE_TTL_SECONDS` = `60.0`

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_tool_readiness_registry.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The readiness registry.

Three states, not a boolean. A tool nobody probed reports "unknown" -- treating
it as "ready" would let the model read "nobody checked" as "verified working",
which is the one outcome that makes this feature worse than not having it.
"""

from __future__ import annotations

import pytest

from core.inference import tool_readiness as tr


@pytest.fixture(autouse = True)
def _clean():
    tr.reset_for_tests()
    yield
    tr.reset_for_tests()


def test_unregistered_tool_is_unknown_not_ready():
    r = tr.resolve("no_such_tool")
    assert r.state == "unknown"
    assert r.missing is None


def test_a_registered_tool_is_NOT_unknown():
    """Control for the test above: if everything returned unknown it would pass."""
    tr.register("t", lambda: tr.Readiness("ready", "fine", None, None))
    assert tr.resolve("t").state == "ready"


def test_missing_carries_what_is_missing_and_how_to_get_it():
    tr.register(
        "t",
        lambda: tr.Readiness("missing", "weights absent", "yolov8n.pt", "it downloads on first use"),
    )
    r = tr.resolve("t")
    assert r.state == "missing"
    assert r.missing == "yolov8n.pt"
    assert r.remedy == "it downloads on first use"


def test_a_raising_probe_reports_unknown_and_does_not_propagate():
    def boom():
        raise OSError("disk gone")

    tr.register("t", boom)
    r = tr.resolve("t")
    assert r.state == "unknown"
    assert "disk gone" in r.detail


def test_results_are_cached_within_the_ttl():
    calls = []

    def probe():
        calls.append(1)
        return tr.Readiness("ready", "ok", None, None)

    tr.register("t", probe)
    tr.resolve("t")
    tr.resolve("t")
    assert len(calls) == 1, "second resolve inside the TTL must not re-probe"


def test_refresh_bypasses_the_cache():
    calls = []

    def probe():
        calls.append(1)
        return tr.Readiness("ready", "ok", None, None)

    tr.register("t", probe)
    tr.resolve("t")
    tr.resolve("t", refresh = True)
    assert len(calls) == 2


def test_register_default_does_NOT_overwrite_an_existing_probe():
    """Load-bearing. install_default_probes() runs on every readiness query and
    every enriched failure, so if defaults clobbered explicit registrations, any
    caller-registered probe -- including a test's -- would be silently replaced
    the moment the feature was exercised."""
    tr.register("t", lambda: tr.Readiness("missing", "explicit", "x", None))
    tr.register_default("t", lambda: tr.Readiness("ready", "default", None, None))
    assert tr.resolve("t").detail == "explicit"


def test_register_default_DOES_register_an_absent_probe():
    """Control: without this, a register_default that did nothing at all would
    pass the test above."""
    tr.register_default("t", lambda: tr.Readiness("ready", "default", None, None))
    assert tr.resolve("t").detail == "default"


def test_resolve_all_returns_every_registered_tool():
    tr.register("a", lambda: tr.Readiness("ready", "", None, None))
    tr.register("b", lambda: tr.Readiness("missing", "", "x", None))
    out = tr.resolve_all()
    assert set(out) == {"a", "b"}
    assert out["b"].state == "missing"
```

- [ ] **Step 2: Run it and watch it fail**

From `C:\Users\Admin\unsloth\studio\backend`:
```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_tool_readiness_registry.py -v -p no:cacheprovider
```
Expected: `ModuleNotFoundError: No module named 'core.inference.tool_readiness'`.

- [ ] **Step 3: Implement the registry**

Create `studio/backend/core/inference/tool_readiness/__init__.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether a tool will actually work, asked cheaply.

The model already receives every tool's schema each turn, so it does not need to
be told which tools EXIST. What it cannot find out is whether one will work:
webcam_look needs a downloaded weight, the code tools need a language server on
PATH, an MCP tool needs its server connected. Today it discovers that by calling
the tool and failing.

THREE STATES, NOT A BOOLEAN. A tool with no probe -- every MCP tool, anything
added later -- reports "unknown", never "ready". Defaulting unknown to ready is
the optimistic lie that makes this feature worse than not having it: the model
would read "nobody checked" as "verified working".

Probes must be CHEAP: a file stat, a PATH lookup, an already-held connection
flag. No process spawning, no network calls. The model may ask on any turn.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

CACHE_TTL_SECONDS = 60.0

READY = "ready"
MISSING = "missing"
UNKNOWN = "unknown"


@dataclass(frozen = True)
class Readiness:
    state: str
    detail: str
    missing: Optional[str] = None
    remedy: Optional[str] = None


Probe = Callable[[], Readiness]

_probes: dict[str, Probe] = {}
_cache: dict[str, tuple[float, Readiness]] = {}
_lock = threading.Lock()


def reset_for_tests() -> None:
    with _lock:
        _probes.clear()
        _cache.clear()


def register(tool_name: str, probe: Probe) -> None:
    """Register or REPLACE a probe. An explicit registration always wins."""
    with _lock:
        _probes[tool_name] = probe
        _cache.pop(tool_name, None)


def register_default(tool_name: str, probe: Probe) -> None:
    """Register only if nothing is registered for this tool yet.

    install_default_probes() is called on every readiness query and every
    enriched failure, so it must be safe to re-run. Using register() there would
    silently overwrite any probe a caller -- or a test -- had registered
    explicitly, the moment the feature was exercised.
    """
    with _lock:
        if tool_name in _probes:
            return
        _probes[tool_name] = probe
        _cache.pop(tool_name, None)


def registered_names() -> list[str]:
    with _lock:
        return sorted(_probes)


def resolve(tool_name: str, *, refresh: bool = False) -> Readiness:
    """Readiness for one tool. Never raises."""
    now = time.monotonic()
    with _lock:
        probe = _probes.get(tool_name)
        if not refresh:
            hit = _cache.get(tool_name)
            if hit is not None and now - hit[0] < CACHE_TTL_SECONDS:
                return hit[1]
    if probe is None:
        return Readiness(UNKNOWN, "no readiness check defined for this tool")
    try:
        result = probe()
        if not isinstance(result, Readiness):  # a probe that returns junk is a bug, not a crash
            result = Readiness(UNKNOWN, f"probe returned {type(result).__name__}, not Readiness")
    except BaseException as exc:  # noqa: BLE001 - a probe must never break a caller
        result = Readiness(UNKNOWN, f"readiness check failed: {exc}")
    with _lock:
        _cache[tool_name] = (now, result)
    return result


def resolve_all(*, refresh: bool = False) -> dict[str, Readiness]:
    return {name: resolve(name, refresh = refresh) for name in registered_names()}
```

- [ ] **Step 4: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_tool_readiness_registry.py -v -p no:cacheprovider
```
Expected: 9 passed.

- [ ] **Step 5: Prove the unknown-default control is live**

Two controls, both demonstrated:

1. Temporarily change `resolve`'s `if probe is None:` branch to return
   `Readiness(READY, "assumed fine")`. Re-run and confirm
   `test_unregistered_tool_is_unknown_not_ready` FAILS. Restore it and confirm it passes.
   This is the single assumption the whole feature's honesty rests on.
2. Temporarily make `register_default` delegate to `register` (clobbering). Re-run and confirm
   `test_register_default_does_NOT_overwrite_an_existing_probe` FAILS. Restore it.
   This one protects Tasks 3 and 4, which call `install_default_probes()` on every invocation.

- [ ] **Step 6: Commit**

```bash
git add studio/backend/core/inference/tool_readiness/__init__.py studio/backend/tests/test_tool_readiness_registry.py
git commit -m "feat(readiness): the tool readiness registry

Three states, not a boolean: a tool with no probe reports unknown, never ready.
Defaulting unknown to ready is the optimistic lie that makes this feature worse
than not having it, because the model would read 'nobody checked' as 'verified
working'.

Probes are cached for 60s and guarded by a bare except: a probe that raises
reports unknown carrying its error, and can never break a caller.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: The probes

**Files:**
- Create: `studio/backend/core/inference/tool_readiness/probes.py`
- Test: `studio/backend/tests/test_tool_readiness_probes.py`

This task creates one module and its test. It does **not** modify `__init__.py` — the lazy
`install_default_probes()` call lives in Task 3's `execute()` and Task 4's `_enrich_failure()`.

**Interfaces:**
- Consumes: `Readiness`, `register_default`, `READY`/`MISSING`/`UNKNOWN` from Task 1
- Produces:
  - `install_default_probes() -> None` — registers every v1 probe; idempotent
  - `ALWAYS_READY_TOOLS: frozenset[str]`

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_tool_readiness_probes.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The concrete probes.

Every probe delegates to a helper that already exists. Reimplementing those
checks is how the audit log's redaction layer drifted from the upstream
classifier it was supposed to reuse.
"""

from __future__ import annotations

import pytest

from core.inference import tool_readiness as tr
from core.inference.tool_readiness import probes


@pytest.fixture(autouse = True)
def _clean():
    tr.reset_for_tests()
    probes.install_default_probes()
    yield
    tr.reset_for_tests()


def test_trivially_ready_tools_are_ready():
    for name in ("terminal", "python", "edit_file", "render_html", "search_conversation"):
        assert tr.resolve(name).state == tr.READY, name


def test_search_knowledge_base_is_unknown_in_v1():
    """Deliberate: storage/ exposes no cheap KB-count helper, so we do not guess."""
    assert tr.resolve("search_knowledge_base").state == tr.UNKNOWN


def test_web_search_missing_when_ddgs_is_absent(monkeypatch):
    monkeypatch.setattr(probes, "_module_present", lambda name: False)
    r = tr.resolve("web_search", refresh = True)
    assert r.state == tr.MISSING
    assert r.missing == "ddgs"


def test_web_search_ready_when_ddgs_is_present(monkeypatch):
    """Control, and the real state on a correctly built install."""
    monkeypatch.setattr(probes, "_module_present", lambda name: True)
    assert tr.resolve("web_search", refresh = True).state == tr.READY


def test_web_search_does_not_claim_network_reachability():
    """ready means 'the package is present'. Saying more would be a lie the
    spec's no-network-calls rule makes unavoidable -- so say it out loud."""
    r = probes._probe_web_search()
    assert "network" in r.detail.lower()


def test_webcam_look_missing_when_the_weight_is_absent(monkeypatch):
    monkeypatch.setattr(probes, "_yolo_weight_present", lambda: False)
    r = tr.resolve("webcam_look", refresh = True)
    assert r.state == tr.MISSING
    assert r.missing and "yolov8n" in r.missing


def test_webcam_look_ready_when_the_weight_is_present(monkeypatch):
    """Control: without this, a probe hardcoded to 'missing' would pass above."""
    monkeypatch.setattr(probes, "_yolo_weight_present", lambda: True)
    assert tr.resolve("webcam_look", refresh = True).state == tr.READY


def test_face_swap_missing_until_the_licence_is_accepted(monkeypatch):
    monkeypatch.setattr(probes, "_face_swap_licence_accepted", lambda: False)
    r = tr.resolve("face_swap", refresh = True)
    assert r.state == tr.MISSING
    assert r.remedy and "licence" in r.remedy.lower()


def test_face_swap_ready_once_accepted(monkeypatch):
    monkeypatch.setattr(probes, "_face_swap_licence_accepted", lambda: True)
    assert tr.resolve("face_swap", refresh = True).state == tr.READY


def test_code_tool_reports_per_language_breakdown(monkeypatch):
    monkeypatch.setattr(probes, "_language_servers", lambda: {"typescript": True, "csharp": False})
    r = tr.resolve("code_definition", refresh = True)
    assert r.state == tr.READY, "usable for at least one language"
    assert "typescript" in r.detail and "csharp" in r.detail


def test_code_tool_missing_when_no_language_server_is_installed(monkeypatch):
    monkeypatch.setattr(probes, "_language_servers", lambda: {"typescript": False, "csharp": False})
    r = tr.resolve("code_definition", refresh = True)
    assert r.state == tr.MISSING


def test_probe_names_match_real_tools():
    """A probe registered under a name no tool has is dead code that looks alive."""
    from core.inference.tools import ALL_TOOLS

    real = {t["function"]["name"] for t in ALL_TOOLS}
    for name in tr.registered_names():
        assert name in real, f"probe registered for unknown tool {name!r}"
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_tool_readiness_probes.py -v -p no:cacheprovider
```
Expected: `ModuleNotFoundError: No module named 'core.inference.tool_readiness.probes'`.

- [ ] **Step 3: Implement the probes**

Create `studio/backend/core/inference/tool_readiness/probes.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Concrete readiness probes.

Each one DELEGATES to a helper that already exists rather than reimplementing
its check -- assist_code.servers for language servers, assist_vision.yolo for
the weight, face_swap.licence_accepted for the gate. Reimplementing is how the
audit log's redaction layer drifted from the upstream classifier it was meant to
reuse.

The module-level `_`-prefixed indirections exist so tests can substitute them
without touching the real filesystem, PATH or licence state.
"""

from __future__ import annotations

import importlib.util
import os

from core.inference.tool_readiness import MISSING, READY, UNKNOWN, Readiness, register_default

# No external dependency: if the process is running, these work.
ALWAYS_READY_TOOLS = frozenset({
    "terminal",
    "python",
    "edit_file",
    "render_html",
    "search_conversation",
})

_CODE_TOOL_NAMES = (
    "code_definition",
    "code_references",
    "code_hover",
    "code_symbols",
    "code_diagnostics",
)


def _yolo_weight_present() -> bool:
    """weights_path() returns the cached file when it exists, else the bare
    filename (ultralytics' cue to download). isfile() therefore distinguishes
    'already here' from 'would be fetched on first use'."""
    from core.inference.assist_vision import yolo

    return os.path.isfile(yolo.weights_path())


def _face_swap_licence_accepted() -> bool:
    from core.inference.assist_vision.face_swap import licence_accepted

    return bool(licence_accepted())


def _module_present(name: str) -> bool:
    """find_spec locates a module WITHOUT importing it -- ~2 ms, no side effects.
    It raises for a submodule whose parent is absent, so it is guarded here as
    well as at the registry level."""
    try:
        return importlib.util.find_spec(name) is not None
    except BaseException:  # noqa: BLE001
        return False


def _language_servers() -> dict[str, bool]:
    """{language: installed}. Binary names come from servers._SERVERS, never
    hardcoded here, so a rename upstream cannot leave this silently wrong."""
    from core.inference.assist_code import servers

    out: dict[str, bool] = {}
    for language in servers.SUPPORTED:
        spec = servers._SERVERS.get(language)
        out[language] = bool(spec and servers._which(spec["binary"]))
    return out


def _probe_always_ready() -> Readiness:
    return Readiness(READY, "no external dependency")


def _probe_unknown_kb() -> Readiness:
    return Readiness(
        UNKNOWN,
        "no cheap knowledge-base check available; call it to find out",
    )


def _probe_webcam_look() -> Readiness:
    if _yolo_weight_present():
        return Readiness(READY, "yolov8n.pt present")
    return Readiness(
        MISSING,
        "yolov8n.pt not in the model cache",
        missing = "yolov8n.pt",
        remedy = "ultralytics downloads it (~6 MB) on first use",
    )


def _probe_face_swap() -> Readiness:
    if _face_swap_licence_accepted():
        return Readiness(READY, "InsightFace licence accepted")
    return Readiness(
        MISSING,
        "InsightFace licence not accepted, so its models cannot be downloaded",
        missing = "InsightFace model licence acceptance",
        remedy = "accept the InsightFace licence before using face swap",
    )


def _probe_web_search() -> Readiness:
    if _module_present("ddgs"):
        return Readiness(READY, "ddgs present; network reachability not checked")
    return Readiness(
        MISSING,
        "the ddgs package is not importable; network reachability not checked either",
        missing = "ddgs",
        remedy = "pip install ddgs (or declare it in the frozen build)",
    )


def _probe_code_tool() -> Readiness:
    servers = _language_servers()
    have = [lang for lang, ok in servers.items() if ok]
    detail = ", ".join(f"{lang} {'OK' if ok else 'not installed'}" for lang, ok in sorted(servers.items()))
    if have:
        return Readiness(READY, detail)
    return Readiness(
        MISSING,
        detail,
        missing = "a language server for any supported language",
        remedy = "installed automatically on first use, or install manually (see the tool description)",
    )


def install_default_probes() -> None:
    """Register every v1 probe.

    Called on EVERY readiness query and every enriched failure, so it must be
    safe to re-run -- hence register_default, which leaves an already-registered
    probe alone. Using register() here would silently overwrite a caller's
    explicit registration the moment the feature was exercised.
    """
    for name in ALWAYS_READY_TOOLS:
        register_default(name, _probe_always_ready)
    register_default("search_knowledge_base", _probe_unknown_kb)
    register_default("web_search", _probe_web_search)
    register_default("webcam_look", _probe_webcam_look)
    register_default("face_swap", _probe_face_swap)
    for name in _CODE_TOOL_NAMES:
        register_default(name, _probe_code_tool)
```

- [ ] **Step 4: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_tool_readiness_probes.py -v -p no:cacheprovider
```
Expected: 12 passed.

All names above were verified against the live registry while this plan was written — `ALL_TOOLS`
holds exactly 17 tools and the five code names are `code_definition`, `code_references`,
`code_hover`, `code_symbols`, `code_diagnostics`. `test_probe_names_match_real_tools` re-checks this
at run time, so a rename upstream surfaces as a test failure rather than a dead probe.

- [ ] **Step 5: Prove the probe controls are live**

Temporarily make `_probe_webcam_look` return `Readiness(MISSING, "x", missing = "yolov8n.pt")`
unconditionally. Re-run and confirm `test_webcam_look_ready_when_the_weight_is_present` FAILS while
the "missing" test still passes — that is what proves the pair is testing the probe's logic rather
than a constant. Restore.

- [ ] **Step 6: Commit**

```bash
git add studio/backend/core/inference/tool_readiness/probes.py studio/backend/tests/test_tool_readiness_probes.py
git commit -m "feat(readiness): the v1 probes

Each delegates to a helper that already exists -- assist_code.servers for
language servers (binary names read from _SERVERS, never hardcoded),
assist_vision.yolo.weights_path() for the weight, face_swap.licence_accepted()
for the gate.

search_knowledge_base reports unknown deliberately: storage/ exposes no cheap
knowledge-base count helper, and inventing a query would be guessing at a schema
this piece does not own. unknown is accurate.

web_search is NOT in the always-ready set. It lazily imports ddgs, and a frozen
build missing a lazily-imported module is this project's most common defect
class. find_spec costs ~2ms and imports nothing. Network reachability is not
checked -- the spec forbids network calls -- so detail says so rather than
letting 'ready' imply more than it means.

Code tools report 'usable for at least one language' with a per-language
breakdown in detail, because csharp-ls can be installed while
typescript-language-server is not.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: The tool and the seam

**Files:**
- Create: `studio/backend/core/inference/tool_readiness/schemas.py`
- Modify: `studio/backend/core/inference/tool_readiness/__init__.py` (add `execute`)
- Modify: `studio/backend/core/inference/tools.py` (+7 lines)
- Test: `studio/backend/tests/test_tool_readiness_seam.py`

**Interfaces:**
- Consumes: `resolve`, `resolve_all`, `registered_names` (Task 1), `install_default_probes` (Task 2)
- Produces:
  - `READINESS_TOOLS: list[dict]`, `READINESS_TOOL_NAMES: frozenset[str]`
  - `tool_readiness.execute(name: str, arguments: dict) -> str`

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_tool_readiness_seam.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The check_tool_readiness tool, its dispatch, and the _ALWAYS_SAFE_TOOLS rebind.

The rebind test is behavioural on purpose. It depends on _ALWAYS_SAFE_TOOLS being
read as a module global at CALL time; if that were wrong the rebind would
silently do nothing, every readiness query would prompt the user for approval,
and no other test would notice. Asserting the rebind's presence rather than its
effect is exactly the inert control this project has produced nine times.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core.inference import tool_audit
from core.inference import tools
from storage import tool_audit_db


@pytest.fixture(autouse = True)
def _isolated(tmp_path, monkeypatch):
    """REQUIRED, and not optional here.

    execute_tool is the audit shadow, so calling it writes an audit row. Without
    this redirect those rows land in the real ~/.unsloth/studio/studio.db -- and
    because the recorder is guarded to never raise, it would do so silently.
    The existing test_tool_audit_seam.py needs no fixture only because its checks
    are static; these actually invoke the tool.
    """
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    tool_audit_db.reset_for_tests()
    tool_audit.reset_degraded_for_tests()
    yield


def test_the_tool_is_registered_in_ALL_TOOLS():
    names = {t["function"]["name"] for t in tools.ALL_TOOLS}
    assert "check_tool_readiness" in names


def test_execute_tool_dispatches_to_the_readiness_handler():
    out = tools.execute_tool("check_tool_readiness", {})
    assert isinstance(out, str) and out.strip(), "should return a non-empty report"
    assert "terminal" in out, "the report should list known tools"


def test_asking_about_one_tool_reports_only_that_tool():
    out = tools.execute_tool("check_tool_readiness", {"tool": "terminal"})
    assert "terminal" in out
    assert "webcam_look" not in out


def test_an_unknown_tool_name_is_reported_not_an_error():
    out = tools.execute_tool("check_tool_readiness", {"tool": "no_such_tool"})
    assert "unknown" in out.lower()


def test_readiness_is_not_treated_as_high_risk():
    """THE load-bearing test. Without the additive _ALWAYS_SAFE_TOOLS rebind,
    unknown tools fail closed and every readiness query prompts the user."""
    assert tools.is_high_risk_tool_call("check_tool_readiness", {}) is False


def test_a_tool_outside_the_safe_set_IS_still_high_risk():
    """Control: proves the assertion above is not passing because everything is
    considered safe."""
    assert tools.is_high_risk_tool_call("zz_not_a_real_tool", {}) is True


def test_the_seam_stays_additive():
    repo = Path(tools.__file__).parents[4]
    base = subprocess.run(
        ["git", "merge-base", "origin/main", "HEAD"],
        cwd = repo, capture_output = True, text = True, check = True,
    ).stdout.strip()
    stat = subprocess.run(
        ["git", "diff", "--numstat", base, "--", "studio/backend/core/inference/tools.py"],
        cwd = repo, capture_output = True, text = True, check = True,
    ).stdout.split()
    assert stat, "no diff recorded for tools.py"
    insertions, deletions = int(stat[0]), int(stat[1])
    assert deletions == 0, f"tools.py lost {deletions} line(s); the seam must be additive"
    assert insertions <= 90, f"seam grew to {insertions}; budget is ~84"
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_tool_readiness_seam.py -v -p no:cacheprovider
```
Expected: the registration, dispatch and rebind tests fail; `test_the_seam_stays_additive` and
`test_a_tool_outside_the_safe_set_IS_still_high_risk` already pass.

- [ ] **Step 3: Create the schema**

Create `studio/backend/core/inference/tool_readiness/schemas.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI-style schema for the readiness tool."""

CHECK_TOOL_READINESS_TOOL = {
    "type": "function",
    "function": {
        "name": "check_tool_readiness",
        "description": (
            "Report whether tools will actually work before you call them: whether their "
            "models, language servers or connections are present. States are 'ready', "
            "'missing' (with what is absent and how to obtain it) and 'unknown' (no check "
            "exists -- calling it is the only way to find out). Read-only and cheap. Use it "
            "when a capability might be unavailable, or after a tool fails unexpectedly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": "One tool name. Omit to report every tool with a known check.",
                },
            },
            "required": [],
        },
    },
}

READINESS_TOOLS = [CHECK_TOOL_READINESS_TOOL]
READINESS_TOOL_NAMES = frozenset({"check_tool_readiness"})
```

- [ ] **Step 4: Add `execute` to the package**

Append to `studio/backend/core/inference/tool_readiness/__init__.py`:

```python
def _format(name: str, r: Readiness) -> str:
    line = f"{name:<24} {r.state:<9} {r.detail}"
    if r.remedy:
        line += f"\n{'':<34}-> {r.remedy}"
    return line


def execute(name: str, arguments: dict) -> str:
    """Handler for the check_tool_readiness tool. Never raises."""
    try:
        from core.inference.tool_readiness.probes import install_default_probes

        install_default_probes()
        requested = (arguments or {}).get("tool")
        if requested:
            return _format(str(requested), resolve(str(requested)))
        rows = resolve_all()
        if not rows:
            return "No readiness checks are registered."
        return "\n".join(_format(n, rows[n]) for n in sorted(rows))
    except BaseException as exc:  # noqa: BLE001 - a readiness query must never break a turn
        return f"Readiness check unavailable: {exc}"
```

- [ ] **Step 5: Add the seam to tools.py**

Three additive edits. **Do not modify or re-indent any existing line.**

(a) In the import area beside the existing fork imports at `tools.py:9834-9835`, add:

```python
from core.inference.tool_readiness.schemas import READINESS_TOOLS, READINESS_TOOL_NAMES
```

(b) In `ALL_TOOLS` (`tools.py:9837`), add one entry after `*ASSIST_CODE_TOOLS,`:

```python
    *READINESS_TOOLS,
```

(c) In `execute_tool`'s dispatch chain, directly after the `ASSIST_CODE_TOOL_NAMES` block
(`tools.py:10071-10076`), add:

```python
    if name in READINESS_TOOL_NAMES:
        from core.inference import tool_readiness
        return tool_readiness.execute(name, arguments)
```

(d) Immediately after the `_ALWAYS_SAFE_TOOLS` definition at `tools.py:4535`, add:

```python
# fork: additive rebind rather than editing the frozenset literal above, which
# would be a DELETION. All three readers (below, _is_potentially_unsafe, and
# is_high_risk_tool_call) look this name up as a module global at CALL time, so
# the rebind is seen -- verified by running it, not assumed.
_ALWAYS_SAFE_TOOLS = _ALWAYS_SAFE_TOOLS | frozenset({"check_tool_readiness"})
```

- [ ] **Step 6: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_tool_readiness_seam.py -v -p no:cacheprovider
```
Expected: 7 passed. Then confirm the seam by hand:
```
git diff --numstat $(git merge-base origin/main HEAD) -- studio/backend/core/inference/tools.py
```
Expected: roughly `84  0` — the second number MUST be 0.

- [ ] **Step 7: Prove the rebind control is live**

Delete the line added in step 5(d). Re-run and confirm `test_readiness_is_not_treated_as_high_risk`
FAILS. Restore it and confirm it passes. This is the test that stops the feature shipping in a state
where it prompts the user on every query.

- [ ] **Step 8: Confirm nothing else broke**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_tool_audit_seam.py tests/test_tool_readiness_registry.py tests/test_tool_readiness_probes.py -v -p no:cacheprovider
```
Expected: all pass. The audit seam test asserts `tools.py` stays additive too, so a regression there
shows up immediately.

- [ ] **Step 9: Commit**

```bash
git add studio/backend/core/inference/tool_readiness/ studio/backend/core/inference/tools.py studio/backend/tests/test_tool_readiness_seam.py
git commit -m "feat(readiness): expose check_tool_readiness to the model

Registered the way the fork already registers its tools: one ALL_TOOLS entry and
a dispatch block mirroring the vision/code ones. tools.py stays additive.

The _ALWAYS_SAFE_TOOLS rebind is the non-obvious part. A read-only readiness
query must not trigger an approval prompt, but that frozenset is a literal --
editing it would be a DELETION. Rebinding after the original is additive, and it
works because all three readers look the name up as a module global at call time.
Verified by running it rather than assuming: before the rebind
is_high_risk_tool_call returns True, after it returns False.

The test asserts that BEHAVIOUR, not the rebind's presence, with a control
proving not everything is considered safe -- assert-the-line tests are how this
project produced nine inert controls.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Enriched failures

**Files:**
- Modify: `studio/backend/core/inference/tool_audit/__init__.py`
- Test: `studio/backend/tests/test_tool_readiness_enrichment.py`

**Interfaces:**
- Consumes: `tool_readiness.resolve` (Task 1), `install_default_probes` (Task 2)
- Produces: no new public names; `around()`'s return value gains an appended line in one case

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_tool_readiness_enrichment.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A failed call whose dependency is missing should say so.

Scoped tightly: only when the call FAILED and the probe reports MISSING. The
original text is never altered, only appended to. Successful calls are never
touched -- an audit/readiness layer that rewrites successful results would be a
behaviour change on a path that works.
"""

from __future__ import annotations

import pytest

from core.inference import tool_audit
from core.inference import tool_readiness as tr
from storage import tool_audit_db


@pytest.fixture(autouse = True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    tool_audit_db.reset_for_tests()
    tool_audit.reset_degraded_for_tests()
    tr.reset_for_tests()
    yield
    tr.reset_for_tests()


def _failing(name, arguments, **kwargs):
    return "Error: something went wrong"


def _succeeding(name, arguments, **kwargs):
    return "all good"


def test_a_failure_with_a_missing_dependency_is_explained():
    """The explicit tr.register below must survive the enricher's own
    install_default_probes() call -- that is what register_default (Task 1)
    guarantees. If defaults clobbered it, this test would resolve the REAL
    face_swap probe and fail on text it never wrote."""
    tr.register("face_swap", lambda: tr.Readiness(
        tr.MISSING, "licence not accepted",
        missing = "InsightFace licence", remedy = "accept it in settings",
    ))
    out = tool_audit.around(_failing, "face_swap", {})
    assert out.startswith("Error: something went wrong"), "original text must survive intact"
    assert "[readiness]" in out
    assert "InsightFace licence" in out
    assert "accept it in settings" in out


def test_a_SUCCESSFUL_call_is_never_enriched():
    """Control: enrichment must not touch a working path."""
    tr.register("face_swap", lambda: tr.Readiness(tr.MISSING, "x", missing = "y"))
    out = tool_audit.around(_succeeding, "face_swap", {})
    assert out == "all good"


def test_a_failure_whose_probe_says_ready_is_not_enriched():
    """Control: enrichment keys off MISSING, not off failure alone."""
    tr.register("terminal", lambda: tr.Readiness(tr.READY, "fine"))
    out = tool_audit.around(_failing, "terminal", {})
    assert out == "Error: something went wrong"


def test_an_unknown_readiness_does_not_enrich():
    out = tool_audit.around(_failing, "no_probe_tool", {})
    assert out == "Error: something went wrong"


def test_enrichment_failure_cannot_break_the_call(monkeypatch):
    def explode(*a, **k):
        raise OSError("probe subsystem down")

    monkeypatch.setattr(tr, "resolve", explode)
    out = tool_audit.around(_failing, "face_swap", {})
    assert out == "Error: something went wrong", "a broken enricher must not cost the caller its result"
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_tool_readiness_enrichment.py -v -p no:cacheprovider
```
Expected: `test_a_failure_with_a_missing_dependency_is_explained` FAILS (no `[readiness]` in the
output); the four control tests already pass, because nothing is enriched yet.

- [ ] **Step 3: Implement enrichment**

In `studio/backend/core/inference/tool_audit/__init__.py`, add this helper above `around`:

```python
def _enrich_failure(name: Any, result: Any) -> Any:
    """Append one line when a FAILED call's dependency is missing.

    Returned error strings only, never raised exceptions: enriching an exception
    would mean raising a different one, changing its identity on a path that
    works. The audit row already records the exception in full.

    Guarded: a broken enricher must never cost the caller its result.
    """
    try:
        if not isinstance(result, str) or not result.startswith("Error:"):
            return result
        from core.inference import tool_readiness
        from core.inference.tool_readiness.probes import install_default_probes

        install_default_probes()
        r = tool_readiness.resolve(str(name))
        if r.state != tool_readiness.MISSING:
            return result
        line = f"\n\n[readiness] {name} is missing: {r.missing or r.detail}"
        if r.remedy:
            line += f"\n            -> {r.remedy}"
        return result + line
    except BaseException:  # noqa: BLE001 - never-raises; see module docstring
        return result
```

Then, in `around`, change the final success return to enrich first. The existing lines are:

```python
    _finish(row_id, started, outcome = "ok", result = result, withhold = withhold)
    return result
```

Replace the `return result` line only with:

```python
    return _enrich_failure(args[0] if args else kwargs.get("name"), result)
```

The `_finish` call is unchanged, so the audit row keeps the ORIGINAL result — enrichment is for the
model, not for the record.

- [ ] **Step 4: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_tool_readiness_enrichment.py -v -p no:cacheprovider
```
Expected: 5 passed.

- [ ] **Step 5: Prove the success control is live**

Temporarily remove the `result.startswith("Error:")` condition from `_enrich_failure` so it enriches
everything. Re-run and confirm `test_a_SUCCESSFUL_call_is_never_enriched` FAILS. Restore it.

- [ ] **Step 6: Confirm the audit log still records the original**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_tool_audit_recorder.py tests/test_tool_audit_db.py -v -p no:cacheprovider
```
Expected: all pass. Enrichment must not have changed what is stored.

- [ ] **Step 7: Commit**

```bash
git add studio/backend/core/inference/tool_audit/__init__.py studio/backend/tests/test_tool_readiness_enrichment.py
git commit -m "feat(readiness): explain a failure caused by a missing dependency

When a call fails AND its probe reports the dependency missing, one line is
appended naming what is absent and how to get it. The original text is never
altered.

Scoped deliberately: returned error strings only, not raised exceptions.
Enriching an exception would mean raising a different one, changing its identity
on a path that works -- the same class of mistake as the audit log's str(result)
outside its guard.

The audit row keeps the ORIGINAL result: enrichment is for the model, not the
record.

Controls prove it keys off MISSING rather than off failure alone, that a
successful call is never touched, and that a broken enricher cannot cost the
caller its result.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage.** Registry and three states → Task 1. Cheap cached probes delegating to existing
helpers, and the per-language breakdown → Task 2. The tool, additive registration and the
`_ALWAYS_SAFE_TOOLS` rebind with its behavioural control → Task 3. Enrichment with its
returned-strings-only limitation → Task 4. Error handling (bare `except BaseException`) appears in
Tasks 1, 3 and 4.

**Two deliberate deviations from the spec, recorded rather than glossed.**

The spec's dependency table lists `search_knowledge_base` as needing an index. Task 2 registers it as
`unknown` instead: `storage/` exposes no cheap knowledge-base count helper, and inventing one would be
guessing at a schema this piece does not own. The three-state design exists precisely so this is
expressible rather than fudged.

The spec's table also lists MCP tools as depending on "its server being connected". v1 registers no
MCP probe — `mcp__*` names are created at runtime, so there is no static name to register against —
and they fall through to the registry's `unknown` default. The spec's own worked example already
shows `mcp__fs__read_file  unknown`, so this matches its output while narrowing its table.

Both deviations move coverage toward `unknown`, never toward `ready`. That direction is the one the
spec's central argument permits.

**One addition beyond the spec.** `web_search` was going to sit in the always-ready set, but it
lazily imports `ddgs`; a probe costing ~2 ms was verified before the plan was finalised. Claiming
"ready" for a tool whose module might be absent from a frozen build is the exact optimistic lie the
spec forbids, in the defect class this project has hit most often.

**No UI and no HTTP surface in this piece**, so `main.py` is untouched and stays at 4 / 0 — the
spec's "Out of scope" says the same.

**A defect caught in this plan's own review, recorded because the shape recurs.** Task 3's tests
call `execute_tool`, which is now the audit shadow, so each call writes an audit row. The first
draft had no isolation fixture, and since the recorder is guarded to never raise, those rows would
have landed in the real `~/.unsloth/studio/studio.db` **silently** — a guard designed to protect the
caller turning a test-pollution bug into an invisible one. This is the fourth instance in this
project of "state an invariant, then violate it at the edge the invariant did not consider".

**Placeholders.** None; every code step carries real code.

**Type consistency.** `Readiness(state, detail, missing, remedy)` is constructed identically in
Tasks 1, 2 and 4. `READY`/`MISSING`/`UNKNOWN` are defined in Task 1 and imported by name thereafter.
`resolve` / `resolve_all` / `registered_names` / `reset_for_tests` keep one signature throughout.
`READINESS_TOOLS` and `READINESS_TOOL_NAMES` are defined in Task 3 and used only there.

**Ordering.** Task 2 depends on Task 1; Task 3 on both; Task 4 on 1 and 2. They must run in order.
