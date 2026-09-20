# Unslothed Model Roles and Delegation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind a model to a role, and let the loaded model hand a task to another role's model — brief out, artifact and short result back — then restore the model that was resident before.

**Architecture:** `core/inference/model_roles/` stores role → model bindings in one app setting and reports each role's availability in the readiness vocabulary. `core/inference/delegation/` adds an `ask_model` tool that writes a brief into the chat's workspace, swaps the chat model, runs a bounded sub-agent through the audited `execute_tool`, writes its artifact and transcript, then restores the previously resident model in a `finally`. Every moving part the tests cannot afford — the loader and the model call — is an injected callable, so no test loads a model.

**Tech Stack:** Python 3.12, pytest, FastAPI (one new router). No new dependencies.

**Spec:** `C:\Users\Admin\odysseus\docs\superpowers\specs\2026-09-20-unslothed-model-roles-delegation-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Repo:** `C:\Users\Admin\unsloth`. Branch `model-delegation` off `unslothed-release` (`6b552aba`). `origin` is UPSTREAM `unslothai/unsloth` and **must never be pushed to**; the fork remote is `fork`. Commit locally only.
- **`studio/backend/core/inference/tools.py` must stay purely additive: zero deletions.** On this base it is **88 insertions / 0 deletions**; this plan takes it to **93 / 0**. A line the FORK inserted may be amended (upstream has no such line, so it stays one insertion); a line UPSTREAM owns must never be edited. Verify:
  `git diff --numstat $(git merge-base origin/main HEAD) -- studio/backend/core/inference/tools.py`
- **`studio/backend/routes/inference.py`** is **58 / 0** on this base and goes to **59 / 0**. Same rule.
- **`studio/backend/main.py`** is 4 / 0 and **gains two lines** (the new router's import and its `include_router`) → 6 / 0, additive.
- **Do-not-edit:** `core/inference/llama_cpp.py`, `pyproject.toml`.
- **NEVER run the full backend test suite.** Upstream fixtures fabricate GGUF files up to 40 GB and filled the disk once. Run only the files each task names.
- **Test runner**, from `C:\Users\Admin\unsloth\studio\backend`, plain command, no `--import-mode` flag:
  `C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest <files> -p no:cacheprovider -q`
- **No test may load a model, start llama-server, or call a real backend.** The loader and the model call are injected callables.
- **Never-raises guards use a bare `except BaseException`**, not `except Exception`.
- **Style:** keyword arguments take spaces around `=`.
- **Every negative control is demonstrated to fail.** Eleven inert controls have appeared in this project, several authored by the plan's author — including a production "fix" that could never fire. **Run mutation controls AFTER committing the task**, restore each immediately with `git checkout -- <file>`, and finish with `git status --porcelain` clean.
- **Commit trailer:** end every commit message with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## Coordination note

`capability-map` is open as **PR #2 against `unslothed-release` and is not merged**. It amends the same registration lines this plan amends (`ALL_TOOLS`, the `execute_tool` dispatch, the `_addable` union). Whichever merges second resolves a small conflict in those three places. This plan deliberately does **not** depend on `capability_map` code, so either order works.

## File Structure

**Created:**

| Path | Responsibility |
|---|---|
| `studio/backend/core/inference/model_roles/__init__.py` | `RoleBinding`, `bindings()`, `resolve()`, `availability()` |
| `studio/backend/core/inference/model_roles/storage.py` | read/write the `model_roles` app setting |
| `studio/backend/routes/model_roles.py` | auth-gated `GET`/`PUT /api/model-roles` |
| `studio/backend/core/inference/delegation/__init__.py` | `execute()` — the `ask_model` handler and its orchestration |
| `studio/backend/core/inference/delegation/workspace.py` | the delegation folder, brief, work file, transcript |
| `studio/backend/core/inference/delegation/subagent.py` | the bounded sub-agent loop |
| `studio/backend/core/inference/delegation/loader.py` | the model swap adapter (the only code that touches the real loader) |
| `studio/backend/core/inference/delegation/schemas.py` | the `ask_model` schema, `DELEGATION_TOOLS`, `DELEGATION_TOOL_NAMES` |
| `studio/backend/tests/test_model_roles.py` | Task 1 |
| `studio/backend/tests/test_model_roles_api.py` | Task 2 |
| `studio/backend/tests/test_delegation_workspace.py` | Task 3 |
| `studio/backend/tests/test_delegation_loader.py` | Task 4 |
| `studio/backend/tests/test_delegation_subagent.py` | Task 5 |
| `studio/backend/tests/test_delegation_tool.py` | Task 6 |
| `studio/backend/tests/test_delegation_audit.py` | Task 7 |
| `studio/backend/tests/test_delegation_seam.py` | Task 8 |

**Modified:**

| Path | Change |
|---|---|
| `studio/backend/storage/tool_audit_db.py` | Task 7: nullable `delegation_id` column + migration |
| `studio/backend/core/inference/tool_audit/__init__.py` | Task 7: read the delegation context variable |
| `studio/backend/main.py` | Task 8: register the roles router (+2 lines) |
| `studio/backend/core/inference/tools.py` | Task 8: +5 lines |
| `studio/backend/routes/inference.py` | Task 8: +1 line |

## Facts verified against live code while writing this plan

Implementers need not re-derive these; reviewers may check them.

- Settings: `storage.studio_db.get_app_setting(key, fallback = None)` reads; `upsert_app_settings(settings: dict) -> dict` writes. `utils/openai_auto_switch_settings.py::_cached_setting(key, default)` is the memoised read pattern, and `model_override_load_kwargs(override, *, is_gguf)` converts an override into load kwargs.
- `core/inference/local_model_resolver.py::resolve_local_gguf(requested, *, allow_scan = True) -> Optional[tuple[str, Optional[str], str]]` returns `(load_path, gguf_variant, loader_id)` for a downloaded local model, else `None`.
- `core/inference/tools.py::_get_workdir(session_id = None) -> str` is the session workdir. `assist_vision/__init__.py::_write_png`'s docstring records the rule: tools return a path, and `resolve_image_bytes` refuses any path outside the conversation's working directory.
- `routes/inference.py::_load_model_impl(request: LoadRequest, fastapi_request: Request, current_subject: str, *, ...)` is the real loader. It uses `fastapi_request` in exactly two places — `_request_used_api_key(request: Any)`, whose own comment says it is "Total by construction… must never fail a load", and `_resolve_parallel_slots`, which is a pure `getattr` chain defaulting to 1 slot. Both tolerate `None`.
- `core/inference/gpu_arbiter.py` exposes `acquire_for(owner)`, `release(owner)`, `current_owner()` and the constant `CHAT`.
- `core/inference/llama_cpp.py` (do-not-edit, read only) exposes `is_loaded`, `model_identifier`, `base_url` (`http://127.0.0.1:<port>`) and `_auth_headers` as properties.
- `core/inference/safetensors_agentic.py::run_safetensors_tool_loop` is dependency-injected (`single_turn`, `execute_tool`, `max_tool_iterations`, `tool_call_timeout`, `cancel_event`, `permission_mode`). This plan does not call it — the sub-agent loop here is model-call-agnostic for the same reason — but it is the precedent for the injected shape.
- `storage/tool_audit_db.py` has `_SCHEMA_READY` (uppercase) and `record_start(*, tool_name, arguments_json, paths_json, redacted, session_id, thread_id, disable_sandbox) -> int`. It has **no** migration pattern yet; `storage/rag_db.py:300-302` is the one to copy (`PRAGMA table_info` → check → `ALTER TABLE ... ADD COLUMN`).
- `routes/tool_audit.py` is the fork's router precedent: `router = APIRouter()` with `current_subject: str = Depends(get_current_subject)` on every endpoint, imported as `from auth.authentication import get_current_subject` (**not** `utils.auth`). `main.py` mounts it as `app.include_router(tool_audit_router, prefix = "/api/tool-audit", tags = ["tool-audit"])`, so each fork router carries its own full prefix and its routes use relative paths.
- **`LoadRequest` lives in `models/inference.py`, and its model field is `model_path`** (`model_path: str`, "Model identifier or local path") — not `model_name`, and not defined in `routes/inference.py`.
- `storage/tool_audit_db.py::_connect()` is a **context manager** (`Iterator[sqlite3.Connection]`), so it is used as `with tool_audit_db._connect() as conn:`.
- `tests/` is a package, but **test modules must not import each other** — `studio/__init__.py` + `studio/backend/__init__.py` make a bare `tests` package resolve elsewhere under pytest, and a collection error interrupts the whole run.

## Deviation from the spec, decided here

The spec caps delegations at **2 per assistant turn**. A tool cannot observe turn boundaries — `execute_tool` receives a session id, not a turn id. This plan implements the cap as **2 per session in any 10-minute window**, which is the closest observable approximation, and names it as such in the code and the refusal message. Everything else in the spec's budget table is implemented literally.

---

### Task 1: Role bindings

**Files:**
- Create: `studio/backend/core/inference/model_roles/__init__.py`, `studio/backend/core/inference/model_roles/storage.py`
- Test: `studio/backend/tests/test_model_roles.py`

**Interfaces:**
- Consumes: `storage.studio_db.get_app_setting` / `upsert_app_settings`; `core.inference.tool_readiness` state constants and `Readiness`
- Produces:
  - `model_roles.storage`: `ROLES_SETTING_KEY = "model_roles"`, `get_role_bindings() -> dict[str, dict]`, `set_role_bindings(raw: dict) -> dict[str, dict]`
  - `model_roles`: `DEFAULT_ROLE_NAMES: tuple[str, ...]`, `RoleBinding(role, model, overrides)`, `normalize_role(name) -> str`, `bindings() -> dict[str, RoleBinding]`, `resolve(role) -> Optional[RoleBinding]`, `availability(role) -> Readiness`, `_resolve_local(model_id)`

- [ ] **Step 1: Create the branch**

```bash
cd C:/Users/Admin/unsloth
git checkout unslothed-release
git checkout -b model-delegation
```

- [ ] **Step 2: Write the failing test**

Create `studio/backend/tests/test_model_roles.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Role bindings: a name like "coding" resolved to a concrete model.

Availability reuses tool readiness's three states deliberately. A role nobody
bound reports "unknown", never "ready" -- the same rule the readiness and
capability work rests on, for the same reason: the model must never read
"nobody checked" as "verified working".
"""

from __future__ import annotations

import pytest

from core.inference import model_roles
from core.inference.model_roles import storage
from core.inference.tool_readiness import MISSING, READY, UNKNOWN


@pytest.fixture(autouse = True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    yield


def _bind(monkeypatch, raw):
    monkeypatch.setattr(storage, "get_role_bindings", lambda: raw)


def test_a_binding_resolves_to_its_model(monkeypatch):
    _bind(monkeypatch, {"coding": {"model": "repo/Coder-GGUF:Q4_K_M"}})
    binding = model_roles.resolve("coding")
    assert binding is not None
    assert binding.model == "repo/Coder-GGUF:Q4_K_M"
    assert binding.overrides == {}


def test_overrides_are_carried_through(monkeypatch):
    _bind(monkeypatch, {"coding": {"model": "m", "overrides": {"n_ctx": 16384}}})
    assert model_roles.resolve("coding").overrides == {"n_ctx": 16384}


def test_role_names_are_normalized(monkeypatch):
    _bind(monkeypatch, {"coding": {"model": "m"}})
    assert model_roles.resolve("  Coding ") is not None
    assert model_roles.normalize_role("  Coding ") == "coding"


def test_an_unbound_role_resolves_to_none(monkeypatch):
    _bind(monkeypatch, {})
    assert model_roles.resolve("coding") is None


def test_a_malformed_binding_is_ignored_rather_than_raising(monkeypatch):
    _bind(monkeypatch, {"coding": "not-a-dict", "vision": {"no_model_key": 1}})
    assert model_roles.resolve("coding") is None
    assert model_roles.resolve("vision") is None


def test_an_unbound_role_is_unknown_not_ready(monkeypatch):
    """The central rule, at the role level."""
    _bind(monkeypatch, {})
    assert model_roles.availability("coding").state == UNKNOWN


def test_a_bound_downloaded_model_is_ready(monkeypatch):
    _bind(monkeypatch, {"coding": {"model": "m"}})
    monkeypatch.setattr(model_roles, "_resolve_local", lambda model: ("/models/m.gguf", None, "m"))
    result = model_roles.availability("coding")
    assert result.state == READY
    assert "m" in result.detail


def test_a_bound_model_that_is_not_downloaded_is_missing(monkeypatch):
    """Control for the test above."""
    _bind(monkeypatch, {"coding": {"model": "m"}})
    monkeypatch.setattr(model_roles, "_resolve_local", lambda model: None)
    result = model_roles.availability("coding")
    assert result.state == MISSING
    assert result.missing == "m"


def test_a_resolver_failure_is_unknown_and_never_raises(monkeypatch):
    _bind(monkeypatch, {"coding": {"model": "m"}})

    def boom(model):
        raise OSError("index unavailable")

    monkeypatch.setattr(model_roles, "_resolve_local", boom)
    result = model_roles.availability("coding")
    assert result.state == UNKNOWN
    assert "index unavailable" in result.detail


def test_bindings_returns_every_valid_role(monkeypatch):
    _bind(monkeypatch, {"coding": {"model": "a"}, "vision": {"model": "b"}, "bad": 3})
    assert set(model_roles.bindings()) == {"coding", "vision"}


def test_the_default_role_names_are_offered_but_not_required(monkeypatch):
    _bind(monkeypatch, {"my_custom_role": {"model": "m"}})
    assert "primary" in model_roles.DEFAULT_ROLE_NAMES
    assert model_roles.resolve("my_custom_role") is not None, "custom roles must work"


def test_settings_round_trip_through_the_app_setting(monkeypatch):
    saved = {}
    monkeypatch.setattr(storage, "_read_setting", lambda: saved.get("v"))
    monkeypatch.setattr(storage, "_write_setting", lambda value: saved.__setitem__("v", value))
    storage.set_role_bindings({"coding": {"model": "m", "overrides": {}}})
    assert storage.get_role_bindings() == {"coding": {"model": "m", "overrides": {}}}
```

- [ ] **Step 3: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_model_roles.py -q -p no:cacheprovider
```
Expected: collection error — `cannot import name 'model_roles' from 'core.inference'`.

- [ ] **Step 4: Create the storage module**

Create `studio/backend/core/inference/model_roles/storage.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where role bindings live: one app setting, read and written through the
existing settings helpers rather than a new table.

The `_read_setting` / `_write_setting` indirections exist so tests can swap the
storage without a database.
"""

from __future__ import annotations

from typing import Any

ROLES_SETTING_KEY = "model_roles"


def _read_setting() -> Any:
    from storage.studio_db import get_app_setting

    return get_app_setting(ROLES_SETTING_KEY, None)


def _write_setting(value: dict) -> None:
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({ROLES_SETTING_KEY: value})


def get_role_bindings() -> dict[str, dict]:
    """The raw stored mapping. Never raises; a corrupt setting reads as empty."""
    try:
        raw = _read_setting()
    except BaseException:  # noqa: BLE001 - a settings read must not break a turn
        return {}
    return raw if isinstance(raw, dict) else {}


def set_role_bindings(raw: dict) -> dict[str, dict]:
    """Replace the stored mapping. Returns what was stored."""
    value = raw if isinstance(raw, dict) else {}
    _write_setting(value)
    return value
```

- [ ] **Step 5: Create the roles module**

Create `studio/backend/core/inference/model_roles/__init__.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Roles: a name like "coding" bound to a concrete model.

Availability reuses tool readiness's three states on purpose. A role nobody
bound is "unknown", never "ready" -- the same rule the rest of this fork's
discovery work rests on, so a later piece can list models beside tools instead
of maintaining a second vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.inference.tool_readiness import MISSING, READY, UNKNOWN, Readiness

# Offered by the API as a starting point. Nothing behavioural is keyed to these;
# any custom role name works.
DEFAULT_ROLE_NAMES = ("primary", "coding", "vision", "image", "video")


@dataclass(frozen = True)
class RoleBinding:
    role: str
    model: str
    overrides: dict = field(default_factory = dict)


def normalize_role(name) -> str:
    return str(name).strip().lower()


def _resolve_local(model_id: str):
    """(load_path, gguf_variant, loader_id) when the model is downloaded, else None.

    Delegates to the resolver auto-switch uses, so "downloaded" means the same
    thing here as it does when a request names a model."""
    from core.inference.local_model_resolver import resolve_local_gguf

    return resolve_local_gguf(model_id)


def bindings() -> dict[str, RoleBinding]:
    from core.inference.model_roles import storage

    out: dict[str, RoleBinding] = {}
    for name, raw in (storage.get_role_bindings() or {}).items():
        if not isinstance(raw, dict):
            continue
        model = raw.get("model")
        if not isinstance(model, str) or not model.strip():
            continue
        overrides = raw.get("overrides")
        out[normalize_role(name)] = RoleBinding(
            role = normalize_role(name),
            model = model.strip(),
            overrides = overrides if isinstance(overrides, dict) else {},
        )
    return out


def resolve(role) -> Optional[RoleBinding]:
    return bindings().get(normalize_role(role))


def availability(role) -> Readiness:
    """Readiness for a role. Never raises."""
    binding = resolve(role)
    if binding is None:
        return Readiness(UNKNOWN, f"no model is bound to the role {normalize_role(role)!r}")
    try:
        found = _resolve_local(binding.model)
    except BaseException as exc:  # noqa: BLE001 - a role check must not break a turn
        return Readiness(UNKNOWN, f"could not check {binding.model}: {exc}")
    if found:
        return Readiness(READY, f"{binding.model} is downloaded")
    return Readiness(
        MISSING,
        f"{binding.model} is bound to this role but is not downloaded",
        missing = binding.model,
        remedy = "download the model, or bind the role to one you have",
    )
```

- [ ] **Step 6: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_model_roles.py -q -p no:cacheprovider
```
Expected: 12 passed.

- [ ] **Step 7: Commit**

```bash
git add studio/backend/core/inference/model_roles/ studio/backend/tests/test_model_roles.py
git commit -m "feat(roles): bind a model to a role

One app setting holds role -> model plus optional launch overrides, in the same
shape openai_api_auto_switch_overrides already uses, so a role's overrides can
go through the existing model_override_load_kwargs rather than a second path.

Availability reuses tool readiness's three states: an unbound role is unknown,
never ready. 'Downloaded' means what it means to auto-switch, because the check
delegates to the same resolve_local_gguf.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 8: Demonstrate the controls (after committing)**

Restore each with `git checkout -- <file>` immediately after its run.

1. In `availability`, make the `binding is None` branch return `Readiness(READY, "assumed")`. Expect `test_an_unbound_role_is_unknown_not_ready` to FAIL.
2. In `availability`, make the `found` check return MISSING unconditionally. Expect `test_a_bound_downloaded_model_is_ready` to FAIL.
3. In `bindings`, drop the `isinstance(raw, dict)` guard. Expect `test_a_malformed_binding_is_ignored_rather_than_raising` to FAIL.

Finish with `git status --porcelain` — no modified tracked files.

---

### Task 2: The roles API

**Files:**
- Create: `studio/backend/routes/model_roles.py`
- Test: `studio/backend/tests/test_model_roles_api.py`

**Interfaces:**
- Consumes: `model_roles.bindings()`, `availability()`, `DEFAULT_ROLE_NAMES`; `model_roles.storage.set_role_bindings`
- Produces: `routes.model_roles.router` — `GET /api/model-roles`, `PUT /api/model-roles`

The router is registered in `main.py` in Task 8, with the other registration work.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_model_roles_api.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The roles API.

Auth is asserted by overriding the dependency, never by monkeypatching the
function: Depends() captures the callable at import time, so a monkeypatch would
leave the real dependency in place and the test would prove nothing.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.inference import model_roles
from core.inference.model_roles import storage
from auth.authentication import get_current_subject
from routes import model_roles as roles_route


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    saved: dict = {}
    monkeypatch.setattr(storage, "get_role_bindings", lambda: dict(saved))
    monkeypatch.setattr(storage, "set_role_bindings", lambda raw: saved.update(raw) or dict(saved))
    monkeypatch.setattr(model_roles, "_resolve_local", lambda model: None)
    app = FastAPI()
    app.include_router(roles_route.router, prefix = "/api/model-roles")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    return TestClient(app)


def test_get_reports_bindings_and_availability(client):
    client.put("/api/model-roles", json = {"roles": {"coding": {"model": "m"}}})
    body = client.get("/api/model-roles").json()
    assert body["roles"]["coding"]["model"] == "m"
    assert body["roles"]["coding"]["state"] == "missing", "not downloaded in this fixture"
    assert "primary" in body["defaults"]


def test_put_replaces_bindings(client):
    assert client.put("/api/model-roles", json = {"roles": {"coding": {"model": "m"}}}).status_code == 200
    assert client.get("/api/model-roles").json()["roles"]["coding"]["model"] == "m"


def test_put_rejects_a_binding_without_a_model(client):
    response = client.put("/api/model-roles", json = {"roles": {"coding": {}}})
    assert response.status_code == 400
    assert "model" in response.json()["detail"].lower()


def test_put_rejects_a_non_object_roles_payload(client):
    assert client.put("/api/model-roles", json = {"roles": []}).status_code == 400


def test_both_endpoints_require_authentication(monkeypatch, tmp_path):
    """Control: with no dependency override the real auth dependency runs, and an
    unauthenticated request must not reach the handler."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    app = FastAPI()
    app.include_router(roles_route.router, prefix = "/api/model-roles")
    unauthenticated = TestClient(app)
    assert unauthenticated.get("/api/model-roles").status_code in (401, 403)
    assert unauthenticated.put("/api/model-roles", json = {"roles": {}}).status_code in (401, 403)
```

- [ ] **Step 2: Confirm the auth dependency's import path**

The test imports `get_current_subject` from `utils.auth`. Confirm that is where `routes/tool_audit.py` imports it from:

```
grep -n "get_current_subject" studio/backend/routes/tool_audit.py
```
Use whatever path that file uses, in both the router and the test. If it differs, fix both before running.

- [ ] **Step 3: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_model_roles_api.py -q -p no:cacheprovider
```
Expected: collection error — `cannot import name 'model_roles' from 'routes'`.

- [ ] **Step 4: Create the router**

Create `studio/backend/routes/model_roles.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read and replace role bindings.

Follows routes/tool_audit.py: a fork-owned router, every endpoint gated by
get_current_subject. Bindings name models and launch settings, so this is not
public.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth.authentication import get_current_subject

router = APIRouter()


class RolesPayload(BaseModel):
    roles: Any


# Paths are relative: main.py mounts this router at prefix "/api/model-roles",
# the way it mounts routes/tool_audit.py at "/api/tool-audit".
@router.get("")
def read_model_roles(current_subject: str = Depends(get_current_subject)):
    from core.inference import model_roles

    out = {}
    for name, binding in model_roles.bindings().items():
        state = model_roles.availability(name)
        out[name] = {
            "model": binding.model,
            "overrides": binding.overrides,
            "state": state.state,
            "detail": state.detail,
        }
    return {"roles": out, "defaults": list(model_roles.DEFAULT_ROLE_NAMES)}


@router.put("")
def write_model_roles(
    payload: RolesPayload,
    current_subject: str = Depends(get_current_subject),
):
    from core.inference import model_roles
    from core.inference.model_roles import storage

    roles = payload.roles
    if not isinstance(roles, dict):
        raise HTTPException(status_code = 400, detail = "roles must be an object")
    cleaned: dict[str, dict] = {}
    for name, raw in roles.items():
        role = model_roles.normalize_role(name)
        if not role:
            raise HTTPException(status_code = 400, detail = "a role name cannot be empty")
        if not isinstance(raw, dict):
            raise HTTPException(status_code = 400, detail = f"{role}: binding must be an object")
        model = raw.get("model")
        if not isinstance(model, str) or not model.strip():
            raise HTTPException(status_code = 400, detail = f"{role}: a model is required")
        overrides = raw.get("overrides")
        if overrides is not None and not isinstance(overrides, dict):
            raise HTTPException(status_code = 400, detail = f"{role}: overrides must be an object")
        cleaned[role] = {"model": model.strip(), "overrides": overrides or {}}
    storage.set_role_bindings(cleaned)
    return read_model_roles(current_subject = current_subject)
```

- [ ] **Step 5: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_model_roles_api.py tests/test_model_roles.py -q -p no:cacheprovider
```
Expected: 17 passed.

- [ ] **Step 6: Commit**

```bash
git add studio/backend/routes/model_roles.py studio/backend/tests/test_model_roles_api.py
git commit -m "feat(roles): auth-gated GET/PUT /api/model-roles

Follows routes/tool_audit.py -- fork-owned router, get_current_subject on every
endpoint. Bindings name models and launch settings, so they are not public.

The auth test overrides the dependency rather than monkeypatching it: Depends()
captures the callable at import time, so a monkeypatch leaves the real
dependency in place and proves nothing.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Demonstrate the controls (after committing)**

1. Remove `Depends(get_current_subject)` from both endpoints. Expect `test_both_endpoints_require_authentication` to FAIL.
2. Drop the `isinstance(model, str)` check in the PUT. Expect `test_put_rejects_a_binding_without_a_model` to FAIL.

Restore with `git checkout --` after each; finish with a clean `git status --porcelain`.

---

### Task 3: The delegation workspace

**Files:**
- Create: `studio/backend/core/inference/delegation/__init__.py` (empty package marker for now — a docstring only), `studio/backend/core/inference/delegation/workspace.py`
- Test: `studio/backend/tests/test_delegation_workspace.py`

**Interfaces:**
- Consumes: `core.inference.tools._get_workdir` (lazily imported — `tools.py` imports the delegation schemas at module level in Task 8, so a module-level import here would be a cycle)
- Produces: `DelegationFiles(delegation_id, root, brief, work, transcript)`, `create(session_id, role) -> DelegationFiles`, `write_brief(files, *, role, task, context) -> None`, `write_work(files, text) -> None`, `append_transcript(files, text) -> None`, `relative_paths(files) -> dict[str, str]`

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_delegation_workspace.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where a delegation's files live.

The location is forced, not chosen: assist_vision's _write_png records that
tools return a path and that resolve_image_bytes refuses any path outside the
conversation's working directory. A delegation folder written anywhere else
could not be read back by the primary's own file tools.
"""

from __future__ import annotations

import os

import pytest

from core.inference.delegation import workspace


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "_workdir_for", lambda session_id: str(tmp_path))
    return tmp_path


def test_a_delegation_folder_is_created_under_the_session_workdir(workdir):
    files = workspace.create("session-1", "coding")
    assert os.path.isdir(files.root)
    assert str(workdir) in files.root
    assert "delegations" in files.root


def test_two_delegations_do_not_collide(workdir):
    first = workspace.create("session-1", "coding")
    second = workspace.create("session-1", "coding")
    assert first.root != second.root
    assert first.delegation_id != second.delegation_id


def test_the_brief_records_the_role_task_and_context(workdir):
    files = workspace.create("session-1", "coding")
    workspace.write_brief(files, role = "coding", task = "rewrite the parser", context = "it is recursive descent")
    text = open(files.brief, encoding = "utf-8").read()
    assert "coding" in text
    assert "rewrite the parser" in text
    assert "recursive descent" in text


def test_the_work_file_and_transcript_are_written(workdir):
    files = workspace.create("session-1", "coding")
    workspace.write_work(files, "def parse(): ...")
    workspace.append_transcript(files, "round 1")
    workspace.append_transcript(files, "round 2")
    assert "def parse()" in open(files.work, encoding = "utf-8").read()
    transcript = open(files.transcript, encoding = "utf-8").read()
    assert "round 1" in transcript and "round 2" in transcript, "transcript appends, never truncates"


def test_paths_returned_to_the_model_are_relative_to_the_workdir(workdir):
    files = workspace.create("session-1", "coding")
    paths = workspace.relative_paths(files)
    for value in paths.values():
        assert not os.path.isabs(value), f"{value} is absolute; the model gets workdir-relative paths"
        assert value.startswith("delegations/")


def test_writing_never_raises_on_an_unwritable_root(workdir, monkeypatch):
    files = workspace.create("session-1", "coding")
    monkeypatch.setattr(workspace, "_write_text", lambda path, text, append = False: (_ for _ in ()).throw(OSError("read-only")))
    workspace.write_work(files, "x")
    workspace.append_transcript(files, "y")


def test_the_real_workdir_helper_is_the_tools_one():
    """Delegation must not invent a second workspace rule."""
    import inspect

    assert "_get_workdir" in inspect.getsource(workspace._workdir_for)
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_delegation_workspace.py -q -p no:cacheprovider
```
Expected: collection error — no module `core.inference.delegation`.

- [ ] **Step 3: Create the package marker**

Create `studio/backend/core/inference/delegation/__init__.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hand a task to another role's model, then come back.

The orchestration lives here from Task 6; this module starts as a package marker
so the workspace and sub-agent pieces can be built and reviewed on their own.
"""
```

- [ ] **Step 4: Create the workspace module**

Create `studio/backend/core/inference/delegation/workspace.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A delegation's files, under the conversation's own working directory.

That location is forced, not chosen. assist_vision/__init__.py::_write_png
records the rule: tools return a path, and resolve_image_bytes refuses any path
outside the conversation's working directory -- so a delegation folder written
elsewhere could not be read back by the primary's own file tools.

Every write is guarded: losing a transcript must not break a delegation.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass

_FOLDER = "delegations"


@dataclass(frozen = True)
class DelegationFiles:
    delegation_id: str
    root: str
    brief: str
    work: str
    transcript: str


def _workdir_for(session_id) -> str:
    """The conversation's working directory, from tools.py's own helper.

    Imported lazily: tools.py imports the delegation schemas at module level, so
    a module-level import back would be a cycle."""
    from core.inference.tools import _get_workdir

    return _get_workdir(session_id)


def _write_text(path: str, text: str, append: bool = False) -> None:
    with open(path, "a" if append else "w", encoding = "utf-8") as handle:
        handle.write(text)


def create(session_id, role) -> DelegationFiles:
    delegation_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    root = os.path.join(_workdir_for(session_id), _FOLDER, delegation_id)
    os.makedirs(root, exist_ok = True)
    return DelegationFiles(
        delegation_id = delegation_id,
        root = root,
        brief = os.path.join(root, "brief.md"),
        work = os.path.join(root, "work.md"),
        transcript = os.path.join(root, "transcript.md"),
    )


def write_brief(files: DelegationFiles, *, role, task, context = "") -> None:
    body = (
        f"# Delegation {files.delegation_id}\n\n"
        f"**Role:** {role}\n\n"
        f"## Task\n\n{task}\n\n"
        f"## What the primary already established\n\n{context or '(nothing supplied)'}\n\n"
        f"## Done means\n\n"
        f"The answer is written to `work.md`, and the short reply says what was done.\n"
    )
    try:
        _write_text(files.brief, body)
    except BaseException:  # noqa: BLE001 - a delegation must not die writing a file
        pass


def write_work(files: DelegationFiles, text: str) -> None:
    try:
        _write_text(files.work, text)
    except BaseException:  # noqa: BLE001
        pass


def append_transcript(files: DelegationFiles, text: str) -> None:
    try:
        _write_text(files.transcript, text.rstrip() + "\n\n", append = True)
    except BaseException:  # noqa: BLE001
        pass


def relative_paths(files: DelegationFiles) -> dict[str, str]:
    """Workdir-relative paths, which is what the model can act on."""
    base = f"{_FOLDER}/{files.delegation_id}"
    return {"brief": f"{base}/brief.md", "work": f"{base}/work.md", "transcript": f"{base}/transcript.md"}
```

- [ ] **Step 5: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_delegation_workspace.py -q -p no:cacheprovider
```
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add studio/backend/core/inference/delegation/ studio/backend/tests/test_delegation_workspace.py
git commit -m "feat(delegation): the brief, work file and transcript

All three live under the conversation's own working directory, because
assist_vision's _write_png records the rule: tools return a path, and
resolve_image_bytes refuses any path outside that directory -- so a folder
written elsewhere could not be read back by the primary's file tools.

The workdir helper is tools.py's own _get_workdir, imported lazily (tools.py
imports the delegation schemas at module level in a later task). Every write is
guarded: losing a transcript must not break a delegation.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Demonstrate the controls (after committing)**

1. Make `relative_paths` return `files.brief` (an absolute path). Expect `test_paths_returned_to_the_model_are_relative_to_the_workdir` to FAIL.
2. Remove the guard from `append_transcript`. Expect `test_writing_never_raises_on_an_unwritable_root` to FAIL.
3. Change `_workdir_for` to `tempfile.gettempdir()`. Expect `test_the_real_workdir_helper_is_the_tools_one` to FAIL.

Restore after each; finish with a clean `git status --porcelain`.

---

### Task 4: The loader adapter

**Files:**
- Create: `studio/backend/core/inference/delegation/loader.py`
- Test: `studio/backend/tests/test_delegation_loader.py`

**Interfaces:**
- Consumes: `routes.inference._load_model_impl` and `LoadRequest` (lazily, inside the call); `core.inference.gpu_arbiter`
- Produces: `resident_model_id() -> Optional[str]`, `load(model_id, overrides) -> None`, `LoaderError`

**This is the only module that touches the real loader.** Everything else takes it as an injected callable, so no other test comes near a model.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_delegation_loader.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The swap adapter.

No test here loads a model. What IS tested is the wiring: that the adapter reads
the resident model from the backend rather than guessing, that it reuses the
route's own loader instead of re-implementing it, and that a load failure
surfaces as LoaderError rather than an arbitrary exception type.

Re-implementing the loader is the failure this project keeps paying for -- the
route's _load_model_impl is ~450 lines of arbiter, load intent, config and
draft-model handling.
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest

from core.inference.delegation import loader


def _route_module(monkeypatch, backend = None, loader_calls = None):
    module = types.ModuleType("routes.inference")
    module._llama_cpp_backend = backend

    async def _load_model_impl(request, fastapi_request, current_subject, **kwargs):
        if loader_calls is not None:
            loader_calls.append((request, fastapi_request, current_subject))

    module._load_model_impl = _load_model_impl
    monkeypatch.setitem(sys.modules, "routes.inference", module)
    return module


def test_the_resident_model_is_read_from_the_backend(monkeypatch):
    backend = types.SimpleNamespace(is_loaded = True, model_identifier = "repo/Model:Q4")
    _route_module(monkeypatch, backend = backend)
    assert loader.resident_model_id() == "repo/Model:Q4"


def test_nothing_loaded_reads_as_none(monkeypatch):
    backend = types.SimpleNamespace(is_loaded = False, model_identifier = "stale")
    _route_module(monkeypatch, backend = backend)
    assert loader.resident_model_id() is None


def test_no_route_module_reads_as_none(monkeypatch):
    monkeypatch.delitem(sys.modules, "routes.inference", raising = False)
    assert loader.resident_model_id() is None


def test_load_delegates_to_the_routes_own_loader(monkeypatch):
    calls = []
    _route_module(monkeypatch, loader_calls = calls)
    monkeypatch.setattr(loader, "_run_coroutine", lambda coro: __import__("asyncio").run(coro))
    loader.load("repo/Model:Q4", {})
    assert len(calls) == 1, "the route's loader must be what runs"
    request, fastapi_request, subject = calls[0]
    assert fastapi_request is None, "no Request is available inside a tool call"
    assert isinstance(subject, str) and subject
    assert request.model_path == "repo/Model:Q4", "LoadRequest's field is model_path"


def test_a_load_failure_becomes_LoaderError(monkeypatch):
    module = _route_module(monkeypatch)

    async def _boom(request, fastapi_request, current_subject, **kwargs):
        raise RuntimeError("no VRAM")

    module._load_model_impl = _boom
    monkeypatch.setattr(loader, "_run_coroutine", lambda coro: __import__("asyncio").run(coro))
    with pytest.raises(loader.LoaderError) as excinfo:
        loader.load("repo/Model:Q4", {})
    assert "no VRAM" in str(excinfo.value)


def test_the_adapter_does_not_reimplement_loading():
    """Control against drift: the adapter must call the route's loader, not
    assemble its own from the arbiter and backend."""
    source = inspect.getsource(loader)
    assert "_load_model_impl" in source
    assert "acquire_for" not in source, "the route's loader already arbitrates; do not do it twice"
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_delegation_loader.py -q -p no:cacheprovider
```
Expected: collection error — no module `core.inference.delegation.loader`.

- [ ] **Step 3: Create the adapter**

Create `studio/backend/core/inference/delegation/loader.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The one place that swaps the chat model.

It calls the ROUTE's own _load_model_impl rather than assembling a load from the
arbiter and the backend. That function is ~450 lines of GPU arbitration, load
intent, config resolution and draft-model handling; a second implementation
would drift from it, which is the failure this fork has already paid for once in
its readiness probes.

Two things it does NOT have, and does not need:
  * a FastAPI Request -- _load_model_impl uses it only for
    _request_used_api_key (whose own comment says it is "Total by construction"
    and "must never fail a load") and _resolve_parallel_slots (a getattr chain
    defaulting to one slot, which is right for a single delegate).
  * a user subject -- an internal label is passed instead.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

DELEGATION_SUBJECT = "delegation"


class LoaderError(RuntimeError):
    """A swap failed. Raised so the caller can report it rather than guess."""


def resident_model_id() -> Optional[str]:
    """What is loaded right now, read from the backend, or None.

    Read through sys.modules: core code must not import the route module."""
    module = sys.modules.get("routes.inference")
    backend = getattr(module, "_llama_cpp_backend", None) if module is not None else None
    if backend is None or not getattr(backend, "is_loaded", False):
        return None
    return getattr(backend, "model_identifier", None) or None


def _run_coroutine(coro):
    """Run a coroutine from this (synchronous) tool thread."""
    return asyncio.run(coro)


def load(model_id: str, overrides: Optional[dict] = None) -> None:
    """Load ``model_id``. Raises LoaderError on any failure."""
    module = sys.modules.get("routes.inference")
    if module is None:
        raise LoaderError("the inference route is not loaded in this process")
    impl = getattr(module, "_load_model_impl", None)
    if impl is None:
        raise LoaderError("the inference route does not expose a loader")
    try:
        from models.inference import LoadRequest

        # model_path, not model_name: the field is "Model identifier or local path".
        request = LoadRequest(model_path = model_id, **(overrides or {}))
        _run_coroutine(impl(request, None, DELEGATION_SUBJECT))
    except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
        raise LoaderError(f"could not load {model_id}: {exc}") from exc
```

- [ ] **Step 5: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_delegation_loader.py -q -p no:cacheprovider
```
Expected: 6 passed.

- [ ] **Step 6: Manual verification — the one thing tests cannot cover**

The spec records this as owed. With Unslothed running and a small model downloaded, from `C:\Users\Admin\unsloth\studio\backend`:

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -c "
import time, routes.inference
from core.inference.delegation import loader
print('resident before:', loader.resident_model_id())
"
```

This confirms `resident_model_id()` reads the real backend. **Do not attempt a real `load()` from a standalone process** — the loader belongs to the running server's event loop. Record in your report what `resident_model_id()` returned, and note that a real swap is exercised for the first time in Task 6's manual step.

- [ ] **Step 7: Commit**

```bash
git add studio/backend/core/inference/delegation/loader.py studio/backend/tests/test_delegation_loader.py
git commit -m "feat(delegation): the swap adapter

The only module that touches the real loader. It calls the route's own
_load_model_impl instead of assembling a load from the arbiter and backend --
that function is ~450 lines of arbitration, load intent, config and draft-model
handling, and a second implementation would drift from it.

It passes None for the Request and an internal subject label, which the route
tolerates by construction: _request_used_api_key is documented as 'Total by
construction' and 'must never fail a load', and _resolve_parallel_slots is a
getattr chain defaulting to one slot.

A control asserts the adapter does not re-implement arbitration.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 8: Demonstrate the controls (after committing)**

1. Add `from core.inference.gpu_arbiter import acquire_for` and a call to it in `load`. Expect `test_the_adapter_does_not_reimplement_loading` to FAIL.
2. Make `resident_model_id` ignore `is_loaded`. Expect `test_nothing_loaded_reads_as_none` to FAIL.
3. Let the exception in `load` propagate instead of wrapping it. Expect `test_a_load_failure_becomes_LoaderError` to FAIL.

Restore after each; finish with a clean `git status --porcelain`.

---

### Task 5: The sub-agent loop

**Files:**
- Create: `studio/backend/core/inference/delegation/subagent.py`
- Test: `studio/backend/tests/test_delegation_subagent.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (deliberately model-agnostic)
- Produces:
  - `DELEGATE_MAX_ITERATIONS = 12`, `DELEGATE_TIMEOUT_S = 600.0`
  - `SubagentResult(text, rounds, stopped_because)`
  - `run_subagent(*, messages, tools, call_model, execute_tool, on_round = None, max_iterations = DELEGATE_MAX_ITERATIONS, timeout_s = DELEGATE_TIMEOUT_S, cancel_event = None, now = time.monotonic) -> SubagentResult`

`call_model(messages, tools) -> dict` returns an OpenAI-style assistant message: `{"content": str | None, "tool_calls": [{"id", "function": {"name", "arguments"}}]}`. Keeping the model call injected is what lets this be tested without a backend, and what lets a GGUF or safetensors delegate use the same loop.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_delegation_subagent.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The bounded sub-agent loop.

The delegate is a real agent with real tools, so every bound here is
load-bearing: without them a delegate can loop forever inside one tool call of
the primary's turn.
"""

from __future__ import annotations

import threading

import pytest

from core.inference.delegation import subagent


def _assistant(text = None, calls = ()):
    return {
        "content": text,
        "tool_calls": [
            {"id": f"c{i}", "function": {"name": name, "arguments": args}}
            for i, (name, args) in enumerate(calls)
        ],
    }


def _script(*turns):
    """call_model returns each turn in order, then repeats the last."""
    seen = []

    def call_model(messages, tools):
        seen.append(list(messages))
        return turns[min(len(seen) - 1, len(turns) - 1)]

    call_model.seen = seen
    return call_model


def test_a_plain_answer_ends_the_loop():
    result = subagent.run_subagent(
        messages = [{"role": "user", "content": "hi"}],
        tools = [],
        call_model = _script(_assistant("done")),
        execute_tool = lambda name, arguments, **kw: "",
    )
    assert result.text == "done"
    assert result.rounds == 1
    assert result.stopped_because == "answered"


def test_tool_results_are_fed_back_and_the_loop_continues():
    executed = []

    def execute_tool(name, arguments, **kwargs):
        executed.append((name, arguments))
        return "tool said hello"

    call_model = _script(_assistant(calls = [("terminal", '{"command": "ls"}')]), _assistant("finished"))
    result = subagent.run_subagent(
        messages = [{"role": "user", "content": "go"}],
        tools = [{"function": {"name": "terminal"}}],
        call_model = call_model,
        execute_tool = execute_tool,
    )
    assert executed == [("terminal", {"command": "ls"})], "arguments are parsed from JSON"
    assert result.text == "finished"
    assert result.rounds == 2
    assert any(m.get("role") == "tool" and "tool said hello" in str(m.get("content"))
               for m in call_model.seen[-1]), "the tool result must reach the next turn"


def test_the_iteration_cap_stops_a_runaway_delegate():
    call_model = _script(_assistant(calls = [("terminal", "{}")]))
    result = subagent.run_subagent(
        messages = [], tools = [], call_model = call_model,
        execute_tool = lambda name, arguments, **kw: "again",
        max_iterations = 3,
    )
    assert result.rounds == 3
    assert result.stopped_because == "iteration-cap"


def test_the_timeout_stops_a_slow_delegate():
    clock = {"t": 0.0}

    def now():
        clock["t"] += 100.0
        return clock["t"]

    result = subagent.run_subagent(
        messages = [], tools = [],
        call_model = _script(_assistant(calls = [("terminal", "{}")])),
        execute_tool = lambda name, arguments, **kw: "again",
        max_iterations = 50, timeout_s = 250.0, now = now,
    )
    assert result.stopped_because == "timeout"
    assert result.rounds < 50


def test_cancellation_stops_the_loop():
    cancel = threading.Event()

    def execute_tool(name, arguments, **kwargs):
        cancel.set()
        return "x"

    result = subagent.run_subagent(
        messages = [], tools = [],
        call_model = _script(_assistant(calls = [("terminal", "{}")])),
        execute_tool = execute_tool, cancel_event = cancel,
    )
    assert result.stopped_because == "cancelled"


def test_a_tool_that_raises_is_reported_to_the_delegate_not_propagated():
    def execute_tool(name, arguments, **kwargs):
        raise OverflowError("boom")

    result = subagent.run_subagent(
        messages = [], tools = [],
        call_model = _script(_assistant(calls = [("terminal", "{}")]), _assistant("recovered")),
        execute_tool = execute_tool,
    )
    assert result.text == "recovered"


def test_malformed_tool_arguments_do_not_crash_the_loop():
    seen = []
    result = subagent.run_subagent(
        messages = [], tools = [],
        call_model = _script(_assistant(calls = [("terminal", "{not json")]), _assistant("ok")),
        execute_tool = lambda name, arguments, **kw: seen.append(arguments) or "",
    )
    assert seen == [{}], "unparseable arguments become an empty dict"
    assert result.text == "ok"


def test_each_round_is_reported_for_the_transcript():
    rounds = []
    subagent.run_subagent(
        messages = [], tools = [],
        call_model = _script(_assistant(calls = [("terminal", "{}")]), _assistant("done")),
        execute_tool = lambda name, arguments, **kw: "result",
        on_round = rounds.append,
    )
    joined = "\n".join(rounds)
    assert "terminal" in joined and "done" in joined
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_delegation_subagent.py -q -p no:cacheprovider
```
Expected: collection error — no module `core.inference.delegation.subagent`.

- [ ] **Step 3: Create the loop**

Create `studio/backend/core/inference/delegation/subagent.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A bounded agent loop for a delegate model.

The model call and the tool executor are injected, exactly as
safetensors_agentic.run_safetensors_tool_loop injects its own. That is what lets
this be tested without a backend, and what lets a GGUF delegate (structured tool
calls) and a safetensors delegate (markup) share one loop.

Every bound here is load-bearing. The delegate runs INSIDE one tool call of the
primary's turn, so an unbounded loop is an unbounded turn.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

DELEGATE_MAX_ITERATIONS = 12
DELEGATE_TIMEOUT_S = 600.0


@dataclass(frozen = True)
class SubagentResult:
    text: str
    rounds: int
    stopped_because: str  # "answered" | "iteration-cap" | "timeout" | "cancelled"


def _arguments(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except BaseException:  # noqa: BLE001 - a malformed call is the model's error, not a crash
        return {}
    return parsed if isinstance(parsed, dict) else {}


def run_subagent(
    *,
    messages: list,
    tools: list,
    call_model: Callable[[list, list], dict],
    execute_tool: Callable[..., str],
    on_round: Optional[Callable[[str], None]] = None,
    max_iterations: int = DELEGATE_MAX_ITERATIONS,
    timeout_s: float = DELEGATE_TIMEOUT_S,
    cancel_event = None,
    now: Callable[[], float] = time.monotonic,
    **tool_kwargs,
) -> SubagentResult:
    history = list(messages)
    started = now()
    rounds = 0
    text = ""
    while True:
        if cancel_event is not None and cancel_event.is_set():
            return SubagentResult(text, rounds, "cancelled")
        if rounds >= max_iterations:
            return SubagentResult(text, rounds, "iteration-cap")
        if now() - started >= timeout_s:
            return SubagentResult(text, rounds, "timeout")

        message = call_model(history, tools)
        rounds += 1
        text = (message or {}).get("content") or text
        calls = (message or {}).get("tool_calls") or []
        if on_round is not None:
            names = ", ".join(c.get("function", {}).get("name", "?") for c in calls)
            on_round(f"round {rounds}: {message.get('content') or ''}{(' -> ' + names) if names else ''}")
        if not calls:
            return SubagentResult(text, rounds, "answered")

        history.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name") or ""
            arguments = _arguments(function.get("arguments"))
            try:
                result = execute_tool(name, arguments, **tool_kwargs)
            except BaseException as exc:  # noqa: BLE001 - reported to the delegate, never propagated
                result = f"Error: {name} failed: {exc}"
            history.append({"role": "tool", "tool_call_id": call.get("id"), "content": str(result)})
            if on_round is not None:
                on_round(f"  {name} -> {str(result)[:400]}")
            if cancel_event is not None and cancel_event.is_set():
                return SubagentResult(text, rounds, "cancelled")
```

- [ ] **Step 4: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_delegation_subagent.py -q -p no:cacheprovider
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add studio/backend/core/inference/delegation/subagent.py studio/backend/tests/test_delegation_subagent.py
git commit -m "feat(delegation): a bounded sub-agent loop

The model call and tool executor are injected, the same shape
run_safetensors_tool_loop already uses -- so this is testable without a backend
and one loop serves both a GGUF delegate (structured tool calls) and a
safetensors one.

Every bound is load-bearing because the delegate runs INSIDE one tool call of
the primary's turn: iteration cap, wall clock, and a cancel check both before
each round and after each tool. A tool that raises is reported to the delegate
rather than propagated, and unparseable arguments become an empty dict.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Demonstrate the controls (after committing)**

1. Remove the `rounds >= max_iterations` check. Expect `test_the_iteration_cap_stops_a_runaway_delegate` to FAIL (it will hang briefly then fail on the round count — if it hangs, that itself proves the point; interrupt and restore).
2. Remove the timeout check. Expect `test_the_timeout_stops_a_slow_delegate` to FAIL.
3. Remove the post-tool cancel check. Expect `test_cancellation_stops_the_loop` to FAIL.
4. Let the tool exception propagate. Expect `test_a_tool_that_raises_is_reported_to_the_delegate_not_propagated` to FAIL.

Restore after each; finish with a clean `git status --porcelain`.

---

### Task 6: The `ask_model` tool

**Files:**
- Create: `studio/backend/core/inference/delegation/schemas.py`
- Modify: `studio/backend/core/inference/delegation/__init__.py`
- Test: `studio/backend/tests/test_delegation_tool.py`

**Interfaces:**
- Consumes: `model_roles.availability` / `resolve` (Task 1); `workspace` (Task 3); `loader` (Task 4); `subagent.run_subagent` (Task 5)
- Produces:
  - `delegation.schemas`: `ASK_MODEL_TOOL`, `DELEGATION_TOOLS`, `DELEGATION_TOOL_NAMES`
  - `delegation`: `execute(name, arguments, **kwargs) -> str`, `MAX_DELEGATIONS_PER_WINDOW = 2`, `DELEGATION_WINDOW_S = 600.0`, `active_delegation_id() -> Optional[str]`, and the injected seams `_loader`, `_call_model_for`

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_delegation_tool.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ask_model: swap, run, swap back.

The load-bearing rule is step 3 of the sequence: what gets restored is the model
that was RESIDENT when the call started, not whatever the "primary" role names.
Conflating them would silently change which model the user is talking to.

No test loads a model: the loader is injected.
"""

from __future__ import annotations

import threading

import pytest

from core.inference import delegation
from core.inference import model_roles
from core.inference.delegation import subagent, workspace
from core.inference.model_roles import storage
from core.inference.tool_readiness import MISSING, READY, UNKNOWN, Readiness


class FakeLoader:
    def __init__(self, resident = "repo/Primary:Q4", fail_on = None):
        self.resident = resident
        self.fail_on = fail_on
        self.calls = []

    def resident_model_id(self):
        return self.resident

    def load(self, model_id, overrides = None):
        self.calls.append(model_id)
        if self.fail_on is not None and model_id == self.fail_on:
            raise delegation._loader_error("could not load " + model_id)
        self.resident = model_id


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "_workdir_for", lambda session_id: str(tmp_path))
    monkeypatch.setattr(storage, "get_role_bindings",
                        lambda: {"coding": {"model": "repo/Coder:Q4"}, "primary": {"model": "repo/Other:Q4"}})
    monkeypatch.setattr(model_roles, "_resolve_local", lambda model: ("/p", None, model))
    fake = FakeLoader()
    monkeypatch.setattr(delegation, "_loader", fake)
    monkeypatch.setattr(delegation, "_call_model_for",
                        lambda binding: (lambda messages, tools: {"content": "delegate answer", "tool_calls": []}))
    delegation._reset_for_tests()
    return fake


def _run(**over):
    arguments = {"role": "coding", "task": "rewrite the parser"}
    arguments.update(over)
    return delegation.execute("ask_model", arguments, session_id = "s1")


def test_a_delegation_swaps_runs_and_restores(rig):
    out = _run()
    assert rig.calls == ["repo/Coder:Q4", "repo/Primary:Q4"], "swap out, then back"
    assert rig.resident == "repo/Primary:Q4"
    assert "delegate answer" in out


def test_the_restore_targets_what_was_resident_not_the_primary_role(rig):
    """THE load-bearing test. The 'primary' role names repo/Other:Q4; the model
    actually loaded was repo/Primary:Q4, and that is what must come back."""
    _run()
    assert rig.calls[-1] == "repo/Primary:Q4"
    assert "repo/Other:Q4" not in rig.calls


def test_the_result_names_the_work_file(rig):
    out = _run()
    assert "delegations/" in out and "work.md" in out


def test_an_unavailable_role_never_loads_anything(rig, monkeypatch):
    monkeypatch.setattr(model_roles, "_resolve_local", lambda model: None)
    out = _run()
    assert rig.calls == [], "no swap may be attempted when the role is not ready"
    assert "not downloaded" in out.lower() or "missing" in out.lower()


def test_an_unbound_role_is_refused(rig):
    out = _run(role = "nonexistent")
    assert rig.calls == []
    assert "nonexistent" in out


def test_the_model_is_restored_even_when_the_delegate_raises(rig, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("sub-agent exploded")

    monkeypatch.setattr(subagent, "run_subagent", boom)
    out = _run()
    assert rig.calls[-1] == "repo/Primary:Q4", "restore must happen in a finally"
    assert "exploded" in out or "failed" in out.lower()


def test_a_failed_restore_is_reported_loudly(rig, monkeypatch):
    rig.fail_on = "repo/Primary:Q4"
    out = _run()
    lowered = out.lower()
    assert "could not" in lowered or "failed" in lowered
    assert "repo/Primary:Q4" in out, "the user must be told which model did not come back"


def test_a_nested_delegation_is_refused(rig):
    """Depth guard: the delegate must not delegate."""
    inner = {}

    def call_model(messages, tools):
        inner["out"] = delegation.execute("ask_model", {"role": "coding", "task": "again"}, session_id = "s1")
        return {"content": "done", "tool_calls": []}

    import types as _types
    object.__setattr__(delegation, "_call_model_for", lambda binding: call_model)
    _run()
    assert "already" in inner["out"].lower() or "nested" in inner["out"].lower()


def test_the_window_cap_refuses_a_third_delegation(rig):
    _run()
    _run()
    out = _run()
    assert "limit" in out.lower() or "too many" in out.lower()
    assert rig.calls.count("repo/Coder:Q4") == 2, "the third must not swap"


def test_cancellation_is_passed_to_the_sub_agent(rig, monkeypatch):
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return subagent.SubagentResult("x", 1, "answered")

    monkeypatch.setattr(subagent, "run_subagent", spy)
    cancel = threading.Event()
    delegation.execute("ask_model", {"role": "coding", "task": "t"}, session_id = "s1", cancel_event = cancel)
    assert seen.get("cancel_event") is cancel


def test_execute_never_raises_on_bad_arguments(rig):
    for arguments in (None, [], {"role": None}, {"role": "coding"}, {"task": "x"}, {"role": 5, "task": 6}):
        out = delegation.execute("ask_model", arguments, session_id = "s1")
        assert isinstance(out, str) and out


def test_the_schema_is_shaped_like_the_other_tools():
    from core.inference.delegation.schemas import (
        ASK_MODEL_TOOL, DELEGATION_TOOLS, DELEGATION_TOOL_NAMES,
    )

    function = ASK_MODEL_TOOL["function"]
    assert function["name"] == "ask_model"
    assert set(function["parameters"]["required"]) == {"role", "task"}
    assert DELEGATION_TOOLS == [ASK_MODEL_TOOL]
    assert DELEGATION_TOOL_NAMES == frozenset({"ask_model"})
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_delegation_tool.py -q -p no:cacheprovider
```
Expected: failures — `delegation` has no `execute`, `_loader`, `_reset_for_tests`.

- [ ] **Step 3: Create the schema**

Create `studio/backend/core/inference/delegation/schemas.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI-style schema for the delegation tool."""

ASK_MODEL_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_model",
        "description": (
            "Hand a self-contained task to another model configured for a role (for example "
            "'coding' or 'vision'), then continue once it is done. The other model is loaded, "
            "works with its own tools, writes its result to a file, and the model you were "
            "using is loaded again. Expensive: it swaps models twice, so use it when the other "
            "model is genuinely better suited, not for small steps. Returns a short summary and "
            "the paths of the files it wrote."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "description": "The configured role to hand the task to, e.g. 'coding'.",
                },
                "task": {
                    "type": "string",
                    "description": (
                        "What to do, written so it stands alone -- the other model cannot see "
                        "this conversation."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": "What you already established that it needs (optional).",
                },
            },
            "required": ["role", "task"],
        },
    },
}

DELEGATION_TOOLS = [ASK_MODEL_TOOL]
DELEGATION_TOOL_NAMES = frozenset({"ask_model"})
```

- [ ] **Step 4: Write the orchestration**

Replace `studio/backend/core/inference/delegation/__init__.py` with:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hand a task to another role's model, then come back.

THE RULE THAT MATTERS: what gets restored is the model that was RESIDENT when
the call started -- never whatever the "primary" role happens to name. If the
user loaded a model by hand, that is what comes back. A delegation must be
invisible to the conversation except for its result.

Two budgets guard the primary's turn, because the delegate runs inside one of
its tool calls: the sub-agent's own caps, and a limit on how often delegation
may happen at all.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from core.inference.delegation import loader as _loader_module
from core.inference.delegation import subagent, workspace
from core.inference.delegation.schemas import DELEGATION_TOOL_NAMES  # noqa: F401 - re-export
from core.inference.tool_readiness import READY

MAX_DELEGATIONS_PER_WINDOW = 2
# The spec says "per assistant turn". A tool cannot observe turn boundaries --
# execute_tool receives a session id, not a turn id -- so this is the closest
# observable approximation, and the refusal says so.
DELEGATION_WINDOW_S = 600.0

_loader = _loader_module
_active = threading.local()
_recent: dict[str, list[float]] = {}
_recent_guard = threading.Lock()


def _loader_error(message: str):
    return _loader_module.LoaderError(message)


def _reset_for_tests() -> None:
    with _recent_guard:
        _recent.clear()
    _active.delegation_id = None


def active_delegation_id() -> Optional[str]:
    return getattr(_active, "delegation_id", None)


def _call_model_for(binding):
    """A callable(messages, tools) -> assistant message for this role's model.

    Separated so tests inject a fake. The real implementation talks to the
    loaded backend's own OpenAI-compatible endpoint (llama_cpp exposes base_url
    and _auth_headers); it is only reachable once the delegate is loaded."""
    from core.inference.delegation.model_call import call_loaded_model

    return call_loaded_model


def _window_ok(session_id: str, now: float) -> bool:
    with _recent_guard:
        stamps = [t for t in _recent.get(session_id, []) if now - t < DELEGATION_WINDOW_S]
        if len(stamps) >= MAX_DELEGATIONS_PER_WINDOW:
            _recent[session_id] = stamps
            return False
        stamps.append(now)
        _recent[session_id] = stamps
        return True


def execute(name: str, arguments, **kwargs) -> str:
    """Handler for ask_model. Never raises."""
    try:
        return _execute(arguments, **kwargs)
    except BaseException as exc:  # noqa: BLE001 - a delegation must not break the turn
        return f"Error: delegation failed: {exc}"


def _execute(arguments, *, session_id = None, cancel_event = None, **kwargs) -> str:
    from core.inference import model_roles

    if active_delegation_id() is not None:
        return (
            "Error: a delegation is already running; the model you delegated to "
            "cannot delegate again. Do this part yourself."
        )

    args = arguments if isinstance(arguments, dict) else {}
    role = str(args.get("role") or "").strip()
    task = str(args.get("task") or "").strip()
    if not role:
        return "Error: ask_model needs a role."
    if not task:
        return "Error: ask_model needs a task, written so it stands alone."

    state = model_roles.availability(role)
    if state.state != READY:
        return f"Error: the role {role!r} is not usable: {state.detail}"
    binding = model_roles.resolve(role)

    if not _window_ok(str(session_id), time.monotonic()):
        return (
            f"Error: already delegated {MAX_DELEGATIONS_PER_WINDOW} times in the last "
            f"{int(DELEGATION_WINDOW_S / 60)} minutes (each swaps models twice). "
            "Continue with the current model."
        )

    resident = _loader.resident_model_id()
    files = workspace.create(session_id, role)
    workspace.write_brief(files, role = role, task = task, context = str(args.get("context") or ""))
    paths = workspace.relative_paths(files)

    _active.delegation_id = files.delegation_id
    try:
        _loader.load(binding.model, binding.overrides)
        result = subagent.run_subagent(
            messages = [
                {"role": "system", "content": f"You are the {role} model. Write your result to {paths['work']}."},
                {"role": "user", "content": open(files.brief, encoding = "utf-8").read()},
            ],
            tools = _delegate_tools(),
            call_model = _call_model_for(binding),
            execute_tool = _delegate_execute_tool(),
            on_round = lambda line: workspace.append_transcript(files, line),
            cancel_event = cancel_event,
            session_id = session_id,
        )
        workspace.write_work(files, result.text or "(the delegate produced no text)")
        summary = (result.text or "").strip()
        note = "" if result.stopped_because == "answered" else f" (stopped: {result.stopped_because})"
        body = (
            f"{binding.model} answered as {role}{note}.\n\n"
            f"{summary[:2000]}\n\n"
            f"Files: {paths['work']} (result), {paths['transcript']} (what it did), {paths['brief']}."
        )
    except BaseException as exc:  # noqa: BLE001 - reported below, after the restore
        body = f"Error: the {role} model failed: {exc}"
    finally:
        _active.delegation_id = None
        restore_note = _restore(resident)
    return body if not restore_note else f"{body}\n\n{restore_note}"


def _restore(resident: Optional[str]) -> str:
    """Put back what was loaded. A failure here is reported, never swallowed:
    otherwise the next turn runs on a different model than the user believes."""
    if not resident:
        return ""
    try:
        _loader.load(resident, {})
        return ""
    except BaseException as exc:  # noqa: BLE001
        return (
            f"WARNING: could not reload {resident} after the delegation ({exc}). "
            f"The model now loaded is not the one you were using."
        )


def _delegate_tools() -> list:
    """Everything the primary can use, minus delegation itself."""
    from core.inference.tools import ALL_TOOLS

    return [t for t in ALL_TOOLS if t.get("function", {}).get("name") not in DELEGATION_TOOL_NAMES]


def _delegate_execute_tool():
    from core.inference.tools import execute_tool

    return execute_tool
```

- [ ] **Step 5: Create the model-call module**

Create `studio/backend/core/inference/delegation/model_call.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One turn from the loaded backend, as an OpenAI-style assistant message.

llama_cpp is do-not-edit and already exposes base_url and _auth_headers, so this
talks to the server it is already running rather than starting anything."""

from __future__ import annotations

import json
import sys
import urllib.request

_TIMEOUT_S = 300


def call_loaded_model(messages: list, tools: list) -> dict:
    module = sys.modules.get("routes.inference")
    backend = getattr(module, "_llama_cpp_backend", None) if module is not None else None
    if backend is None or not getattr(backend, "is_loaded", False):
        return {"content": "Error: no model is loaded to answer as the delegate.", "tool_calls": []}
    payload = {"model": backend.model_identifier or "local", "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    request = urllib.request.Request(
        f"{backend.base_url}/v1/chat/completions",
        data = json.dumps(payload).encode("utf-8"),
        headers = {"Content-Type": "application/json", **(backend._auth_headers or {})},
    )
    with urllib.request.urlopen(request, timeout = _TIMEOUT_S) as response:
        body = json.loads(response.read().decode("utf-8"))
    choice = (body.get("choices") or [{}])[0]
    return choice.get("message") or {"content": "", "tool_calls": []}
```

- [ ] **Step 6: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_delegation_tool.py tests/test_delegation_subagent.py tests/test_delegation_workspace.py tests/test_delegation_loader.py tests/test_model_roles.py tests/test_model_roles_api.py -q -p no:cacheprovider
```
Expected: 45 passed.

- [ ] **Step 7: Commit**

```bash
git add studio/backend/core/inference/delegation/ studio/backend/tests/test_delegation_tool.py
git commit -m "feat(delegation): ask_model -- swap, run, restore

Resolve the role, refuse if it is not ready (no swap is attempted), write the
brief, record the RESIDENT model, swap, run the bounded sub-agent through the
audited execute_tool, write the work file, and restore in a finally.

What gets restored is what was resident when the call started -- never whatever
the 'primary' role names. Conflating them would silently change which model the
user is talking to. A FAILED restore is reported in the returned text: this is
the one place where never-raises must not become quietly-lies.

The per-turn cap is implemented as 2 per session per 10 minutes, because a tool
receives a session id and cannot observe turn boundaries; the refusal says so.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 8: Demonstrate the controls (after committing)**

1. In `_execute`, restore `model_roles.resolve("primary").model` instead of `resident`. Expect `test_the_restore_targets_what_was_resident_not_the_primary_role` to FAIL.
2. Move the `_restore` call out of the `finally` into the success path. Expect `test_the_model_is_restored_even_when_the_delegate_raises` to FAIL.
3. Make `_restore` swallow its exception and return `""`. Expect `test_a_failed_restore_is_reported_loudly` to FAIL.
4. Remove the `active_delegation_id()` check. Expect `test_a_nested_delegation_is_refused` to FAIL.
5. Make `_window_ok` always return True. Expect `test_the_window_cap_refuses_a_third_delegation` to FAIL.
6. Move the availability check after the `_loader.load` call. Expect `test_an_unavailable_role_never_loads_anything` to FAIL.

Restore after each; finish with a clean `git status --porcelain`.

---

### Task 7: Audit attribution

**Files:**
- Modify: `studio/backend/storage/tool_audit_db.py`, `studio/backend/core/inference/tool_audit/__init__.py`
- Test: `studio/backend/tests/test_delegation_audit.py`

**Interfaces:**
- Consumes: `delegation.active_delegation_id()` (Task 6)
- Produces: `tool_audit` rows carry a nullable `delegation_id`; `record_start(..., delegation_id = None)`

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_delegation_audit.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Delegate tool calls must be distinguishable from the primary's.

Attribution goes in its own column rather than in session_id: approvals are
keyed by session, so changing it would send the prompt somewhere the user is not
looking.
"""

from __future__ import annotations

import pytest

from core.inference import delegation
from core.inference import tool_audit
from storage import tool_audit_db


@pytest.fixture(autouse = True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    tool_audit_db.reset_for_tests()
    tool_audit.reset_degraded_for_tests()
    delegation._reset_for_tests()
    yield
    delegation._reset_for_tests()


def _fake_tool(name, arguments, **kwargs):
    return "ok"


def test_a_primary_tool_call_has_no_delegation_id():
    tool_audit.around(_fake_tool, "terminal", {"command": "ls"}, session_id = "s1")
    row = tool_audit_db.query_entries()[0]
    assert row["delegation_id"] in (None, "")


def test_a_delegate_tool_call_carries_the_delegation_id():
    delegation._active.delegation_id = "deleg-123"
    try:
        tool_audit.around(_fake_tool, "terminal", {"command": "ls"}, session_id = "s1")
    finally:
        delegation._active.delegation_id = None
    row = tool_audit_db.query_entries()[0]
    assert row["delegation_id"] == "deleg-123"


def test_the_session_id_is_unchanged_by_a_delegation():
    """Approvals are keyed by session; attribution must not move it."""
    delegation._active.delegation_id = "deleg-123"
    try:
        tool_audit.around(_fake_tool, "terminal", {}, session_id = "s1")
    finally:
        delegation._active.delegation_id = None
    assert tool_audit_db.query_entries()[0]["session_id"] == "s1"


def test_an_existing_database_gains_the_column(tmp_path, monkeypatch):
    """The table already exists in installed databases, so this needs a
    migration, not just a new CREATE TABLE."""
    tool_audit_db.reset_for_tests()
    tool_audit.around(_fake_tool, "terminal", {}, session_id = "s1")
    with tool_audit_db._connect() as conn:
        conn.execute("ALTER TABLE tool_audit RENAME TO tool_audit_old")
        conn.execute(
            "CREATE TABLE tool_audit AS SELECT id, tool_name, session_id FROM tool_audit_old"
        )
    tool_audit_db.reset_for_tests()
    tool_audit.around(_fake_tool, "terminal", {}, session_id = "s2")
    with tool_audit_db._connect() as conn:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(tool_audit)").fetchall()}
    assert "delegation_id" in columns
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_delegation_audit.py -q -p no:cacheprovider
```
Expected: `KeyError: 'delegation_id'`.

- [ ] **Step 3: Add the column and its migration**

In `studio/backend/storage/tool_audit_db.py`:

(a) Add `delegation_id TEXT` to the `CREATE TABLE tool_audit` statement, after `session_id`.

(b) In the schema-preparation function (the one guarded by `_SCHEMA_READY`), directly after the `CREATE TABLE` executes, add the migration — the pattern `storage/rag_db.py:300-302` uses:

```python
    columns = {r[1] for r in conn.execute("PRAGMA table_info(tool_audit)").fetchall()}
    if columns and "delegation_id" not in columns:
        conn.execute("ALTER TABLE tool_audit ADD COLUMN delegation_id TEXT")
```

(c) Give `record_start` a `delegation_id: Optional[str] = None` keyword and include it in the INSERT's columns and values.

(d) Include `delegation_id` in whatever `query_entries`/`get_entry` select, so the row dict carries it.

- [ ] **Step 4: Read the context variable in the recorder**

In `studio/backend/core/inference/tool_audit/__init__.py`, where `record_start` is called, pass the active delegation id:

```python
            delegation_id = _active_delegation_id(),
```

and add the helper near the top of the module:

```python
def _active_delegation_id():
    """The delegation running on this thread, if any. Never raises: attribution
    must not be able to break a tool call."""
    try:
        from core.inference import delegation

        return delegation.active_delegation_id()
    except BaseException:  # noqa: BLE001
        return None
```

- [ ] **Step 5: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_delegation_audit.py tests/test_tool_audit_recorder.py tests/test_tool_audit_db.py tests/test_tool_audit_routes.py -q -p no:cacheprovider
```
Expected: all pass — the existing audit tests included, since the column is additive and nullable.

- [ ] **Step 6: Commit**

```bash
git add studio/backend/storage/tool_audit_db.py studio/backend/core/inference/tool_audit/__init__.py studio/backend/tests/test_delegation_audit.py
git commit -m "feat(delegation): attribute delegate tool calls in the audit log

A nullable delegation_id column, set from the thread-local a delegation holds
while it runs. Rows written by the primary carry none.

Attribution goes in its own column rather than in session_id, because approvals
are keyed by session: changing it would route the confirmation prompt somewhere
the user is not looking.

The table exists in installed databases, so this ships with the same
PRAGMA-then-ALTER migration storage/rag_db.py already uses.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Demonstrate the controls (after committing)**

1. Make `_active_delegation_id` always return `None`. Expect `test_a_delegate_tool_call_carries_the_delegation_id` to FAIL.
2. Remove the `ALTER TABLE` migration. Expect `test_an_existing_database_gains_the_column` to FAIL.

Restore after each; finish with a clean `git status --porcelain`.

---

### Task 8: Registration and the seam

**Files:**
- Modify: `studio/backend/core/inference/tools.py`, `studio/backend/routes/inference.py`, `studio/backend/main.py`
- Test: `studio/backend/tests/test_delegation_seam.py`

**Interfaces:**
- Consumes: `delegation.schemas.DELEGATION_TOOLS` / `DELEGATION_TOOL_NAMES`, `delegation.execute`, `routes.model_roles.router`
- Produces: `ask_model` reachable by the model and dispatched; the roles API mounted

**`ask_model` is deliberately NOT added to `_ALWAYS_SAFE_TOOLS` or `_ANTHROPIC_UNPROMPTED_SAFE_TOOLS.`** It loads models, runs a sub-agent and can edit files, so it goes through the normal risk classification. Both omissions are asserted by tests.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_delegation_seam.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Registration, asserted by behaviour.

Being in ALL_TOOLS proves nothing: a previous tool in this project was
registered, dispatched, safe-listed and fully tested -- and unreachable in every
Studio chat, because a request-level filter stripped it. Reachability is
therefore driven through the REAL _select_request_tools.

Nothing is imported from another test module: studio/__init__.py and
studio/backend/__init__.py make a bare `tests` package resolve elsewhere under
pytest, and a collection error interrupts the whole run.
"""

from __future__ import annotations

import asyncio
import subprocess
import types
from pathlib import Path

import pytest

from core.inference import tool_audit
from core.inference import tools
from storage import tool_audit_db

_STUDIO_ALLOWLIST = ["web_search", "python", "terminal", "edit_file"]


@pytest.fixture(autouse = True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    tool_audit_db.reset_for_tests()
    tool_audit.reset_degraded_for_tests()
    yield


def _payload(**over):
    base = dict(enabled_tools = list(_STUDIO_ALLOWLIST), rag_scope = None,
                thread_id = None, bypass_permissions = False)
    base.update(over)
    return types.SimpleNamespace(**base)


def _select(payload, **kwargs):
    import routes.inference as routes_mod

    kwargs.setdefault("tools_on", True)
    kwargs.setdefault("mcp_allowed", False)
    return [t["function"]["name"]
            for t in asyncio.run(routes_mod._select_request_tools(payload, **kwargs))]


def test_ask_model_is_registered():
    assert "ask_model" in {t["function"]["name"] for t in tools.ALL_TOOLS}


def test_ask_model_reaches_the_model_through_the_real_route():
    assert "ask_model" in _select(_payload())


def test_an_empty_selection_stays_empty():
    assert _select(_payload(enabled_tools = [])) == []


def test_execute_tool_dispatches_to_the_delegation_handler():
    out = tools.execute_tool("ask_model", {})
    assert isinstance(out, str) and "role" in out.lower(), out


def test_ask_model_is_NOT_always_safe():
    """It loads models, runs a sub-agent and can edit files."""
    assert tools.is_high_risk_tool_call("ask_model", {"role": "coding", "task": "t"}) is True


def test_a_read_only_tool_is_still_not_high_risk():
    """Control: the assertion above is not passing because everything is high risk."""
    assert tools.is_high_risk_tool_call("check_tool_readiness", {}) is False


def test_ask_model_is_absent_from_the_anthropic_unprompted_set():
    inference = pytest.importorskip("routes.inference", reason = "inference stack not installed")
    assert "ask_model" not in inference._ANTHROPIC_UNPROMPTED_SAFE_TOOLS


def test_the_roles_router_is_mounted():
    from main import app

    assert any(getattr(r, "path", "") == "/api/model-roles" for r in app.routes)


def test_the_seams_stay_additive():
    repo = Path(tools.__file__).parents[4]
    base = subprocess.run(["git", "merge-base", "origin/main", "HEAD"], cwd = repo,
                          capture_output = True, text = True, check = True).stdout.strip()
    for path, ceiling in (
        ("studio/backend/core/inference/tools.py", 100),
        ("studio/backend/routes/inference.py", 70),
        ("studio/backend/main.py", 10),
    ):
        stat = subprocess.run(["git", "diff", "--numstat", base, "--", path], cwd = repo,
                              capture_output = True, text = True, check = True).stdout.split()
        assert stat, f"no diff recorded for {path}"
        insertions, deletions = int(stat[0]), int(stat[1])
        assert deletions == 0, f"{path} lost {deletions} line(s); the seam must be additive"
        assert insertions <= ceiling, f"{path} grew to {insertions}"
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_delegation_seam.py -q -p no:cacheprovider
```
Expected: the registration, reachability, dispatch and router tests FAIL; the two "absent from the safe sets" tests and the seam test already pass.

- [ ] **Step 3: Register in `tools.py`**

Three additive edits. **Do not modify or re-indent any existing line.**

(a) Directly after the readiness schemas import (`from core.inference.tool_readiness.schemas import READINESS_TOOLS, READINESS_TOOL_NAMES`), add:

```python
from core.inference.delegation.schemas import DELEGATION_TOOLS, DELEGATION_TOOL_NAMES
```

(b) In `ALL_TOOLS`, directly after `    *READINESS_TOOLS,`, add:

```python
    *DELEGATION_TOOLS,
```

(c) In `execute_tool`, directly after the readiness dispatch block, add:

```python
    if name in DELEGATION_TOOL_NAMES:
        from core.inference import delegation
        return delegation.execute(name, arguments, session_id = session_id, cancel_event = cancel_event)
```

**Do not touch `_ALWAYS_SAFE_TOOLS`.**

- [ ] **Step 4: Register in `routes/inference.py`**

Both lines are fork-inserted. In the `_addable` block's import, directly after `            READINESS_TOOL_NAMES,`, add:

```python
            DELEGATION_TOOL_NAMES,
```

and extend the union — **keeping it on ONE line**, because `tests/test_tool_readiness_seam.py::_route_addable()` parses that single line:

```python
        _addable = ASSIST_VISION_TOOL_NAMES | ASSIST_CODE_TOOL_NAMES | READINESS_TOOL_NAMES | DELEGATION_TOOL_NAMES
```

**Do not touch `_ANTHROPIC_UNPROMPTED_SAFE_TOOLS`.**

- [ ] **Step 5: Mount the roles router in `main.py`**

Two additive lines, matching `routes/tool_audit.py`'s registration exactly. Beside
`from routes.tool_audit import router as tool_audit_router`, add:

```python
from routes.model_roles import router as model_roles_router
```

and beside `app.include_router(tool_audit_router, prefix = "/api/tool-audit", tags = ["tool-audit"])`, add:

```python
app.include_router(model_roles_router, prefix = "/api/model-roles", tags = ["model-roles"])
```

That makes `main.py` 4 → **6** insertions, still zero deletions.

- [ ] **Step 6: Verify the seams**

From `C:\Users\Admin\unsloth`:
```
git diff --numstat $(git merge-base origin/main HEAD) -- studio/backend/core/inference/tools.py
git diff --numstat $(git merge-base origin/main HEAD) -- studio/backend/routes/inference.py
git diff --numstat $(git merge-base origin/main HEAD) -- studio/backend/main.py
```
Expected: `93 0`, `59 0`, `6 0`. **The second number must be 0 on every line.**

- [ ] **Step 7: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_delegation_seam.py tests/test_delegation_tool.py tests/test_delegation_audit.py tests/test_delegation_subagent.py tests/test_delegation_workspace.py tests/test_delegation_loader.py tests/test_model_roles.py tests/test_model_roles_api.py tests/test_tool_readiness_seam.py tests/test_tool_readiness_probes.py tests/test_tool_readiness_registry.py tests/test_tool_readiness_enrichment.py tests/test_tool_audit_seam.py tests/test_tool_audit_recorder.py tests/test_tool_audit_db.py tests/test_tool_audit_routes.py tests/test_assist_vision_request_tools.py tests/test_assist_code_registration.py tests/test_run_tools_locally_discriminator.py -q -p no:cacheprovider
```
Expected: all pass. If any of the three real-route suites fails, stop and report it — adding a schema to the re-add must not change their behaviour.

- [ ] **Step 8: Commit**

```bash
git add studio/backend/core/inference/tools.py studio/backend/routes/inference.py studio/backend/main.py studio/backend/tests/test_delegation_seam.py
git commit -m "feat(delegation): register ask_model, mount the roles API

ALL_TOOLS, the execute_tool dispatch, and the _addable re-add in
routes/inference.py -- the re-add has been missed three times in this project,
so reachability is asserted by driving the REAL _select_request_tools.

ask_model is deliberately absent from _ALWAYS_SAFE_TOOLS and
_ANTHROPIC_UNPROMPTED_SAFE_TOOLS: it loads models, runs a sub-agent and can edit
files, so it goes through the normal risk classification. Both omissions are
asserted, with a control proving not everything reads as high risk.

tools.py 88 -> 93 insertions, routes/inference.py 58 -> 59, main.py 4 -> 6, all
with zero deletions.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 9: Demonstrate the controls (after committing)**

1. Remove ` | DELEGATION_TOOL_NAMES` from `_addable`. Expect `test_ask_model_reaches_the_model_through_the_real_route` to FAIL.
2. Add `"ask_model"` to the `_ALWAYS_SAFE_TOOLS` rebind. Expect `test_ask_model_is_NOT_always_safe` to FAIL.
3. Remove the `include_router` line from `main.py`. Expect `test_the_roles_router_is_mounted` to FAIL.

Restore after each with `git checkout -- <file>`; finish with a clean `git status --porcelain` and re-run Step 6's numstats.

- [ ] **Step 10: Manual verification — the real swap**

The spec records this as owed, and it is the first time a real swap happens. With Unslothed running, two roles bound (one to the model you have loaded, one to a small second model), ask the model in chat to use `ask_model` with the second role. Record in your report:

- how long the delegation took end to end, and how much of that was the two loads
- whether the model you started with is the one loaded afterwards
- whether the delegation folder contains `brief.md`, `work.md` and `transcript.md`
- whether the delegate's tool calls appear in the audit log with a `delegation_id`

If the swap-back does not happen, or the returned text does not warn about it, that is a Critical finding — report it rather than working around it.

---

## Self-Review

**Spec coverage.** Roles, storage shape and availability → Task 1. The configuration surface → Task 2. Brief/work/transcript under the session workdir → Task 3. The swap adapter reusing the route's loader → Task 4. Budgets, cancellation and the bounded loop → Task 5. Orchestration, restore-what-was-resident, loud restore failure, depth guard and the delegation cap → Task 6. Audit attribution and its migration → Task 7. Five registration points, the two deliberate omissions, and the seam → Task 8. The spec's "manual run owed" appears twice: a read-only check in Task 4 and the real swap in Task 8.

**Deviations from the spec, stated rather than glossed.**
1. **The delegation cap is per session per 10 minutes, not per assistant turn.** A tool receives a session id and cannot observe turn boundaries. The code and the refusal message both say so.
2. **`model_call.py` is an extra module** the spec's file table did not name. The spec says a GGUF delegate needs "a small fork-owned loop posting to the loaded backend's OpenAI-compatible endpoint"; splitting the HTTP call out of the loop is what keeps the loop testable without a backend.
3. **The delegate receives every tool except `ask_model`.** The spec says "full tools"; the exclusion is the depth guard's first layer.

**Placeholders.** None; every code step carries real code. Two steps deliberately verify a name against the live code before use (`LoadRequest`'s field in Task 4, the router import style in Task 8) rather than guessing — those are checks, not gaps.

**Type consistency.** `RoleBinding(role, model, overrides)` is produced in Task 1 and consumed in Tasks 2 and 6. `DelegationFiles(delegation_id, root, brief, work, transcript)` is produced in Task 3 and consumed in Task 6. `SubagentResult(text, rounds, stopped_because)` is produced in Task 5 and consumed in Task 6. `loader.resident_model_id()` / `loader.load(model_id, overrides)` are produced in Task 4 and consumed in Task 6 through the `_loader` seam. `DELEGATION_TOOLS` / `DELEGATION_TOOL_NAMES` are produced in Task 6 and consumed in Task 8.

**Cross-task shared state.** Tasks 6 and 7 share the delegation thread-local: Task 6 owns it, Task 7 only reads it, through a guarded helper that returns `None` if the import fails. Tasks 6 and 8 share `tools.ALL_TOOLS`: Task 6 reads it lazily at call time (so registration in Task 8 is picked up without an import cycle), and Task 8 writes it.

**Ordering.** 1 → 2; 3, 4 and 5 are independent of each other; 6 needs 1, 3, 4 and 5; 7 needs 6; 8 needs 2, 6 and 7. Execute in order 1 → 8.
