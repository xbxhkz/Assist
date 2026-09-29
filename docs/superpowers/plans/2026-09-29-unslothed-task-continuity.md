# Task Continuity Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dependency-free continuity core (project state, a task graph, a failure log,
checkpoints) that a dev CLI and a new built-in agent tool both call, replacing nothing that
already works (`progress.md`, `.remember`, the memory files) and filling only the genuine gap:
machine-readable, queryable project/task state.

**Architecture:** One pure-Python package (`core/continuity/`) holds all state logic — atomic
file writes, schema validation, status-transition rules, rendering. Two thin consumers sit on
top of it: a CLI script for direct use (mirroring how `check_branding.py` is invoked), and a
built-in agent tool (`continuity_task`) registered the same way every other built-in tool in
this fork is. The core raises on error; each consumer decides what to do with that.

**Tech Stack:** Python stdlib only in the core (`dataclasses`, `json`, `os`, `re`, `time`,
`uuid`) — no new dependency. pytest for tests, matching the rest of `studio/backend`.

**Spec:** `C:\Users\Admin\odysseus\docs\superpowers\specs\2026-09-29-unslothed-task-continuity-design.md`
— read it alongside this plan; the plan argues from it and does not repeat its reasoning.

## Global Constraints

- Core lives at `studio/backend/core/continuity/` (`__init__.py`, `schemas.py`, `storage.py`,
  `render.py`) and imports nothing outside the standard library plus `dataclasses`.
- Every write is atomic: write to a dotted temp path, then `os.replace(tmp, final)`, with the
  temp file removed on any failure. This is the existing pattern at
  `core/inference/image_gallery.py:71-81` — follow it, don't invent a new one.
- The core **raises** `ContinuityError` (defined in `schemas.py`) on a bad read, a bad write, an
  illegal task-status transition, or a `render_context_summary` that cannot fit its cap. It never
  guesses at recovery and never returns a sentinel in place of raising. Never-raises is imposed
  by the *caller* (the CLI catches and prints; the app tool catches and returns a string) — the
  core itself does not have a bare `except`.
- `task_queue.json`'s `"ready"` status is **derived**, never stored: a `pending` task whose every
  `depends_on` id is `complete`. There is no code path that writes `status = "ready"` directly.
- `set_task_status` enforces the transition table from the spec (§5). An illegal transition
  raises `ContinuityError` naming the attempted and current status. A transition *to* `"complete"`
  additionally requires the task's `acceptance_criteria` list to be non-empty, or it raises.
- `repair()` regenerates only *derived* data (the rendered `.md` files; re-deriving which tasks
  are ready is implicit, since readiness is never stored) — it never changes a task's `status`
  field, never adds an acceptance criterion, never marks anything complete.
- `record_error` is the one core function that is never-raises **internally**: if the primary
  write to `logs/errors.jsonl` fails, it falls back to appending a line into
  `C:\Users\Admin\odysseus\.remember\now.md` (format: `## <ISO time> | continuity-fallback\n<what_tried> -- FAILED: <why_failed>\n`) rather than losing the record. This fallback path is
  itself wrapped in a bare `except BaseException: pass` — losing the record of a failure is bad,
  crashing the caller over it is worse.
- `render_context_summary` targets 300–800 tokens, hard-caps at 1500. Token count is approximated
  as `len(text) // 4` (no tokenizer dependency) — document this approximation in the function's
  docstring so nobody mistakes it for an exact count. Exceeding the cap after truncation raises
  `ContinuityError`, it does not silently return an oversized string.
- Path confinement reuses `core.inference.tools._get_workdir(session_id)` and
  `core.inference.tools._is_outside_workdir(candidate, workdir)` exactly as
  `core/inference/assist_vision/paths.py:44-50` does — do not write a second confinement helper.
  Both are imported lazily inside the function that needs them, matching that file's own stated
  reason (`tools` is large and would eventually import this package back).
- Dev CLI: `studio/backend/tools/continuity_cli.py`, run directly with
  `C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe`, the same way
  `packaging/check_branding.py` is invoked — no new venv, no new entry point registered anywhere.
- The app tool, `continuity_task`, needs exactly **four** registration points (not five — this
  piece adds no HTTP router, so `main.py` is untouched):
  1. `ALL_TOOLS` in `core/inference/tools.py` (`from core.inference.continuity.schemas import CONTINUITY_TOOLS, CONTINUITY_TOOL_NAMES`, added after `*DELEGATION_TOOLS,`)
  2. the `execute_tool` dispatch in the same file, added after the `DELEGATION_TOOL_NAMES` block
  3. the `_addable` re-add in `routes/inference.py` (`continuity_task` has no UI pill, same as
     `check_tool_readiness`, `find_capability`, and delegation's tools — omitting this makes it
     registered, dispatched, safe-listed, fully tested, and unreachable in Studio chat, which has
     happened three times in this project already)
  4. `_ALWAYS_SAFE_TOOLS` in `tools.py` **and** `_ANTHROPIC_UNPROMPTED_SAFE_TOOLS` in
     `routes/inference.py` — both rebound additively (`X = X | frozenset({...})`, never editing
     the literal), because `continuity_task` reads and writes only local sandboxed files, no
     network, no code execution, so unlike `ask_model` it belongs on both safe lists.
- `main.py` is **not** touched by this plan. Its current additive count (6 insertions, 0
  deletions vs. the upstream merge-base) does not change.
- `tools.py` is currently **114** insertions / 0 deletions vs. `git merge-base origin/main HEAD`;
  `routes/inference.py` is **60/0**. Both grow in Task 5 and the seam-ceiling tests in
  `tests/test_delegation_seam.py`, `tests/test_tool_audit_seam.py`, and
  `tests/test_tool_readiness_seam.py` (three files, kept in sync per their own comments) must be
  raised together, and only together, with the `deletions == 0` assertion in each left untouched.
- `llama_cpp.py` and `pyproject.toml` are do-not-edit for this plan, full stop.
- **NEVER run the full backend test suite.** Upstream fixtures fabricate GGUF files up to 40 GB
  and have filled this machine's disk before. Every "run the tests" step in this plan names exact
  files. If you are ever tempted to run `pytest` with no path argument, stop — that is the full
  suite.
- No test in this plan may load a model, start a backend, or call a real tool. The core is pure
  file I/O; nothing here needs a model or a network call to test.
- Every guard gets its own negative control, demonstrated by mutation **after** that task's
  commit (never before — mutating uncommitted work and then running `git checkout -- <file>` to
  "restore" it has destroyed real work twice in this project's history; commit first, always).

## Review Focus

The spec describes the steady-state API in detail but is quieter on five inputs a real caller
will eventually hand it. Each line names the input and the behavior a reasonable person would
expect; each gets a test in the task that owns the relevant code.

- **A `project_dir` that does not exist yet.** `load_state`/`load_tasks` on a directory with no
  `.ai/` folder should return `None` / an empty `TaskQueue`, not raise — "nothing has been
  initialized here" is not an error, it is the starting state. (Task 2, Task 3)
- **`add_task` given a `depends_on` id that names no task, at all — not even one that exists in a
  cycle.** A task graph a human hand-edits will have typos. This should raise `ContinuityError`
  naming the missing id, not silently create an orphan dependency that `ready_tasks` then
  mis-evaluates. (Task 3)
- **A `task_queue.json` with a dependency cycle** (A depends on B, B depends on A). `validate`
  must report it as a `Problem`, not hang or silently treat both as permanently not-ready with no
  explanation. (Task 6)
- **`prior_failures` called with a `symptom_query` that matches nothing.** Must return `[]`, not
  raise and not return every entry (a naive "empty query matches everything" bug would make the
  guard against repeating a failure into noise the first time nothing matches). (Task 4)
- **`checkpoint()` called when more than the retention limit already exist.** The spec says
  "pruned to the most recent N (default 20)" — a test must prove the *oldest* checkpoint is the
  one removed, not an arbitrary one, since `os.listdir` order is not sorted order on every
  filesystem. (Task 5)

---

### Task 1: Schemas, `ContinuityError`, and the storage primitives

**Files:**
- Create: `studio/backend/core/continuity/__init__.py` (empty at this point — populated in later tasks; this task only needs the package to exist so `schemas.py`/`storage.py` are importable as `core.inference.continuity` — wait, check the actual package path)
- Create: `studio/backend/core/continuity/schemas.py`
- Create: `studio/backend/core/continuity/storage.py`
- Test: `studio/backend/tests/test_continuity_storage.py`

**Interfaces:**
- Consumes: nothing (this is the bottom of the stack)
- Produces:
  - `schemas.ContinuityError(Exception)` — one attribute, `.message: str`
  - `schemas.ProjectState` — a `@dataclass` with the fields from spec §5 (`schema_version: int`,
    `project: str`, `status: str`, `current_phase: str`, `current_task: Optional[str]`,
    `completion_percent: int`, `last_checkpoint: Optional[str]`, `completed: list[str]`,
    `in_progress: list[str]`, `remaining: list[str]`, `next_action: str`,
    `modified_files: list[str]`, `tests: dict`, `blocking_issues: list[str]`,
    `important_decisions: list[str]`), plus `.to_dict() -> dict` and
    `ProjectState.from_dict(d: dict) -> ProjectState` (raises `ContinuityError` on a missing
    required key or wrong type for `schema_version`)
  - `schemas.Task` — `@dataclass(id: str, title: str, status: str, depends_on: list[str],
    acceptance_criteria: list[str], owner: Optional[str])`, `.to_dict()`, `Task.from_dict(d)`
  - `schemas.TaskQueue` — `@dataclass(schema_version: int, tasks: list[Task])`, `.to_dict()`,
    `TaskQueue.from_dict(d)`
  - `schemas.ErrorEntry` — `@dataclass(ts: str, project: str, what_tried: str, why_failed: str,
    symptom: str, do_not_repeat: bool)`, `.to_dict()`, `ErrorEntry.from_dict(d)`
  - `storage.CURRENT_SCHEMA_VERSION = 1`
  - `storage.write_json_atomic(path: str, data: dict) -> None` — the temp-then-replace primitive
    every later write builds on
  - `storage.read_json(path: str) -> Optional[dict]` — returns `None` if the file does not exist;
    raises `ContinuityError` if it exists but is not valid JSON (never returns `None` for a
    corrupt file — that would be indistinguishable from "never written")
  - `storage.ai_dir(project_dir: str) -> str` — returns `os.path.join(project_dir, ".ai")`,
    creating it (and `logs/`, `checkpoints/` beneath it) with `os.makedirs(..., exist_ok=True)` if
    absent

The package path is `studio/backend/core/continuity/` (a top-level package under `core/`, a
sibling of `core/inference/`, `core/training/`, `core/rag/` — not nested under `core/inference/`,
since this is not inference-specific). Confirm this against the existing layout before creating
directories: run `ls studio/backend/core/` first.

- [ ] **Step 1: Write the failing test for the atomic write primitive**

Create `studio/backend/tests/test_continuity_storage.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The storage primitives every continuity write builds on: atomic writes and a
read that tells a missing file apart from a corrupt one.

No test here touches a model, a backend, or a real tool -- this is pure file I/O
against tmp_path.
"""

from __future__ import annotations

import json
import os

import pytest

from core.continuity.schemas import ContinuityError
from core.continuity import storage


def test_write_json_atomic_writes_valid_json(tmp_path):
    target = str(tmp_path / "state.json")
    storage.write_json_atomic(target, {"a": 1})
    with open(target, encoding = "utf-8") as f:
        assert json.load(f) == {"a": 1}


def test_write_json_atomic_leaves_no_temp_file_on_success(tmp_path):
    target = str(tmp_path / "state.json")
    storage.write_json_atomic(target, {"a": 1})
    leftovers = [p for p in os.listdir(tmp_path) if p != "state.json"]
    assert leftovers == [], f"temp file(s) left behind: {leftovers}"


def test_read_json_missing_file_returns_none(tmp_path):
    assert storage.read_json(str(tmp_path / "nope.json")) is None


def test_read_json_corrupt_file_raises(tmp_path):
    target = tmp_path / "bad.json"
    target.write_text("{not valid json", encoding = "utf-8")
    with pytest.raises(ContinuityError):
        storage.read_json(str(target))


def test_ai_dir_creates_the_directory_tree(tmp_path):
    project_dir = str(tmp_path)
    result = storage.ai_dir(project_dir)
    assert result == os.path.join(project_dir, ".ai")
    assert os.path.isdir(os.path.join(result, "logs"))
    assert os.path.isdir(os.path.join(result, "checkpoints"))
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_storage.py -q -p no:cacheprovider
```
Expected: `ModuleNotFoundError: No module named 'core.continuity'`.

- [ ] **Step 3: Create the package and schemas**

Create `studio/backend/core/continuity/__init__.py` (empty for now):

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
```

Create `studio/backend/core/continuity/schemas.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The continuity engine's data shapes.

Every dataclass here round-trips through to_dict()/from_dict() rather than a
generic asdict(): from_dict is where a missing or wrong-typed field becomes a
ContinuityError instead of a confusing KeyError three calls later, at whichever
place first happens to read the missing field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class ContinuityError(Exception):
    """Raised by every core.continuity function on a bad read, a bad write, an
    illegal status transition, or a render that cannot fit its cap. The core
    never guesses at recovery in place of raising -- that is each caller's own
    decision to make, not this module's."""


def _require(d: dict, key: str, expected_type: type):
    if key not in d:
        raise ContinuityError(f"missing required field {key!r}")
    value = d[key]
    if not isinstance(value, expected_type):
        raise ContinuityError(
            f"field {key!r} must be {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


@dataclass
class ProjectState:
    schema_version: int
    project: str
    status: str
    current_phase: str
    current_task: Optional[str]
    completion_percent: int
    last_checkpoint: Optional[str]
    completed: list[str] = field(default_factory = list)
    in_progress: list[str] = field(default_factory = list)
    remaining: list[str] = field(default_factory = list)
    next_action: str = ""
    modified_files: list[str] = field(default_factory = list)
    tests: dict = field(default_factory = dict)
    blocking_issues: list[str] = field(default_factory = list)
    important_decisions: list[str] = field(default_factory = list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "project": self.project,
            "status": self.status, "current_phase": self.current_phase,
            "current_task": self.current_task,
            "completion_percent": self.completion_percent,
            "last_checkpoint": self.last_checkpoint, "completed": self.completed,
            "in_progress": self.in_progress, "remaining": self.remaining,
            "next_action": self.next_action, "modified_files": self.modified_files,
            "tests": self.tests, "blocking_issues": self.blocking_issues,
            "important_decisions": self.important_decisions,
        }

    @staticmethod
    def from_dict(d: dict) -> "ProjectState":
        return ProjectState(
            schema_version = _require(d, "schema_version", int),
            project = _require(d, "project", str),
            status = _require(d, "status", str),
            current_phase = _require(d, "current_phase", str),
            current_task = d.get("current_task"),
            completion_percent = _require(d, "completion_percent", int),
            last_checkpoint = d.get("last_checkpoint"),
            completed = list(d.get("completed") or []),
            in_progress = list(d.get("in_progress") or []),
            remaining = list(d.get("remaining") or []),
            next_action = d.get("next_action") or "",
            modified_files = list(d.get("modified_files") or []),
            tests = dict(d.get("tests") or {}),
            blocking_issues = list(d.get("blocking_issues") or []),
            important_decisions = list(d.get("important_decisions") or []),
        )


@dataclass
class Task:
    id: str
    title: str
    status: str
    depends_on: list[str] = field(default_factory = list)
    acceptance_criteria: list[str] = field(default_factory = list)
    owner: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "title": self.title, "status": self.status,
            "depends_on": self.depends_on,
            "acceptance_criteria": self.acceptance_criteria, "owner": self.owner,
        }

    @staticmethod
    def from_dict(d: dict) -> "Task":
        return Task(
            id = _require(d, "id", str), title = _require(d, "title", str),
            status = _require(d, "status", str),
            depends_on = list(d.get("depends_on") or []),
            acceptance_criteria = list(d.get("acceptance_criteria") or []),
            owner = d.get("owner"),
        )


@dataclass
class TaskQueue:
    schema_version: int
    tasks: list[Task] = field(default_factory = list)

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version,
                "tasks": [t.to_dict() for t in self.tasks]}

    @staticmethod
    def from_dict(d: dict) -> "TaskQueue":
        return TaskQueue(
            schema_version = _require(d, "schema_version", int),
            tasks = [Task.from_dict(t) for t in (d.get("tasks") or [])],
        )


@dataclass
class ErrorEntry:
    ts: str
    project: str
    what_tried: str
    why_failed: str
    symptom: str
    do_not_repeat: bool = True

    def to_dict(self) -> dict:
        return {
            "ts": self.ts, "project": self.project, "what_tried": self.what_tried,
            "why_failed": self.why_failed, "symptom": self.symptom,
            "do_not_repeat": self.do_not_repeat,
        }

    @staticmethod
    def from_dict(d: dict) -> "ErrorEntry":
        return ErrorEntry(
            ts = _require(d, "ts", str), project = _require(d, "project", str),
            what_tried = _require(d, "what_tried", str),
            why_failed = _require(d, "why_failed", str),
            symptom = _require(d, "symptom", str),
            do_not_repeat = bool(d.get("do_not_repeat", True)),
        )
```

Create `studio/backend/core/continuity/storage.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Atomic file I/O for the continuity engine. Every write in this package goes
through write_json_atomic; nothing writes a JSON file directly.

Pattern taken from core/inference/image_gallery.py:71-81 -- a dotted temp name
(never matches a real *.json listing), os.replace for the atomic swap, and the
temp file removed on any failure so a crash never leaves a stray partial write
sitting next to the real file.
"""

from __future__ import annotations

import json
import os
import uuid

from core.continuity.schemas import ContinuityError

_AI_DIR_NAME = ".ai"


def write_json_atomic(path: str, data: dict) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok = True)
    tmp_path = os.path.join(directory, f".{os.path.basename(path)}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp_path, "w", encoding = "utf-8") as f:
            json.dump(data, f, indent = 2)
            f.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def read_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding = "utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContinuityError(f"could not read {path}: {exc}") from exc


def ai_dir(project_dir: str) -> str:
    """The .ai/ directory under project_dir, created (with logs/ and
    checkpoints/) if it does not exist yet. Never raises for "not initialized
    yet" -- that is the normal starting state, not an error."""
    root = os.path.join(project_dir, _AI_DIR_NAME)
    os.makedirs(os.path.join(root, "logs"), exist_ok = True)
    os.makedirs(os.path.join(root, "checkpoints"), exist_ok = True)
    return root
```

- [ ] **Step 4: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_storage.py -q -p no:cacheprovider
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add studio/backend/core/continuity/__init__.py studio/backend/core/continuity/schemas.py studio/backend/core/continuity/storage.py studio/backend/tests/test_continuity_storage.py
git commit -m "feat(continuity): schemas and atomic storage primitives

Dependency-free dataclasses (ProjectState, Task, TaskQueue, ErrorEntry) with
explicit from_dict validation, and the atomic write every later write builds on
-- dotted temp file, os.replace, cleanup on failure. Pattern taken from
image_gallery.py's own atomic PNG write rather than invented fresh.

read_json tells a missing file (None) apart from a corrupt one (raises
ContinuityError) -- collapsing the two would make an uninitialized project
indistinguishable from a damaged one.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Demonstrate the control (after committing)**

In `write_json_atomic`, comment out the `os.replace(tmp_path, path)` line (leave everything
else). Run `test_write_json_atomic_writes_valid_json`. Expected: FAIL — `state.json` does not
exist (the write never landed), proving the test actually checks the final file rather than
something incidental. Restore with `git checkout -- studio/backend/core/continuity/storage.py`.
Finish with a clean `git status --porcelain`.

---

### Task 2: `ProjectState` load / write / update

**Files:**
- Modify: `studio/backend/core/continuity/__init__.py`
- Test: `studio/backend/tests/test_continuity_state.py`

**Interfaces:**
- Consumes: `storage.write_json_atomic`, `storage.read_json`, `storage.ai_dir` (Task 1);
  `schemas.ProjectState`, `schemas.ContinuityError` (Task 1)
- Produces: `continuity.load_state(project_dir: str) -> Optional[ProjectState]`,
  `continuity.write_state(project_dir: str, state: ProjectState) -> None`,
  `continuity.update_state(project_dir: str, **fields) -> ProjectState`

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_continuity_state.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""project_state.json: load, write, update. No test touches a model, a backend,
or a real tool -- this is file I/O against tmp_path."""

from __future__ import annotations

import pytest

from core.continuity import load_state, write_state, update_state
from core.continuity.schemas import ContinuityError, ProjectState


def _state(**over):
    base = dict(
        schema_version = 1, project = "demo", status = "active",
        current_phase = "phase-1", current_task = None, completion_percent = 0,
        last_checkpoint = None,
    )
    base.update(over)
    return ProjectState(**base)


def test_load_state_on_an_uninitialized_project_returns_none(tmp_path):
    assert load_state(str(tmp_path)) is None


def test_write_then_load_round_trips(tmp_path):
    write_state(str(tmp_path), _state(project = "demo", next_action = "do the thing"))
    loaded = load_state(str(tmp_path))
    assert loaded is not None
    assert loaded.project == "demo"
    assert loaded.next_action == "do the thing"


def test_update_state_requires_an_existing_state(tmp_path):
    with pytest.raises(ContinuityError):
        update_state(str(tmp_path), next_action = "x")


def test_update_state_merges_fields(tmp_path):
    write_state(str(tmp_path), _state(current_phase = "phase-1"))
    updated = update_state(str(tmp_path), current_phase = "phase-2", completion_percent = 40)
    assert updated.current_phase == "phase-2"
    assert updated.completion_percent == 40
    # untouched field survives the merge
    assert updated.project == "demo"
    assert load_state(str(tmp_path)).current_phase == "phase-2"


def test_update_state_rejects_an_unknown_field(tmp_path):
    write_state(str(tmp_path), _state())
    with pytest.raises(ContinuityError):
        update_state(str(tmp_path), not_a_real_field = "x")
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_state.py -q -p no:cacheprovider
```
Expected: `ImportError: cannot import name 'load_state' from 'core.continuity'`.

- [ ] **Step 3: Implement**

Write `studio/backend/core/continuity/__init__.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The continuity engine's public API.

Every function here operates on a project_dir the caller names explicitly.
Nothing in this module resolves a path on its own -- the app-side tool (added
in a later task) is what confines project_dir to a sandbox; this module trusts
whatever path it is given, the same way storage.py does.

The core RAISES ContinuityError rather than guessing at recovery. Never-raises
is each caller's own decision (the CLI catches and prints; the app tool catches
and returns a string) -- this module does not have a bare except anywhere
except inside record_error's fallback path, documented at that function.
"""

from __future__ import annotations

import dataclasses
import os

from core.continuity import storage
from core.continuity.schemas import ContinuityError, ProjectState

_STATE_FILENAME = "project_state.json"


def _state_path(project_dir: str) -> str:
    return os.path.join(storage.ai_dir(project_dir), _STATE_FILENAME)


def load_state(project_dir: str) -> ProjectState | None:
    raw = storage.read_json(_state_path(project_dir))
    if raw is None:
        return None
    return ProjectState.from_dict(raw)


def write_state(project_dir: str, state: ProjectState) -> None:
    storage.write_json_atomic(_state_path(project_dir), state.to_dict())


def update_state(project_dir: str, **fields) -> ProjectState:
    current = load_state(project_dir)
    if current is None:
        raise ContinuityError(
            f"no project_state.json at {project_dir!r} yet -- call write_state first"
        )
    valid_fields = {f.name for f in dataclasses.fields(ProjectState)}
    unknown = set(fields) - valid_fields
    if unknown:
        raise ContinuityError(f"unknown ProjectState field(s): {sorted(unknown)}")
    merged = dataclasses.replace(current, **fields)
    write_state(project_dir, merged)
    return merged
```

- [ ] **Step 4: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_state.py -q -p no:cacheprovider
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add studio/backend/core/continuity/__init__.py studio/backend/tests/test_continuity_state.py
git commit -m "feat(continuity): project_state.json load/write/update

An uninitialized project returns None from load_state, not an error --
'nothing recorded yet' is the normal starting state. update_state is a
read-modify-write over dataclasses.replace, so a caller passing an unknown
field name (a typo) is caught immediately rather than silently ignored.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Demonstrate the control (after committing)**

In `update_state`, delete the `unknown = set(fields) - valid_fields` check and its `raise` (keep
the merge). Run `test_update_state_rejects_an_unknown_field`. Expected: FAIL — the bogus field is
silently accepted. Restore with `git checkout -- studio/backend/core/continuity/__init__.py`.
Finish with a clean `git status --porcelain`.

---

### Task 3: Task queue — add, status transitions, derived readiness

**Files:**
- Modify: `studio/backend/core/continuity/__init__.py`
- Test: `studio/backend/tests/test_continuity_tasks.py`

**Interfaces:**
- Consumes: `storage.*`, `schemas.Task`, `schemas.TaskQueue`, `schemas.ContinuityError` (Task 1)
- Produces: `continuity.load_tasks(project_dir) -> TaskQueue`,
  `continuity.add_task(project_dir, task: Task) -> None`,
  `continuity.set_task_status(project_dir, task_id: str, status: str) -> None`,
  `continuity.ready_tasks(project_dir) -> list[Task]`

**This is the task that owns two Review Focus items:** an uninitialized project's `load_tasks`
returning an empty queue rather than raising, and `add_task` rejecting a `depends_on` id that
names no existing task.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_continuity_tasks.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""task_queue.json: add, transitions, derived readiness.

"ready" is never a stored value -- test_ready_tasks_is_derived_not_stored below
is the one that would catch a future change that starts persisting it."""

from __future__ import annotations

import pytest

from core.continuity import add_task, load_tasks, ready_tasks, set_task_status
from core.continuity.schemas import ContinuityError, Task


def _task(id, **over):
    base = dict(id = id, title = f"task {id}", status = "pending", depends_on = [],
                acceptance_criteria = ["it works"], owner = None)
    base.update(over)
    return Task(**base)


def test_load_tasks_on_an_uninitialized_project_returns_an_empty_queue(tmp_path):
    q = load_tasks(str(tmp_path))
    assert q.tasks == []


def test_add_task_then_load_round_trips(tmp_path):
    add_task(str(tmp_path), _task("t1"))
    q = load_tasks(str(tmp_path))
    assert [t.id for t in q.tasks] == ["t1"]


def test_add_task_rejects_a_duplicate_id(tmp_path):
    add_task(str(tmp_path), _task("t1"))
    with pytest.raises(ContinuityError):
        add_task(str(tmp_path), _task("t1"))


def test_add_task_rejects_a_depends_on_naming_no_real_task(tmp_path):
    """Review Focus: a hand-edited task graph will have typos. A dependency on
    an id that exists nowhere must be caught at add time, not silently accepted
    and then mis-evaluated by ready_tasks."""
    with pytest.raises(ContinuityError):
        add_task(str(tmp_path), _task("t1", depends_on = ["does-not-exist"]))


def test_ready_tasks_is_derived_not_stored(tmp_path):
    add_task(str(tmp_path), _task("t1", status = "pending"))
    add_task(str(tmp_path), _task("t2", status = "pending", depends_on = ["t1"]))
    assert [t.id for t in ready_tasks(str(tmp_path))] == ["t1"]
    # Straight to in_progress: there is no stored "ready" state to pass
    # through, only the computed answer ready_tasks() already gave above.
    set_task_status(str(tmp_path), "t1", "in_progress")
    set_task_status(str(tmp_path), "t1", "complete")
    # t2's status field is still "pending" on disk; readiness is computed, not read.
    loaded = load_tasks(str(tmp_path))
    assert next(t for t in loaded.tasks if t.id == "t2").status == "pending"
    assert [t.id for t in ready_tasks(str(tmp_path))] == ["t2"]


def test_set_task_status_refuses_ready_as_a_write_target(tmp_path):
    """The other half of "derived, not stored": ready is not merely unused as
    a transition target, it is actively refused if a caller tries to set it
    explicitly -- there must be no way to create the second source of truth
    the whole design exists to avoid."""
    add_task(str(tmp_path), _task("t1", status = "pending"))
    with pytest.raises(ContinuityError):
        set_task_status(str(tmp_path), "t1", "ready")


def test_illegal_transition_raises(tmp_path):
    add_task(str(tmp_path), _task("t1", status = "pending"))
    with pytest.raises(ContinuityError):
        # pending -> complete directly is not a legal edge (spec section 5)
        set_task_status(str(tmp_path), "t1", "complete")


def test_complete_requires_non_empty_acceptance_criteria(tmp_path):
    add_task(str(tmp_path), _task("t1", status = "pending", acceptance_criteria = []))
    set_task_status(str(tmp_path), "t1", "in_progress")
    with pytest.raises(ContinuityError):
        set_task_status(str(tmp_path), "t1", "complete")


def test_set_task_status_on_a_missing_id_raises(tmp_path):
    with pytest.raises(ContinuityError):
        set_task_status(str(tmp_path), "nope", "in_progress")
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_tasks.py -q -p no:cacheprovider
```
Expected: `ImportError: cannot import name 'add_task' from 'core.continuity'`.

- [ ] **Step 3: Implement**

Append to `studio/backend/core/continuity/__init__.py`:

```python
from core.continuity.schemas import Task, TaskQueue

_TASKS_FILENAME = "task_queue.json"

# Legal edges. "ready" is NEVER a key here and never appears as a write
# target -- it is a status a task can only be JUDGED to have (ready_tasks
# computes it from depends_on + status), never one it can be SET to. Deriving
# "ready" and then also allowing set_task_status(id, "ready") would be two
# sources of truth for the same fact, able to disagree. Because "ready" is
# absent from this dict, it is absent from _VALID_STATUSES too (built from
# this dict's keys below), so set_task_status(id, "ready") is refused with
# "unknown status" before the transition table is even consulted -- there is
# no special-case check needed to keep the promise the comment makes.
#
# Concretely: a pending task goes straight to in_progress once a caller has
# decided (typically by first calling ready_tasks()) that it is unblocked --
# there is no intermediate stored state to pass through. blocked returns to
# pending, not to a "ready" it was never blocked away from reaching that way.
_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"in_progress"},
    "in_progress": {"complete", "failed", "blocked"},
    "blocked": {"pending"},
    "failed": {"abandoned", "pending"},
    "abandoned": set(),
    "complete": set(),
}
_VALID_STATUSES = frozenset(_TRANSITIONS)


def _tasks_path(project_dir: str) -> str:
    return os.path.join(storage.ai_dir(project_dir), _TASKS_FILENAME)


def load_tasks(project_dir: str) -> TaskQueue:
    raw = storage.read_json(_tasks_path(project_dir))
    if raw is None:
        return TaskQueue(schema_version = storage.__dict__.get("CURRENT_SCHEMA_VERSION", 1) or 1, tasks = [])
    return TaskQueue.from_dict(raw)


def _write_tasks(project_dir: str, queue: TaskQueue) -> None:
    storage.write_json_atomic(_tasks_path(project_dir), queue.to_dict())


def add_task(project_dir: str, task: Task) -> None:
    if task.status not in _VALID_STATUSES:
        raise ContinuityError(f"unknown status {task.status!r}")
    queue = load_tasks(project_dir)
    existing_ids = {t.id for t in queue.tasks}
    if task.id in existing_ids:
        raise ContinuityError(f"a task with id {task.id!r} already exists")
    missing_deps = [d for d in task.depends_on if d not in existing_ids]
    if missing_deps:
        raise ContinuityError(
            f"task {task.id!r} depends on unknown id(s) {missing_deps} -- add those tasks first"
        )
    queue.tasks.append(task)
    _write_tasks(project_dir, queue)


def set_task_status(project_dir: str, task_id: str, status: str) -> None:
    if status not in _VALID_STATUSES:
        raise ContinuityError(f"unknown status {status!r}")
    queue = load_tasks(project_dir)
    task = next((t for t in queue.tasks if t.id == task_id), None)
    if task is None:
        raise ContinuityError(f"no task with id {task_id!r}")
    allowed = _TRANSITIONS.get(task.status, set())
    if status not in allowed:
        raise ContinuityError(
            f"task {task_id!r}: {task.status!r} -> {status!r} is not a legal transition "
            f"(allowed from {task.status!r}: {sorted(allowed) or 'none'})"
        )
    if status == "complete" and not task.acceptance_criteria:
        raise ContinuityError(
            f"task {task_id!r} cannot be marked complete with no acceptance_criteria"
        )
    task.status = status
    _write_tasks(project_dir, queue)


def ready_tasks(project_dir: str) -> list[Task]:
    """Every pending task whose depends_on are all complete. This is the ONLY
    place "ready" is decided -- it is never a field written to disk."""
    queue = load_tasks(project_dir)
    complete_ids = {t.id for t in queue.tasks if t.status == "complete"}
    return [
        t for t in queue.tasks
        if t.status == "pending" and all(d in complete_ids for d in t.depends_on)
    ]
```

Note: `load_tasks`'s fallback `schema_version` expression is deliberately defensive but
`storage.CURRENT_SCHEMA_VERSION` is a plain module attribute — simplify it to
`storage.CURRENT_SCHEMA_VERSION` directly (the `.__dict__.get(...)` form above is overcautious;
use the straightforward attribute access).

- [ ] **Step 4: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_tasks.py -q -p no:cacheprovider
```
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add studio/backend/core/continuity/__init__.py studio/backend/tests/test_continuity_tasks.py
git commit -m "feat(continuity): task queue with derived readiness and guarded transitions

ready_tasks computes readiness from depends_on + status on every call; there is
no status='ready' write path, so there is no way for the derived fact and a
stored one to disagree.

set_task_status enforces the transition table and refuses complete on a task
with no acceptance_criteria -- code being written is not the same as a task
being done. add_task refuses a depends_on naming an id that does not exist,
catching a typo in a hand-edited graph before ready_tasks silently misjudges it.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Demonstrate the controls (after committing)**

1. In `set_task_status`, delete the `if status not in allowed: raise ...` block. Run
   `test_illegal_transition_raises`. Expected: FAIL — `pending -> complete` now succeeds silently.
2. In `set_task_status`, delete the `if status == "complete" and not task.acceptance_criteria`
   check. Run `test_complete_requires_non_empty_acceptance_criteria`. Expected: FAIL — a
   criterion-free task is marked complete.
3. In `add_task`, delete the `missing_deps` check. Run
   `test_add_task_rejects_a_depends_on_naming_no_real_task`. Expected: FAIL — the bogus dependency
   is silently accepted.

Restore after each with `git checkout -- studio/backend/core/continuity/__init__.py`. Finish with
a clean `git status --porcelain`.

---

### Task 4: Error log — record and query prior failures

**Files:**
- Modify: `studio/backend/core/continuity/__init__.py`
- Test: `studio/backend/tests/test_continuity_errors.py`

**Interfaces:**
- Consumes: `storage.ai_dir` (Task 1)
- Produces: `continuity.record_error(project_dir, *, what_tried: str, why_failed: str, symptom: str) -> None`,
  `continuity.prior_failures(project_dir, symptom_query: str) -> list[ErrorEntry]`

This task owns the Review Focus item on `prior_failures` with a query matching nothing.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_continuity_errors.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""logs/errors.jsonl: append-only, queried by a substring match on symptom.

record_error is never-raises internally: if the primary write to errors.jsonl
fails, it falls back to appending into .remember/now.md rather than losing the
record of a failure silently. That fallback is tested here against a redirected
fallback path (monkeypatched), not the real ~/odysseus/.remember, since a test
must never touch the user's real notes.
"""

from __future__ import annotations

from core.continuity import prior_failures, record_error
from core.continuity import __init__ as continuity_module


def test_record_then_query_by_matching_symptom(tmp_path):
    record_error(str(tmp_path), what_tried = "loose regex", why_failed = "matched both brands",
                 symptom = "inert negative control")
    hits = prior_failures(str(tmp_path), "inert")
    assert len(hits) == 1
    assert hits[0].what_tried == "loose regex"


def test_query_matching_nothing_returns_empty_not_everything(tmp_path):
    record_error(str(tmp_path), what_tried = "a", why_failed = "b", symptom = "xyz")
    assert prior_failures(str(tmp_path), "this matches nothing recorded") == []


def test_multiple_entries_all_queryable(tmp_path):
    record_error(str(tmp_path), what_tried = "a", why_failed = "b", symptom = "git checkout on uncommitted work")
    record_error(str(tmp_path), what_tried = "c", why_failed = "d", symptom = "inert control")
    assert len(prior_failures(str(tmp_path), "git checkout")) == 1
    assert len(prior_failures(str(tmp_path), "inert")) == 1


def test_record_error_falls_back_when_the_primary_write_fails(tmp_path, monkeypatch):
    """A write failure must not lose the record, and must not raise into the
    caller either -- record_error is never-raises internally."""
    fallback_calls = []
    monkeypatch.setattr(
        continuity_module, "_append_fallback_note",
        lambda text: fallback_calls.append(text),
    )

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(continuity_module.storage, "write_json_atomic", _boom)
    # errors.jsonl append does not go through write_json_atomic (it is an
    # append-only file, not a replace-whole-file one) -- patch the actual append
    # primitive instead. See the implementation step below for its name.
    monkeypatch.setattr(continuity_module, "_append_error_line", _boom)

    record_error(str(tmp_path), what_tried = "x", why_failed = "y", symptom = "z")
    assert len(fallback_calls) == 1
    assert "x" in fallback_calls[0] and "y" in fallback_calls[0]
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_errors.py -q -p no:cacheprovider
```
Expected: `ImportError: cannot import name 'record_error' from 'core.continuity'`.

- [ ] **Step 3: Implement**

Append to `studio/backend/core/continuity/__init__.py`:

```python
import datetime
import json as _json

from core.continuity.schemas import ErrorEntry

_ERRORS_FILENAME = "errors.jsonl"
# The real fallback target. A test never writes here directly -- it monkeypatches
# _append_fallback_note itself, so this path is only ever touched by a human
# running the real CLI or app tool.
_REMEMBER_NOW_PATH = os.path.expanduser(r"~\odysseus\.remember\now.md")


def _errors_path(project_dir: str) -> str:
    return os.path.join(storage.ai_dir(project_dir), "logs", _ERRORS_FILENAME)


def _append_error_line(project_dir: str, entry: ErrorEntry) -> None:
    path = _errors_path(project_dir)
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "a", encoding = "utf-8") as f:
        f.write(_json.dumps(entry.to_dict()) + "\n")


def _append_fallback_note(text: str) -> None:
    """Last resort: record_error's own write failed. Losing the record of a
    failure is bad; crashing the caller over it is worse, so this is wrapped in
    its own bare except and never propagates."""
    try:
        os.makedirs(os.path.dirname(_REMEMBER_NOW_PATH), exist_ok = True)
        with open(_REMEMBER_NOW_PATH, "a", encoding = "utf-8") as f:
            f.write(text)
    except BaseException:
        pass


def record_error(project_dir: str, *, what_tried: str, why_failed: str, symptom: str) -> None:
    """Never raises. A failed primary write falls back to .remember/now.md
    rather than silently losing the record of a failure."""
    entry = ErrorEntry(
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat(),
        project = os.path.basename(os.path.normpath(project_dir)),
        what_tried = what_tried, why_failed = why_failed, symptom = symptom,
    )
    try:
        _append_error_line(project_dir, entry)
    except BaseException:
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M")
        _append_fallback_note(
            f"\n## {now} | continuity-fallback\n{what_tried} -- FAILED: {why_failed}\n"
        )


def prior_failures(project_dir: str, symptom_query: str) -> list[ErrorEntry]:
    path = _errors_path(project_dir)
    if not os.path.isfile(path):
        return []
    query = symptom_query.strip().lower()
    if not query:
        return []
    hits = []
    with open(path, encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = ErrorEntry.from_dict(_json.loads(line))
            if query in entry.symptom.lower():
                hits.append(entry)
    return hits
```

- [ ] **Step 4: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_errors.py -q -p no:cacheprovider
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add studio/backend/core/continuity/__init__.py studio/backend/tests/test_continuity_errors.py
git commit -m "feat(continuity): errors.jsonl, with a never-raises fallback

record_error is the one core function that is never-raises internally: a
failed primary write falls back to appending into .remember/now.md rather than
silently losing the record of a failure. prior_failures does a plain substring
match on symptom -- no embedding search, no new dependency -- and an empty
query returns [] rather than every entry, so a caller checking 'have we tried
this' before a risky operation gets silence instead of false reassurance.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Demonstrate the control (after committing)**

In `prior_failures`, delete the `if not query: return []` line. Run
`test_query_matching_nothing_returns_empty_not_everything` — note this test's name says
"matches nothing", so first confirm it still passes with the guard removed for THAT case, then
add a temporary assertion `assert prior_failures(str(tmp_path), "") == []` at the end of the same
test and re-run: expected FAIL with the guard removed (an empty query now returns every entry).
Restore with `git checkout -- studio/backend/core/continuity/__init__.py studio/backend/tests/test_continuity_errors.py`.
Finish with a clean `git status --porcelain`.

---

### Task 5: Checkpoints, with pruning by true age

**Files:**
- Modify: `studio/backend/core/continuity/__init__.py`
- Test: `studio/backend/tests/test_continuity_checkpoints.py`

**Interfaces:**
- Consumes: `load_state`, `storage.write_json_atomic`, `storage.ai_dir` (Tasks 1–2)
- Produces: `continuity.checkpoint(project_dir, note: str, *, retain: int = 20) -> str`

This task owns the Review Focus item on pruning the *oldest* checkpoint, not an arbitrary one.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_continuity_checkpoints.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""checkpoints/<timestamp>.json: a snapshot of project_state.json plus a note,
pruned to the most recent N. Filesystem directory order is not creation order
on every OS, so pruning must sort on a value baked into each entry, not on
os.listdir's order.
"""

from __future__ import annotations

import json
import os

from core.continuity import checkpoint, write_state
from core.continuity.schemas import ProjectState


def _state(**over):
    base = dict(schema_version = 1, project = "demo", status = "active",
                current_phase = "p", current_task = None, completion_percent = 0,
                last_checkpoint = None)
    base.update(over)
    return ProjectState(**base)


def test_checkpoint_writes_a_readable_snapshot(tmp_path):
    write_state(str(tmp_path), _state(next_action = "step one"))
    name = checkpoint(str(tmp_path), "first checkpoint")
    path = os.path.join(str(tmp_path), ".ai", "checkpoints", name)
    with open(path, encoding = "utf-8") as f:
        data = json.load(f)
    assert data["note"] == "first checkpoint"
    assert data["state"]["next_action"] == "step one"


def test_checkpoint_prunes_the_oldest_first(tmp_path):
    write_state(str(tmp_path), _state())
    names = [checkpoint(str(tmp_path), f"note {i}", retain = 3) for i in range(5)]
    remaining = set(os.listdir(os.path.join(str(tmp_path), ".ai", "checkpoints")))
    # The three most recently WRITTEN (by call order, which this test controls),
    # not the three with the "largest" filesystem mtime or alphabetically-last name.
    assert remaining == set(names[-3:]), f"expected the 3 latest, got {remaining}"
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_checkpoints.py -q -p no:cacheprovider
```
Expected: `ImportError: cannot import name 'checkpoint' from 'core.continuity'`.

- [ ] **Step 3: Implement**

Append to `studio/backend/core/continuity/__init__.py`:

```python
import time
import uuid as _uuid

_DEFAULT_CHECKPOINT_RETAIN = 20


def checkpoint(project_dir: str, note: str, *, retain: int = _DEFAULT_CHECKPOINT_RETAIN) -> str:
    """Snapshot project_state.json plus a note. Returns the written filename.
    Pruned to the RETAIN most recently written checkpoints -- ordered by a
    sequence number baked into the filename, not by os.listdir's order, which
    is not creation order on every filesystem."""
    state = load_state(project_dir)
    checkpoints_dir = os.path.join(storage.ai_dir(project_dir), "checkpoints")
    existing = sorted(
        (p for p in os.listdir(checkpoints_dir) if p.endswith(".json")),
        key = lambda p: int(p.split("-", 1)[0]),
    )
    next_seq = (int(existing[-1].split("-", 1)[0]) + 1) if existing else 0
    filename = f"{next_seq:08d}-{_uuid.uuid4().hex[:8]}.json"
    payload = {
        "seq": next_seq,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note,
        "state": state.to_dict() if state is not None else None,
    }
    storage.write_json_atomic(os.path.join(checkpoints_dir, filename), payload)

    all_now = sorted(
        (p for p in os.listdir(checkpoints_dir) if p.endswith(".json")),
        key = lambda p: int(p.split("-", 1)[0]),
    )
    if len(all_now) > retain:
        for stale in all_now[: len(all_now) - retain]:
            try:
                os.unlink(os.path.join(checkpoints_dir, stale))
            except OSError:
                pass
    return filename
```

- [ ] **Step 4: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_checkpoints.py -q -p no:cacheprovider
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add studio/backend/core/continuity/__init__.py studio/backend/tests/test_continuity_checkpoints.py
git commit -m "feat(continuity): checkpoints, pruned by a sequence number not directory order

Each filename carries a zero-padded sequence number, and pruning sorts on that
number rather than os.listdir's order -- which is not creation order on every
filesystem -- so 'the oldest N are removed' is actually true rather than
incidentally true on whichever OS the test happened to run on.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Demonstrate the control (after committing)**

**Corrected 2026-09-29, during Task 5's review — the original instruction below this line was
inert and must not be used.** The filename is an 8-digit zero-padded sequence number followed by
a random hex suffix (`f"{seq:08d}-{hex}.json"`). For distinct sequence numbers of equal width,
lexicographic string order and numeric order are mathematically identical — bare alphabetical
`sorted()` on the full filename reproduces true write order exactly, every time, because the
comparison is always decided inside the zero-padded prefix before the suffix is ever reached. A
mutation to bare `sorted(...)` therefore cannot fail regardless of whether the real guard (sort by
sequence, not directory order) is present or missing. Verified empirically by the Task 5 reviewer.

Use this mutation instead, which sorts by the random suffix rather than the sequence prefix — the
suffix is `uuid.uuid4().hex[:8]`, uncorrelated with write order, so sorting by it reliably desyncs
the result from write order:

In `checkpoint`, replace both `sorted(..., key = lambda p: int(p.split("-", 1)[0]))` calls with
`sorted(..., key = lambda p: p.split("-", 1)[1])` (sorts by the random suffix instead of the
sequence number). Run `test_checkpoint_prunes_the_oldest_first`. Expected: FAIL — pruning removes
an essentially random 2 of the 5 rather than the first 2 written. Restore with
`git checkout -- studio/backend/core/continuity/__init__.py`. Finish with a clean
`git status --porcelain`.

---

### Task 6: `validate` and `repair`

**Files:**
- Modify: `studio/backend/core/continuity/__init__.py`
- Test: `studio/backend/tests/test_continuity_validate.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5
- Produces: `continuity.validate(project_dir) -> list[str]` (each string is one problem, plain
  text — no separate `Problem` dataclass needed for v1),
  `continuity.repair(project_dir) -> list[str]` (each string is one action taken)

This task owns the Review Focus item on a dependency cycle being reported, not hung on or
silently misjudged.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_continuity_validate.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""validate() finds problems; repair() fixes only what is safe to fix
automatically and NEVER marks a task complete -- that is the one thing spec
section 28 forbids outright."""

from __future__ import annotations

import json
import os

from core.continuity import add_task, repair, set_task_status, storage, validate
from core.continuity.schemas import Task


def _task(id, **over):
    base = dict(id = id, title = f"task {id}", status = "pending", depends_on = [],
                acceptance_criteria = ["ok"], owner = None)
    base.update(over)
    return Task(**base)


def test_validate_clean_project_reports_nothing(tmp_path):
    add_task(str(tmp_path), _task("t1"))
    assert validate(str(tmp_path)) == []


def test_validate_reports_a_dependency_cycle(tmp_path):
    """Review Focus: A depends on B, B depends on A. Must be reported, not
    hang, and not silently leave both permanently un-ready with no explanation."""
    add_task(str(tmp_path), _task("t1"))
    add_task(str(tmp_path), _task("t2", depends_on = ["t1"]))
    # Hand-edit the queue to create the cycle -- add_task's own validation
    # (Task 3) would refuse to create one through the public API, so the test
    # writes the file directly to simulate a hand-edited or corrupted graph.
    path = os.path.join(str(tmp_path), ".ai", "task_queue.json")
    with open(path, encoding = "utf-8") as f:
        data = json.load(f)
    for t in data["tasks"]:
        if t["id"] == "t1":
            t["depends_on"] = ["t2"]
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(data, f)

    problems = validate(str(tmp_path))
    assert any("cycle" in p.lower() for p in problems), problems


def test_validate_reports_corrupt_json(tmp_path):
    storage.ai_dir(str(tmp_path))
    path = os.path.join(str(tmp_path), ".ai", "task_queue.json")
    with open(path, "w", encoding = "utf-8") as f:
        f.write("{not json")
    problems = validate(str(tmp_path))
    assert any("task_queue.json" in p for p in problems), problems


def test_repair_never_completes_a_task(tmp_path):
    add_task(str(tmp_path), _task("t1"))
    set_task_status(str(tmp_path), "t1", "in_progress")
    repair(str(tmp_path))
    from core.continuity import load_tasks
    reloaded = load_tasks(str(tmp_path))
    assert next(t for t in reloaded.tasks if t.id == "t1").status == "in_progress"


def test_repair_regenerates_the_rendered_files(tmp_path):
    add_task(str(tmp_path), _task("t1"))
    report = repair(str(tmp_path))
    assert any("project_state.md" in r or "context_summary" in r for r in report)
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_validate.py -q -p no:cacheprovider
```
Expected: `ImportError: cannot import name 'validate' from 'core.continuity'`.

- [ ] **Step 3: Implement**

Append to `studio/backend/core/continuity/__init__.py`:

```python
def _find_cycle(tasks: list[Task]) -> list[str] | None:
    """DFS cycle detection over depends_on edges. Returns the cycle's task ids,
    or None. Bounded by len(tasks) recursion depth, which this project's task
    graphs are nowhere near large enough to make a concern."""
    by_id = {t.id: t for t in tasks}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {t.id: WHITE for t in tasks}
    stack: list[str] = []

    def visit(tid: str) -> list[str] | None:
        color[tid] = GRAY
        stack.append(tid)
        for dep in by_id.get(tid, Task(id = tid, title = "", status = "")).depends_on:
            if dep not in by_id:
                continue  # unknown ids are validate's own separate problem
            if color.get(dep) == GRAY:
                cycle_start = stack.index(dep)
                return stack[cycle_start:] + [dep]
            if color.get(dep) == WHITE:
                found = visit(dep)
                if found:
                    return found
        stack.pop()
        color[tid] = BLACK
        return None

    for t in tasks:
        if color[t.id] == WHITE:
            found = visit(t.id)
            if found:
                return found
    return None


def validate(project_dir: str) -> list[str]:
    """Read-only. Never raises -- a validator that can crash on the exact data
    it exists to check is not trustworthy. Problems are returned as plain
    strings; there is no downstream consumer yet that needs structure richer
    than 'read this to a human'."""
    problems: list[str] = []

    try:
        state = load_state(project_dir)
    except ContinuityError as exc:
        problems.append(f"project_state.json: {exc}")
        state = None

    try:
        tasks = load_tasks(project_dir).tasks
    except ContinuityError as exc:
        problems.append(f"task_queue.json: {exc}")
        tasks = []

    ids = [t.id for t in tasks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        problems.append(f"duplicate task id(s): {sorted(dupes)}")

    known = set(ids)
    for t in tasks:
        for dep in t.depends_on:
            if dep not in known:
                problems.append(f"task {t.id!r} depends on unknown id {dep!r}")

    cycle = _find_cycle(tasks)
    if cycle:
        problems.append(f"dependency cycle: {' -> '.join(cycle)}")

    if state is not None and state.current_task and state.current_task not in known:
        problems.append(
            f"project_state.json names current_task {state.current_task!r}, "
            "which is not in task_queue.json"
        )

    return problems


def repair(project_dir: str) -> list[str]:
    """Regenerates ONLY derived data: the rendered .md files. Never touches a
    task's status, never adds an acceptance criterion, never marks anything
    complete -- see spec section 28. Readiness itself needs no repair, since it
    was never stored (Task 3)."""
    from core.continuity import render

    actions: list[str] = []
    summary = render.render_context_summary(project_dir)
    storage.write_json_atomic  # no-op reference to keep import used if reshuffled
    _write_text(os.path.join(storage.ai_dir(project_dir), "context_summary.md"), summary)
    actions.append("regenerated context_summary.md")

    state_md = render.render_project_state_md(project_dir)
    _write_text(os.path.join(storage.ai_dir(project_dir), "project_state.md"), state_md)
    actions.append("regenerated project_state.md")

    return actions


def _write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding = "utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
```

Delete the stray `storage.write_json_atomic  # no-op reference...` line before running tests —
it was left in the draft above by mistake and does nothing; `repair` does not need it.

This task's `repair` calls into `render.render_context_summary` / `render.render_project_state_md`,
which do not exist until Task 7. **Do not implement Task 7 here** — instead, stub `render.py` with
just enough to make Task 6's tests pass, and let Task 7 replace the stub with the real renderer:

Create `studio/backend/core/continuity/render.py` (stub, replaced in Task 7):

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stub -- replaced in the next task with the real renderer and its token cap."""


def render_context_summary(project_dir: str) -> str:
    return f"stub summary for {project_dir}"


def render_project_state_md(project_dir: str) -> str:
    return f"stub state.md for {project_dir}"
```

- [ ] **Step 4: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_validate.py -q -p no:cacheprovider
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add studio/backend/core/continuity/__init__.py studio/backend/core/continuity/render.py studio/backend/tests/test_continuity_validate.py
git commit -m "feat(continuity): validate() and repair()

validate finds duplicate ids, unknown dependency targets, dependency cycles
(DFS over depends_on, reported as the actual cycle rather than a silent
never-ready state), and a current_task that names nothing in the queue.

repair regenerates ONLY the rendered .md files -- it does not, and structurally
cannot, touch a task's status. render.py is a stub here; the next task replaces
it with the real renderer and its token cap.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Demonstrate the control (after committing)**

In `_find_cycle`, change `if color.get(dep) == GRAY:` to `if False:` (disabling cycle detection).
Run `test_validate_reports_a_dependency_cycle`. Expected: FAIL — the cycle is present but
unreported. Restore with `git checkout -- studio/backend/core/continuity/__init__.py`. Finish
with a clean `git status --porcelain`.

---

### Task 7: Rendering — `context_summary.md` and `project_state.md`, with the token cap

**Files:**
- Modify: `studio/backend/core/continuity/render.py`
- Test: `studio/backend/tests/test_continuity_render.py`

**Interfaces:**
- Consumes: `load_state`, `load_tasks`, `ready_tasks` (Tasks 2–3)
- Produces: `render.render_context_summary(project_dir) -> str`,
  `render.render_project_state_md(project_dir) -> str`

This task owns the spec's token-cap requirement (§21, §6) precisely — target 300–800 tokens,
hard cap ~1500, `ContinuityError` if it cannot fit after truncation.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_continuity_render.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""context_summary.md must fit its cap -- that is the whole point of a
context-reset summary: a caller must be able to trust it never blows the
budget it exists to protect. Token count is approximated as len(text) // 4
(documented in render.py's docstring); no tokenizer dependency."""

from __future__ import annotations

import pytest

from core.continuity import add_task, checkpoint, update_state, write_state
from core.continuity.render import _approx_tokens, render_context_summary, render_project_state_md
from core.continuity.schemas import ContinuityError, ProjectState, Task


def _state(**over):
    base = dict(schema_version = 1, project = "demo", status = "active",
                current_phase = "p1", current_task = None, completion_percent = 10,
                last_checkpoint = None, next_action = "do the next thing")
    base.update(over)
    return ProjectState(**base)


def test_render_context_summary_on_empty_project_is_short(tmp_path):
    write_state(str(tmp_path), _state())
    summary = render_context_summary(str(tmp_path))
    assert _approx_tokens(summary) <= 1500
    assert "demo" in summary


def test_render_context_summary_includes_next_action_and_blockers(tmp_path):
    write_state(str(tmp_path), _state(
        next_action = "fix the thing", blocking_issues = ["waiting on X"],
    ))
    summary = render_context_summary(str(tmp_path))
    assert "fix the thing" in summary
    assert "waiting on X" in summary


def test_render_context_summary_caps_even_with_a_huge_blocking_issues_list(tmp_path):
    huge = [f"blocker number {i} with a fair bit of explanatory text attached" for i in range(500)]
    write_state(str(tmp_path), _state(blocking_issues = huge))
    summary = render_context_summary(str(tmp_path))
    assert _approx_tokens(summary) <= 1500


def test_render_project_state_md_lists_tasks(tmp_path):
    write_state(str(tmp_path), _state())
    add_task(str(tmp_path), Task(id = "t1", title = "the first task", status = "pending",
                                  acceptance_criteria = ["ok"]))
    md = render_project_state_md(str(tmp_path))
    assert "the first task" in md
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_render.py -q -p no:cacheprovider
```
Expected: `ImportError: cannot import name '_approx_tokens' from 'core.continuity.render'` (the
stub from Task 6 has neither this name nor real content).

- [ ] **Step 3: Implement**

Replace `studio/backend/core/continuity/render.py` entirely:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Renders project_state.json + task_queue.json into the two files a human (or
a fresh context) reads first.

Both outputs are GENERATED, never hand-authored: MEMORY.md already does
index-first progressive disclosure for durable cross-session facts, so these
render from the structured state rather than becoming a second thing someone
has to remember to update by hand.

Token counting is approximate: len(text) // 4, the common rule-of-thumb ratio
for English text. This is not a real tokenizer and is not meant to be exact --
it exists only to keep render_context_summary inside its budget, and the cap
is generous enough (1500) that the approximation's error does not matter.
"""

from __future__ import annotations

from core.continuity.schemas import ContinuityError

_TARGET_TOKENS = 800
_HARD_CAP_TOKENS = 1500


def _approx_tokens(text: str) -> int:
    return len(text) // 4


def render_context_summary(project_dir: str) -> str:
    """~300-800 tokens target, 1500 hard cap. Raises ContinuityError rather
    than silently returning an oversized string if it cannot fit even after
    truncating the lists -- a context-reset summary that can silently blow its
    own budget has failed at the one thing it exists to guarantee."""
    from core.continuity import load_state, load_tasks, ready_tasks

    state = load_state(project_dir)
    if state is None:
        return "No project state recorded yet (.ai/project_state.json does not exist)."

    ready = ready_tasks(project_dir)
    queue = load_tasks(project_dir)
    in_progress = [t for t in queue.tasks if t.status == "in_progress"]

    def _fmt_list(items: list[str], limit: int) -> str:
        shown = items[:limit]
        text = "\n".join(f"- {i}" for i in shown)
        if len(items) > limit:
            text += f"\n- ... and {len(items) - limit} more"
        return text or "(none)"

    limit = 10
    while True:
        lines = [
            f"# {state.project} -- {state.status}",
            f"Phase: {state.current_phase}",
            f"Current task: {state.current_task or '(none)'}",
            f"Completion: {state.completion_percent}%",
            "",
            "## Next action",
            state.next_action or "(not recorded)",
            "",
            "## In progress",
            _fmt_list([t.id for t in in_progress], limit),
            "",
            "## Ready to start",
            _fmt_list([t.id for t in ready], limit),
            "",
            "## Blocking issues",
            _fmt_list(state.blocking_issues, limit),
        ]
        summary = "\n".join(lines)
        if _approx_tokens(summary) <= _HARD_CAP_TOKENS:
            return summary
        if limit <= 1:
            raise ContinuityError(
                f"context summary for {state.project!r} exceeds the {_HARD_CAP_TOKENS}-token "
                "cap even at minimum truncation -- the underlying state is too large to summarize "
                "safely; trim blocking_issues or the task queue directly"
            )
        limit = max(1, limit // 2)


def render_project_state_md(project_dir: str) -> str:
    """A fuller, human-readable render -- no token cap, this is the level-2
    disclosure in the spec's progressive-disclosure model, read deliberately
    rather than loaded automatically."""
    from core.continuity import load_state, load_tasks

    state = load_state(project_dir)
    queue = load_tasks(project_dir)
    lines = ["# Project State", ""]
    if state is None:
        lines.append("(no project_state.json recorded yet)")
    else:
        lines += [
            f"**Project:** {state.project}",
            f"**Status:** {state.status}",
            f"**Phase:** {state.current_phase}",
            f"**Completion:** {state.completion_percent}%",
            f"**Next action:** {state.next_action or '(not recorded)'}",
            "",
            "## Modified files",
            *[f"- {p}" for p in state.modified_files] or ["(none recorded)"],
            "",
            "## Blocking issues",
            *[f"- {b}" for b in state.blocking_issues] or ["(none)"],
        ]
    lines += ["", "## Tasks"]
    for t in queue.tasks:
        deps = f" (depends on {', '.join(t.depends_on)})" if t.depends_on else ""
        lines.append(f"- [{t.status}] {t.id}: {t.title}{deps}")
    if not queue.tasks:
        lines.append("(no tasks recorded)")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_render.py tests/test_continuity_validate.py -q -p no:cacheprovider
```
Expected: 9 passed (5 render + the 5 from Task 6, now against the real renderer instead of the
stub — `test_repair_regenerates_the_rendered_files` must still pass unchanged).

- [ ] **Step 5: Commit**

```bash
git add studio/backend/core/continuity/render.py
git commit -m "feat(continuity): render context_summary.md and project_state.md, with a real token cap

context_summary progressively truncates its lists (10 items, then halved) until
it fits the 1500-token approximate cap, and raises ContinuityError rather than
silently exceeding it if even minimum truncation cannot make it fit. Both files
are generated from project_state.json + task_queue.json, never hand-authored --
the reconciliation this whole piece is built around.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Demonstrate the control (after committing)**

In `render_context_summary`, change `if _approx_tokens(summary) <= _HARD_CAP_TOKENS: return summary`
to `return summary` unconditionally (skip the cap check entirely). Run
`test_render_context_summary_caps_even_with_a_huge_blocking_issues_list`. Expected: FAIL — the
500-item blocking-issues list is rendered in full, well past 1500 tokens. Restore with
`git checkout -- studio/backend/core/continuity/render.py`. Finish with a clean
`git status --porcelain`.

---

### Task 8: Path confinement for the app-side caller

**Files:**
- Create: `studio/backend/core/continuity/sandbox.py`
- Test: `studio/backend/tests/test_continuity_sandbox.py`

**Interfaces:**
- Consumes: `core.inference.tools._get_workdir`, `core.inference.tools._is_outside_workdir`
  (existing, do not modify)
- Produces: `sandbox.confine_project_dir(session_id: str | None) -> str` — returns the sandbox
  workdir for that session, confined the same way `resolve_image_bytes` confines an image path.
  For this piece, the "project" IS the sandbox workdir itself — `continuity_task` does not take a
  project path argument; it always operates on the calling conversation's own workdir. This
  function exists to make that one decision in one place rather than inline in the tool handler.

This task owns the spec's path-confinement requirement (§7) — separated from the tool itself
(Task 9) so it can be tested without touching `tools.py`, `execute_tool`, or the tool-audit shadow.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_continuity_sandbox.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""continuity_task never takes a project_dir argument from the model -- it is
always the calling session's own sandbox workdir, resolved the same way
resolve_image_bytes confines an image path (core/inference/assist_vision/paths.py).
"""

from __future__ import annotations

from core.continuity.sandbox import confine_project_dir


def test_confine_project_dir_returns_a_real_directory(tmp_path, monkeypatch):
    import core.inference.tools as tools_module
    monkeypatch.setattr(tools_module, "_get_workdir", lambda session_id: str(tmp_path))
    result = confine_project_dir("some-session")
    assert result == str(tmp_path)


def test_confine_project_dir_works_with_no_session_id(tmp_path, monkeypatch):
    """_get_workdir is None-safe by Studio's own design (anonymous sandbox) --
    there must be no session_id-omitted escape hatch."""
    import core.inference.tools as tools_module
    monkeypatch.setattr(tools_module, "_get_workdir", lambda session_id: str(tmp_path))
    assert confine_project_dir(None) == str(tmp_path)
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_sandbox.py -q -p no:cacheprovider
```
Expected: `ModuleNotFoundError: No module named 'core.continuity.sandbox'`.

- [ ] **Step 3: Implement**

Create `studio/backend/core/continuity/sandbox.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where continuity_task is allowed to read and write, for a given session.

For this tool the "project" IS the conversation's own sandbox workdir -- there
is no separate project_dir argument for a model to name, which is what makes a
path-escape attempt structurally impossible rather than merely checked for.
Reuses core.inference.tools._get_workdir exactly as
core/inference/assist_vision/paths.py:resolve_image_bytes does; this module
does not invent a second confinement policy.

Imported lazily inside the function, matching paths.py's own stated reason:
tools.py is large and will eventually import continuity back (to register
continuity_task), so a module-level import here would be circular.
"""

from __future__ import annotations


def confine_project_dir(session_id: str | None) -> str:
    """The sandbox workdir continuity_task operates on for this session.
    _get_workdir is None-safe (session_id or an anonymous key), so there is no
    session_id-omitted escape from confinement."""
    from core.inference import tools as _tools

    return _tools._get_workdir(session_id)
```

- [ ] **Step 4: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_sandbox.py -q -p no:cacheprovider
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add studio/backend/core/continuity/sandbox.py studio/backend/tests/test_continuity_sandbox.py
git commit -m "feat(continuity): confine the app tool to the session's own sandbox workdir

continuity_task takes no project_dir argument at all -- it always operates on
the calling conversation's sandbox workdir, reusing _get_workdir exactly as
resolve_image_bytes does. No new confinement policy; the escape this closes is
structural (there is nothing to escape TO) rather than merely checked for.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

No mutation control for this task: it has no guard to disable (there is no bypass branch to
delete) — it is a thin wrapper whose only behaviour is calling the existing, already-tested
`_get_workdir`. Confirm `git status --porcelain` is clean before moving on.

---

### Task 9: The `continuity_task` tool — schema, handler, never-raises boundary

**Files:**
- Create: `studio/backend/core/continuity/schemas_tool.py` (the OpenAI-style tool schema,
  separate from `schemas.py`'s dataclasses to match the naming split `delegation/schemas.py`
  already uses for its tool schema vs. `delegation/subagent.py`'s internal dataclasses — but here
  both live under `continuity/`, so name this file distinctly)
- Modify: `studio/backend/core/continuity/__init__.py` (add `execute`)
- Test: `studio/backend/tests/test_continuity_tool.py`

**Interfaces:**
- Consumes: everything from Tasks 1–8
- Produces: `continuity.execute(name: str, arguments: dict, *, session_id: str | None = None, **kwargs) -> str`,
  `schemas_tool.CONTINUITY_TOOLS: list[dict]`, `schemas_tool.CONTINUITY_TOOL_NAMES: frozenset[str]`

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_continuity_tool.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The continuity_task tool handler: never raises, dispatches to the six
actions, and is confined to the session's own sandbox workdir. No test here
loads a model, starts a backend, or calls a real tool -- execute() is driven
directly, the way test_delegation_tool.py drives delegation.execute directly.
"""

from __future__ import annotations

import pytest

from core.continuity import execute
from core.continuity.schemas_tool import CONTINUITY_TOOL_NAMES, CONTINUITY_TOOLS


@pytest.fixture(autouse = True)
def _sandbox(tmp_path, monkeypatch):
    import core.inference.tools as tools_module
    monkeypatch.setattr(tools_module, "_get_workdir", lambda session_id: str(tmp_path))
    yield


def test_the_schema_is_shaped_like_the_other_tools():
    function = CONTINUITY_TOOLS[0]["function"]
    assert function["name"] == "continuity_task"
    assert CONTINUITY_TOOL_NAMES == frozenset({"continuity_task"})
    # project_dir must never be a parameter a model can supply -- confinement
    # is structural (Task 8), not a value the tool call carries.
    assert "project_dir" not in function["parameters"]["properties"]


def test_status_on_an_uninitialized_sandbox_says_so_rather_than_erroring():
    out = execute("continuity_task", {"action": "status"}, session_id = "s1")
    assert isinstance(out, str)
    assert "No project state" in out or "not recorded" in out


def test_add_task_then_status_round_trips():
    out = execute("continuity_task",
                   {"action": "add_task", "id": "t1", "title": "do the thing",
                    "acceptance_criteria": ["it works"]},
                   session_id = "s1")
    assert "t1" in out
    status_out = execute("continuity_task", {"action": "status"}, session_id = "s1")
    assert "t1" in status_out


def test_an_unknown_action_returns_an_error_string_not_a_raise():
    out = execute("continuity_task", {"action": "not_a_real_action"}, session_id = "s1")
    assert isinstance(out, str)
    assert "Error" in out or "error" in out


def test_a_core_error_is_caught_and_returned_as_a_string_not_raised():
    """The never-raises boundary. set_task_status on a missing id raises
    ContinuityError inside the core (Task 3); the tool handler must catch it."""
    out = execute("continuity_task",
                   {"action": "set_status", "id": "does-not-exist", "status": "in_progress"},
                   session_id = "s1")
    assert isinstance(out, str)
    assert "Error" in out or "error" in out


def test_two_sessions_do_not_see_each_others_tasks(tmp_path, monkeypatch):
    import core.inference.tools as tools_module
    workdirs = {"s1": str(tmp_path / "s1"), "s2": str(tmp_path / "s2")}
    monkeypatch.setattr(tools_module, "_get_workdir", lambda session_id: workdirs[session_id])
    execute("continuity_task", {"action": "add_task", "id": "t1", "title": "x",
                                 "acceptance_criteria": ["y"]}, session_id = "s1")
    out = execute("continuity_task", {"action": "status"}, session_id = "s2")
    assert "t1" not in out
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_tool.py -q -p no:cacheprovider
```
Expected: `ModuleNotFoundError: No module named 'core.continuity.schemas_tool'`.

- [ ] **Step 3: Create the schema**

Create `studio/backend/core/continuity/schemas_tool.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI-style schema for continuity_task.

Named schemas_tool.py, not schemas.py, because that name is already the
dataclasses module in this package (ProjectState, Task, ...) -- delegation
keeps its tool schema and its internal dataclasses in separate files too, just
under different names (delegation/schemas.py is the tool schema there, and
delegation/subagent.py holds its own internal SubagentResult dataclass). This
package's split is the other way around for a reason worth stating: schemas.py
was written first (Task 1) and named for the data shapes, so the tool schema
gets the more specific name rather than forcing a rename across every earlier
task's imports.
"""

CONTINUITY_TASK_TOOL = {
    "type": "function",
    "function": {
        "name": "continuity_task",
        "description": (
            "Track your own progress on a multi-step task so it survives a context reset: "
            "record what you tried and whether it failed, add tasks with dependencies, mark "
            "them in progress or done, and check what state you're in. Always scoped to this "
            "conversation's own files -- there is no separate project to name. Prefer this over "
            "re-deriving status by re-reading the conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "status", "add_task", "set_status",
                        "record_error", "check_prior_failures", "checkpoint",
                    ],
                    "description": "Which operation to perform.",
                },
                "id": {"type": "string", "description": "Task id (add_task, set_status)."},
                "title": {"type": "string", "description": "Task title (add_task)."},
                "status": {
                    "type": "string",
                    "description": "New status (set_status): in_progress, complete, failed, blocked, abandoned, pending. ('ready' is never set explicitly -- it is computed; use status action to see which pending tasks are unblocked.)",
                },
                "depends_on": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Task ids this one depends on (add_task, optional).",
                },
                "acceptance_criteria": {
                    "type": "array", "items": {"type": "string"},
                    "description": "What 'done' means for this task (add_task). Required to later mark it complete.",
                },
                "what_tried": {"type": "string", "description": "What was attempted (record_error)."},
                "why_failed": {"type": "string", "description": "Why it failed (record_error)."},
                "symptom": {
                    "type": "string",
                    "description": "Short tag for later matching (record_error, check_prior_failures).",
                },
                "note": {"type": "string", "description": "A note for this checkpoint (checkpoint)."},
            },
            "required": ["action"],
        },
    },
}

CONTINUITY_TOOLS = [CONTINUITY_TASK_TOOL]
CONTINUITY_TOOL_NAMES = frozenset({"continuity_task"})
```

- [ ] **Step 4: Implement the handler**

Append to `studio/backend/core/continuity/__init__.py`:

```python
def execute(name: str, arguments, *, session_id: str | None = None, **kwargs) -> str:
    """Handler for continuity_task. Never raises -- a continuity read/write
    failure must degrade the turn, not break it, the same never-raises
    boundary every other built-in tool in this fork keeps at its execute()."""
    try:
        return _execute(arguments, session_id = session_id)
    except BaseException as exc:  # noqa: BLE001 - a continuity failure must not break the turn
        return f"Error: continuity_task failed: {exc}"


def _execute(arguments, *, session_id: str | None) -> str:
    from core.continuity.sandbox import confine_project_dir

    args = arguments if isinstance(arguments, dict) else {}
    action = str(args.get("action") or "").strip()
    project_dir = confine_project_dir(session_id)

    if action == "status":
        from core.continuity.render import render_context_summary
        return render_context_summary(project_dir)

    if action == "add_task":
        task_id = str(args.get("id") or "").strip()
        if not task_id:
            return "Error: add_task needs an id."
        add_task(project_dir, Task(
            id = task_id, title = str(args.get("title") or task_id), status = "pending",
            depends_on = list(args.get("depends_on") or []),
            acceptance_criteria = list(args.get("acceptance_criteria") or []),
        ))
        return f"Added task {task_id!r}."

    if action == "set_status":
        task_id = str(args.get("id") or "").strip()
        status = str(args.get("status") or "").strip()
        if not task_id or not status:
            return "Error: set_status needs an id and a status."
        set_task_status(project_dir, task_id, status)
        return f"Task {task_id!r} is now {status!r}."

    if action == "record_error":
        record_error(
            project_dir,
            what_tried = str(args.get("what_tried") or ""),
            why_failed = str(args.get("why_failed") or ""),
            symptom = str(args.get("symptom") or ""),
        )
        return "Recorded."

    if action == "check_prior_failures":
        hits = prior_failures(project_dir, str(args.get("symptom") or ""))
        if not hits:
            return "No prior recorded failures match that."
        return "\n".join(f"- {h.what_tried} -- {h.why_failed}" for h in hits)

    if action == "checkpoint":
        name = checkpoint(project_dir, str(args.get("note") or ""))
        return f"Checkpointed as {name}."

    return f"Error: unknown continuity_task action {action!r}."
```

- [ ] **Step 5: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_tool.py -q -p no:cacheprovider
```
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add studio/backend/core/continuity/schemas_tool.py studio/backend/core/continuity/__init__.py studio/backend/tests/test_continuity_tool.py
git commit -m "feat(continuity): the continuity_task tool, never-raises, sandbox-confined

execute() wraps every action in a bare except BaseException, the same
never-raises boundary every built-in tool in this fork keeps -- a continuity
failure degrades the turn, it does not break it. project_dir is never a tool
parameter (see schemas_tool.py's parameters block): the model cannot name a
path, because confine_project_dir (Task 8) always resolves the calling
session's own sandbox workdir.

Not yet registered anywhere -- ALL_TOOLS, execute_tool's dispatch, the
_addable re-add, and both safe-tool lists are the next task.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Demonstrate the control (after committing)**

In `execute`, change `except BaseException as exc:` to `except ValueError as exc:` (narrowing
it). Then temporarily make `_execute` raise something else — in `add_task`'s branch, change
`add_task(project_dir, Task(...))` to `raise RuntimeError("simulated")` right before that line.
Run `test_add_task_then_status_round_trips`. Expected: FAIL — the test crashes with an unhandled
`RuntimeError` instead of getting an error string back, proving the never-raises boundary was
actually doing work. Restore with
`git checkout -- studio/backend/core/continuity/__init__.py`. Finish with a clean
`git status --porcelain`.

---

### Task 10: Registration — `continuity_task` reachable in Studio, and the seam ceilings raised together

**Files:**
- Modify: `studio/backend/core/inference/tools.py` (additive only)
- Modify: `studio/backend/routes/inference.py` (additive only)
- Modify: `studio/backend/tests/test_delegation_seam.py`, `studio/backend/tests/test_tool_audit_seam.py`,
  `studio/backend/tests/test_tool_readiness_seam.py` (raise the ceiling in all three, together)
- Test: `studio/backend/tests/test_continuity_seam.py`

**Interfaces:**
- Consumes: `schemas_tool.CONTINUITY_TOOLS`, `schemas_tool.CONTINUITY_TOOL_NAMES` (Task 9);
  `continuity.execute` (Task 9)
- Produces: `continuity_task` reachable through the real Studio route, on both safe-tool lists,
  reachable through `execute_tool`'s dispatch

**Before touching anything, record the current seam numbers** (this project's standing
discipline — a plan that quotes stale numbers has caused real defects before):

```
git diff --numstat $(git merge-base origin/main HEAD) -- studio/backend/core/inference/tools.py
git diff --numstat $(git merge-base origin/main HEAD) -- studio/backend/routes/inference.py
```

Confirm these read `114 0` and `60 0` (the values in this plan's Global Constraints) before
proceeding — if they differ, something landed on `unslothed-release` since this plan was written
and every ceiling number below needs recomputing against the real baseline, not this plan's.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_continuity_seam.py` — this is a close structural copy of
`tests/test_delegation_seam.py`, adapted for `continuity_task`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Registration, asserted by behaviour, not by "is it in ALL_TOOLS" -- that
alone has passed while a tool was unreachable in Studio three times in this
project's history. Reachability is driven through the REAL
_select_request_tools, exactly as test_delegation_seam.py does.
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


def test_continuity_task_is_registered():
    assert "continuity_task" in {t["function"]["name"] for t in tools.ALL_TOOLS}


def test_continuity_task_reaches_the_model_through_the_real_route():
    """continuity_task has no UI pill, exactly like check_tool_readiness and
    find_capability -- without the _addable re-add it would be registered,
    dispatched, safe-listed, fully tested, and unreachable in Studio, which
    has happened three times in this project already."""
    assert "continuity_task" in _select(_payload())


def test_execute_tool_dispatches_to_the_continuity_handler(tmp_path, monkeypatch):
    import core.inference.tools as tools_module
    monkeypatch.setattr(tools_module, "_get_workdir", lambda session_id: str(tmp_path))
    out = tools.execute_tool("continuity_task", {"action": "status"})
    assert "No project state" in out or "not recorded" in out


def test_continuity_task_is_always_safe():
    """Unlike ask_model, this reads and writes only local sandboxed files --
    no network, no code execution -- so it belongs on the safe list rather
    than needing a confirmation prompt."""
    assert tools.is_high_risk_tool_call("continuity_task", {"action": "status"}) is False


def test_continuity_task_is_present_in_the_anthropic_unprompted_set():
    inference = pytest.importorskip("routes.inference", reason = "inference stack not installed")
    assert "continuity_task" in inference._ANTHROPIC_UNPROMPTED_SAFE_TOOLS


def test_the_seams_stay_additive():
    repo = Path(tools.__file__).parents[4]
    base = subprocess.run(["git", "merge-base", "origin/main", "HEAD"], cwd = repo,
                          capture_output = True, text = True, check = True).stdout.strip()
    for path, ceiling in (
        ("studio/backend/core/inference/tools.py", 140),
        ("studio/backend/routes/inference.py", 80),
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
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_seam.py -q -p no:cacheprovider
```
Expected: several failures — `continuity_task` is not in `ALL_TOOLS` yet, and `main.py`'s current
count (6) is already under the new ceiling (10) so that one line may pass early; the others fail.

- [ ] **Step 3: Register in `tools.py`**

Two additive edits. **Do not modify or re-indent any existing line.**

(a) Directly after the delegation schema import (find
`from core.inference.delegation.schemas import DELEGATION_TOOLS, DELEGATION_TOOL_NAMES`), add:

```python
from core.inference.continuity.schemas_tool import CONTINUITY_TOOLS, CONTINUITY_TOOL_NAMES
```

Wait — check the actual import path first: Task 1 placed the package at
`studio/backend/core/continuity/`, a sibling of `core/inference/`, **not** nested under
`core/inference/`. So the import is:

```python
from core.continuity.schemas_tool import CONTINUITY_TOOLS, CONTINUITY_TOOL_NAMES
```

Confirm this against how Task 1 actually placed the package (run
`ls studio/backend/core/continuity/` and check it is a sibling of `inference/`, not inside it)
before writing this line — the two constraint blocks above both assumed the sibling placement;
get the import path from the real directory layout, not from memory of this plan.

(b) In `ALL_TOOLS`, directly after `    *DELEGATION_TOOLS,`, add:

```python
    *CONTINUITY_TOOLS,
```

(c) In `execute_tool`, directly after the delegation dispatch block (after its closing `)` for
the `delegation.execute(...)` call), add:

```python
    if name in CONTINUITY_TOOL_NAMES:
        from core.continuity import execute as _continuity_execute
        return _continuity_execute(name, arguments, session_id = session_id)
```

(d) Rebind `_ALWAYS_SAFE_TOOLS` additively, directly after delegation's own rebind if one exists,
or after the `check_tool_readiness` / `find_capability` rebind at line ~4541 otherwise — **do not
edit the existing rebind line; add a new one**:

```python
# fork: additive rebind. continuity_task reads and writes only local sandboxed
# files (core/continuity/), no network, no code execution -- unlike ask_model,
# which is deliberately NOT on this list because it loads models and can edit
# files through its delegate.
_ALWAYS_SAFE_TOOLS = _ALWAYS_SAFE_TOOLS | frozenset({"continuity_task"})
```

- [ ] **Step 4: Register in `routes/inference.py`**

Two edits, both additive.

(a) In the `_addable` block's import (the `from core.inference.tools import (...)` inside
`_select_request_tools`), directly after `DELEGATION_TOOL_NAMES,`, add:

```python
            CONTINUITY_TOOL_NAMES,
```

and extend the union — **keeping it on ONE line**, because `test_tool_readiness_seam.py` and this
task's own `_addable`-driven test both rely on the route's real behaviour, and past experience in
this project (`_route_addable()` in the readiness seam test) shows a wrapped multi-line union has
broken a test that parses the line before:

```python
        _addable = ASSIST_VISION_TOOL_NAMES | ASSIST_CODE_TOOL_NAMES | READINESS_TOOL_NAMES | CAPABILITY_TOOL_NAMES | DELEGATION_TOOL_NAMES | CONTINUITY_TOOL_NAMES
```

(b) Rebind `_ANTHROPIC_UNPROMPTED_SAFE_TOOLS` additively, directly after the existing
`check_tool_readiness` / `find_capability` rebind — **add a new rebind, do not edit the
existing one**:

```python
# fork: additive rebind, same reasoning as tools._ALWAYS_SAFE_TOOLS above --
# continuity_task only reads/writes local sandboxed files.
_ANTHROPIC_UNPROMPTED_SAFE_TOOLS = _ANTHROPIC_UNPROMPTED_SAFE_TOOLS | frozenset(
    {"continuity_task"}
)
```

**Do not touch `main.py`.** This piece adds no HTTP router.

- [ ] **Step 5: Run this task's tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_seam.py -q -p no:cacheprovider
```
Expected: 6 passed. If `test_the_seams_stay_additive` still fails on a ceiling, re-measure the
real insertions with the `git diff --numstat` commands above and adjust ceiling numbers in THIS
task's test only for now (Step 6 raises the other three files' ceilings to match).

- [ ] **Step 6: Raise the other three seam-ceiling tests, together**

Re-run the numstat commands from before Step 1 to get the real final counts for `tools.py` and
`routes/inference.py`. In `tests/test_delegation_seam.py`, `tests/test_tool_audit_seam.py`, and
`tests/test_tool_readiness_seam.py`, update each file's `insertions <= N` ceiling to a round
number comfortably above the new measured count (matching the existing style — see
`tests/test_delegation_seam.py`'s own ceiling comment for the pattern: state the measured number
in a comment, set the ceiling a bit above it, and note that three files must move together). Do
**not** touch any `deletions == 0` assertion in any of the three files — that half is load-bearing
and unrelated to this change.

- [ ] **Step 7: Run every seam test together**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_seam.py tests/test_delegation_seam.py tests/test_tool_audit_seam.py tests/test_tool_readiness_seam.py -q -p no:cacheprovider
```
Expected: all pass, all four agreeing on the same real insertion counts.

- [ ] **Step 8: Run the full continuity suite plus the adjacent suites this touches**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_storage.py tests/test_continuity_state.py tests/test_continuity_tasks.py tests/test_continuity_errors.py tests/test_continuity_checkpoints.py tests/test_continuity_validate.py tests/test_continuity_render.py tests/test_continuity_sandbox.py tests/test_continuity_tool.py tests/test_continuity_seam.py tests/test_delegation_seam.py tests/test_delegation_tool.py tests/test_tool_audit_seam.py tests/test_tool_readiness_seam.py tests/test_tool_readiness_probes.py tests/test_capability_map_seam.py -q -p no:cacheprovider
```
Expected: all pass. If anything outside `test_continuity_*` fails, stop and report it — adding a
schema to the re-add must not change any pre-existing suite's behaviour.

- [ ] **Step 9: Commit**

```bash
git add studio/backend/core/inference/tools.py studio/backend/routes/inference.py studio/backend/tests/test_continuity_seam.py studio/backend/tests/test_delegation_seam.py studio/backend/tests/test_tool_audit_seam.py studio/backend/tests/test_tool_readiness_seam.py
git commit -m "feat(continuity): register continuity_task, reachable in Studio chat

ALL_TOOLS, the execute_tool dispatch, the _addable re-add in
routes/inference.py (continuity_task has no UI pill, same as
check_tool_readiness / find_capability / delegation's tools -- the re-add has
been missed three times in this project, so reachability is asserted by
driving the REAL _select_request_tools), and BOTH safe-tool lists.

continuity_task belongs on _ALWAYS_SAFE_TOOLS and
_ANTHROPIC_UNPROMPTED_SAFE_TOOLS, unlike ask_model: it only reads and writes
local sandboxed files, no network, no code execution, so it can never need the
confirmation prompt those lists exist to skip.

main.py is untouched -- this piece adds no HTTP router.

tools.py N -> M insertions, routes/inference.py N -> M, all zero deletions.
Three sibling seam-ceiling tests raised together to match.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

(Fill in the actual N -> M numbers from Step 6's re-measurement before committing — do not leave
placeholder text in the real commit message.)

- [ ] **Step 10: Demonstrate the controls (after committing)**

1. Remove ` | CONTINUITY_TOOL_NAMES` from `_addable`. Expect
   `test_continuity_task_reaches_the_model_through_the_real_route` to FAIL.
2. Add `"continuity_task"` to a `_deny`-shaped mutation — i.e., temporarily wrap the
   `_ALWAYS_SAFE_TOOLS` rebind's `frozenset({"continuity_task"})` to `frozenset()` (empty). Expect
   `test_continuity_task_is_always_safe` to FAIL.
3. Remove the `CONTINUITY_TOOL_NAMES,` line from the `_ANTHROPIC_UNPROMPTED_SAFE_TOOLS` rebind's
   frozenset (set it to `frozenset()`). Expect
   `test_continuity_task_is_present_in_the_anthropic_unprompted_set` to FAIL.

Restore after each with `git checkout -- studio/backend/core/inference/tools.py studio/backend/routes/inference.py`
(both files touched by these edits — restore both, since the tree must be clean between
mutations and `git checkout` on a single file when both changed would leave the other mutated).
Finish with a clean `git status --porcelain` and re-run Step 7's four-file command once more to
confirm the real (unmutated) state is green.

---

### Task 11: Dev CLI

**Files:**
- Create: `studio/backend/tools/continuity_cli.py`
- Test: `studio/backend/tests/test_continuity_cli.py`

**Interfaces:**
- Consumes: the full `core.continuity` public API (Tasks 1–7)
- Produces: a runnable script, `continuity_cli.py <verb> [args...]`

Check first whether `studio/backend/tools/` exists as a directory for this kind of script (
`packaging/check_branding.py` lives in `packaging/`, not `studio/backend/tools/` — run
`ls studio/backend/tools/ 2>/dev/null || echo "does not exist yet"` before creating it). If it
does not exist, create it as a plain directory (no `__init__.py` needed — this script is invoked
directly with a path, not imported as a package member, matching how `check_branding.py` and
`studio/install_llama_prebuilt.py` are both invoked directly rather than imported).

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_continuity_cli.py` — this tests the CLI's **argument-parsing
and dispatch logic** as a function, not by shelling out to a subprocess (a subprocess test would
need the real venv Python on the test-runner's PATH, which is an environment assumption this
project's tests avoid elsewhere):

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""continuity_cli.py's dispatch logic, driven as a function -- not via
subprocess, which would require the real venv Python on the test runner's
PATH. sys.path is extended so this pure-script file (not a package member) is
importable the same way pytest's own conftest resolution would find it.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_CLI_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tools", "continuity_cli.py"
)


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("continuity_cli", _CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_status_on_a_fresh_project_prints_something(tmp_path, capsys):
    cli = _load_cli_module()
    exit_code = cli.main(["status", "--project-dir", str(tmp_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "No project state" in out or "not recorded" in out


def test_init_creates_project_state_then_status_shows_it(tmp_path, capsys):
    """The one gap Task 9's review confirmed spans the whole plan: nothing else
    -- not the tool, not any other CLI verb -- can ever create the FIRST
    project_state.json. This is that path."""
    cli = _load_cli_module()
    exit_code = cli.main(["init", "demo-project", "--project-dir", str(tmp_path)])
    assert exit_code == 0
    capsys.readouterr()
    exit_code = cli.main(["status", "--project-dir", str(tmp_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "demo-project" in out


def test_init_refuses_to_overwrite_an_existing_state(tmp_path, capsys):
    """A second init must not silently erase real progress -- completed work,
    blocking_issues, etc. -- that a fresh ProjectState would discard."""
    cli = _load_cli_module()
    assert cli.main(["init", "first", "--project-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    exit_code = cli.main(["init", "second", "--project-dir", str(tmp_path)])
    assert exit_code != 0
    capsys.readouterr()
    exit_code = cli.main(["status", "--project-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "first" in out, "the original state must survive the refused second init"


def test_task_add_then_status_round_trips(tmp_path, capsys):
    cli = _load_cli_module()
    cli.main(["task", "add", "t1", "--title", "do the thing", "--project-dir", str(tmp_path)])
    capsys.readouterr()  # discard the add's own output
    cli.main(["status", "--project-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "t1" in out


def test_an_unknown_verb_exits_nonzero_and_says_so(tmp_path, capsys):
    cli = _load_cli_module()
    exit_code = cli.main(["not-a-real-verb", "--project-dir", str(tmp_path)])
    assert exit_code != 0


def test_a_continuity_error_exits_nonzero_with_a_readable_message(tmp_path, capsys):
    cli = _load_cli_module()
    # set_status on a task that does not exist raises ContinuityError inside
    # the core; the CLI must print it and exit nonzero, not print a traceback.
    exit_code = cli.main(["task", "set-status", "nope", "in_progress",
                          "--project-dir", str(tmp_path)])
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "nope" in err
```

- [ ] **Step 2: Run it and watch it fail**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_cli.py -q -p no:cacheprovider
```
Expected: failure loading the module (the file does not exist yet).

- [ ] **Step 3: Implement**

Create `studio/backend/tools/continuity_cli.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Direct-use CLI over core.continuity, for a controller session at a context
reset -- mirrors how packaging/check_branding.py and
studio/install_llama_prebuilt.py are both invoked directly with the venv
Python, never imported as a package.

    C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe studio/backend/tools/continuity_cli.py status --project-dir .
    ... status
    ... task add <id> --title "..." [--depends-on id1,id2] [--criteria "c1;c2"]
    ... task set-status <id> <status>
    ... checkpoint "<note>"
    ... validate
    ... repair

Errors from core.continuity (ContinuityError) are printed to stderr and the
process exits 1 -- never a raw traceback, which is unreadable at 2am during an
actual context reset.
"""

from __future__ import annotations

import argparse
import os
import sys

# This script is invoked directly (python continuity_cli.py ...), so
# studio/backend needs to be on sys.path for `core.continuity` to import --
# it is not installed as a package. Inserted once, at the top, before any
# core.* import below.
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)


def _default_project_dir() -> str:
    return os.getcwd()


def main(argv: list[str]) -> int:
    # --project-dir is declared on a shared PARENT parser, given to every leaf
    # subparser via parents=[...], not just on the top-level parser. argparse
    # requires an option declared only on the top-level parser to come BEFORE
    # the subcommand name; every call in this module's own usage and every test
    # writes it AFTER the verb ("status --project-dir X"), which argparse
    # rejects as "unrecognized arguments" unless each leaf parser also declares
    # it. Verified empirically, not assumed.
    common = argparse.ArgumentParser(add_help = False)
    common.add_argument("--project-dir", default = None)

    parser = argparse.ArgumentParser(prog = "continuity_cli", parents = [common])
    sub = parser.add_subparsers(dest = "verb")

    sub.add_parser("status", parents = [common])
    sub.add_parser("validate", parents = [common])
    sub.add_parser("repair", parents = [common])

    init = sub.add_parser("init", parents = [common])
    init.add_argument("project")
    init.add_argument("--phase", default = "")

    cp = sub.add_parser("checkpoint", parents = [common])
    cp.add_argument("note")

    task = sub.add_parser("task", parents = [common])
    task_sub = task.add_subparsers(dest = "task_verb")
    add = task_sub.add_parser("add", parents = [common])
    add.add_argument("id")
    add.add_argument("--title", default = None)
    add.add_argument("--depends-on", default = "")
    add.add_argument("--criteria", default = "")
    set_status = task_sub.add_parser("set-status", parents = [common])
    set_status.add_argument("id")
    set_status.add_argument("status")

    # argparse raises SystemExit(2) on an unrecognized verb, not a normal
    # exception -- left uncaught, that escapes main() entirely rather than
    # becoming the nonzero return code test_an_unknown_verb_exits_nonzero_and_says_so
    # (and every other caller of main()) expects.
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    project_dir = args.project_dir or _default_project_dir()

    from core.continuity import (
        add_task, checkpoint, load_state, load_tasks, repair, set_task_status,
        validate, write_state,
    )
    from core.continuity.render import render_context_summary
    from core.continuity.schemas import ContinuityError, ProjectState, Task

    try:
        if args.verb == "init":
            # The one gap the tool deliberately leaves to a human: nothing in
            # continuity_task (the app-side tool) or this CLI's other verbs can
            # ever create the FIRST project_state.json -- update_state requires
            # one to already exist, and write_state itself has no other caller.
            # Refuses rather than overwrites: init is a one-time bootstrap, and
            # an accidental second run must not silently erase real progress
            # (completed, in_progress, blocking_issues, ...).
            if load_state(project_dir) is not None:
                print(f"Error: project_state.json already exists at {project_dir!r} "
                      "-- init refuses to overwrite it.", file = sys.stderr)
                return 1
            write_state(project_dir, ProjectState(
                schema_version = 1, project = args.project, status = "active",
                current_phase = args.phase, current_task = None,
                completion_percent = 0, last_checkpoint = None,
            ))
            print(f"Initialized {args.project!r} at {project_dir!r}.")
            return 0

        if args.verb == "status":
            # Same fallback Task 9's continuity_task tool needed for the
            # identical reason: render_context_summary reports "no project
            # state" whenever project_state.json doesn't exist, regardless of
            # whether tasks have been added -- and add_task alone (without
            # init) never creates one. Without this, adding a task and then
            # checking status shows nothing, even though real work is tracked.
            if load_state(project_dir) is None:
                queue = load_tasks(project_dir)
                if queue.tasks:
                    print("No project_state.json yet (run `init` to create one). Tasks:")
                    for t in queue.tasks:
                        print(f"- [{t.status}] {t.id}: {t.title}")
                    return 0
            print(render_context_summary(project_dir))
            return 0

        if args.verb == "validate":
            problems = validate(project_dir)
            if not problems:
                print("No problems found.")
                return 0
            for p in problems:
                print(f"- {p}")
            return 1

        if args.verb == "repair":
            actions = repair(project_dir)
            for a in actions:
                print(f"- {a}")
            return 0

        if args.verb == "checkpoint":
            name = checkpoint(project_dir, args.note)
            print(f"Checkpointed as {name}.")
            return 0

        if args.verb == "task":
            if args.task_verb == "add":
                depends = [d for d in args.depends_on.split(",") if d]
                criteria = [c for c in args.criteria.split(";") if c]
                add_task(project_dir, Task(
                    id = args.id, title = args.title or args.id, status = "pending",
                    depends_on = depends, acceptance_criteria = criteria,
                ))
                print(f"Added task {args.id!r}.")
                return 0
            if args.task_verb == "set-status":
                set_task_status(project_dir, args.id, args.status)
                print(f"Task {args.id!r} is now {args.status!r}.")
                return 0
            print(f"Unknown task verb: {args.task_verb!r}", file = sys.stderr)
            return 2

        print(f"Unknown verb: {args.verb!r}", file = sys.stderr)
        return 2
    except ContinuityError as exc:
        print(f"Error: {exc}", file = sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run the tests and watch them pass**

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe -m pytest tests/test_continuity_cli.py -q -p no:cacheprovider
```
Expected: 6 passed.

- [ ] **Step 5: Manually verify the CLI runs standalone**

This is the one check this plan cannot express as a pytest assertion — run it directly and read
the output:

```
cd C:\Users\Admin\unsloth\studio\backend
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe tools/continuity_cli.py status --project-dir C:\Users\Admin\unsloth
```

Expected: prints "No project state recorded yet..." with no traceback. Then:

```
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe tools/continuity_cli.py task add smoke-test --title "smoke test" --criteria "it printed" --project-dir C:\Users\Admin\unsloth
C:/Users/Admin/.unsloth/studio/unsloth_studio/Scripts/python.exe tools/continuity_cli.py status --project-dir C:\Users\Admin\unsloth
```

Expected: the second command's output names `smoke-test`. **Then delete the resulting
`C:\Users\Admin\unsloth\.ai\` directory** — this was a manual smoke test against the real repo
root, not a fixture, and must not be left behind as stray state:

```
rm -rf C:\Users\Admin\unsloth\.ai
```

Confirm with `git status --porcelain` in `C:\Users\Admin\unsloth` that `.ai/` is gone and nothing
else changed (it is untracked, so `git status` will simply stop listing it).

- [ ] **Step 6: Commit**

```bash
git add studio/backend/tools/continuity_cli.py studio/backend/tests/test_continuity_cli.py
git commit -m "feat(continuity): dev CLI over core.continuity

status/validate/repair/checkpoint/task add/task set-status, invoked directly
with the venv Python the same way check_branding.py and
install_llama_prebuilt.py already are -- no new entry point registered
anywhere, no package install. ContinuityError prints to stderr and exits 1;
never a raw traceback.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Demonstrate the control (after committing)**

In `main`, change `except ContinuityError as exc:` to `except ValueError as exc:` (narrowing it
so a real `ContinuityError` is no longer caught). Run
`test_a_continuity_error_exits_nonzero_with_a_readable_message`. Expected: FAIL — the test's
`cli.main(...)` call now raises an unhandled `ContinuityError` instead of returning a nonzero
exit code with a message on stderr. Restore with
`git checkout -- studio/backend/tools/continuity_cli.py`. Finish with a clean
`git status --porcelain`.

---

## Self-Review

**1. Spec coverage.**

| spec section | task |
|---|---|
| §4 architecture, one core two consumers | Tasks 1–11 (structure), explicitly Task 9 (app) / Task 11 (CLI) |
| §5 `project_state.json` schema | Task 1 (schema), Task 2 (load/write/update) |
| §5 `task_queue.json` schema, derived `ready`, transition table | Task 1 (schema), Task 3 |
| §5 `logs/errors.jsonl` schema | Task 1 (schema), Task 4 |
| §5 `checkpoints/` | Task 5 |
| §6 core API, raises rather than guesses | Tasks 2–7 throughout; every function raises `ContinuityError`, none swallows |
| §6 `validate`/`repair`, repair never completes a task | Task 6 |
| §6 `render_context_summary` token cap | Task 7 |
| §7 app tool, sandbox-confined, never-raises, five (here: four) registration points | Task 8 (confinement), Task 9 (handler), Task 10 (registration) |
| §7 explicitly out of scope: delegate write-back | not built — correctly absent from every task |
| §9 deferred: locks/, UI panel | not built — correctly absent |
| §2 reconciliation table (drop decisions.json / activity.jsonl) | enforced by omission — no task creates either file; `important_decisions` holds pointer strings (Task 1's `ProjectState` field), never copied text |

No spec requirement was found without an owning task.

**2. Placeholder scan.** No TBD/TODO/"add appropriate error handling" anywhere in the task
bodies — every step has real code or a concrete manual verification with expected output stated.
The one deliberately-flagged non-literal spot (Task 10 Step 9's `N -> M` commit message) is
explicit about needing the real measured numbers before committing, which is a plan correctly
refusing to guess a number it cannot know until Task 10 Step 6 runs — not a placeholder left for
convenience.

**3. Type consistency.** Traced the public surface end to end: `Task`/`TaskQueue`/`ProjectState`/
`ErrorEntry` are defined once in Task 1 and every later task imports them by the same names and
constructor signatures (`Task(id=, title=, status=, depends_on=, acceptance_criteria=, owner=)`
matches across Tasks 3, 6, 9, 11). `continuity.execute(name, arguments, *, session_id=None, **kwargs)`
(Task 9) matches the dispatch call written into `tools.py` in Task 10 Step 3(c). `CONTINUITY_TOOLS`
/ `CONTINUITY_TOOL_NAMES` are defined once in `schemas_tool.py` (Task 9) and consumed by that
exact name in Task 10's `tools.py` and `routes/inference.py` edits.

**4. Review Focus.** All five items from the header section have an owning task and a named test:
uninitialized-project reads returning empty/None (Tasks 2–3), unknown `depends_on` id (Task 3),
dependency cycle (Task 6), `prior_failures` with a non-matching query (Task 4), checkpoint pruning
by true write order rather than directory order (Task 5). None were skipped.

**One thing worth flagging explicitly, not hidden in a task body:** Task 1's file list opens with
a parenthetical asking the implementer to confirm the actual package path
(`core/continuity/` vs. nested under `core/inference/`) against the real repo layout, and Task 10
repeats that same check before writing the import line. This is intentional, not an oversight —
the constraint block states the sibling-of-`core/inference/` placement as this plan's assumption,
and the two places that assumption matters both carry an explicit "verify this against `ls`
before proceeding" instruction, so a wrong assumption fails loudly and early rather than
propagating silently into Task 10's `tools.py` edit.
