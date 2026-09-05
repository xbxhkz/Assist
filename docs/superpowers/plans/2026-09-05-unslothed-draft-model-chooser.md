# Unslothed Draft-Model Chooser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user pin a specific speculative draft model — a local GGUF, a Hugging Face repo, or a named auto-discovered sidecar — at both the per-chat and saved-per-model surfaces.

**Architecture:** One new additive backend module (`studio/backend/routes/draft_model.py`) owns validation *and* `llama_extra_args` flag composition. The frontend never builds a flag: it sends a choice and receives a finished argument array, which it writes into the existing `llama_extra_args` field that both surfaces already persist and the load path already honours end to end.

**Tech Stack:** Python 3.12 / FastAPI / pytest (backend); React + TypeScript / Vite (frontend).

**Spec:** `docs/superpowers/specs/2026-09-05-unslothed-draft-model-chooser-design.md`

## Global Constraints

- **Upstream seam budget: TWO lines in one file.** The only permitted edit to an existing upstream Python file is `studio/backend/main.py`, gaining one import and one `app.include_router(...)` call. (The spec says "one router-registration line"; the registration needs its import, so the honest number is two insertions. Anything beyond that is over budget.) Everything else is new files under `studio/backend/routes/`, `studio/backend/tests/`, and frontend files.
- **Do NOT edit** `core/inference/tools.py`, `routes/inference.py`, `core/inference/llama_cpp.py`, or `pyproject.toml`. These carry the existing fork seam or are explicitly out of bounds.
- **NEVER run the full backend test suite.** Upstream fixtures fabricate GGUF files up to 40 GB and have filled this machine's disk once. Run only the specific test file named in each task: `python -m pytest tests/<file> -v` from `studio/backend/`.
- **Every negative control must be proven to fail.** This project has produced six controls that turned out inert. A control that passes before the fix is not a control — delete it or fix it, and say which.
- Backend tests begin with the `_BACKEND_DIR` `sys.path` preamble used by every file in `studio/backend/tests/`.
- Python style in this codebase uses spaces around `=` in keyword arguments (`foo(bar = 1)`). Match it.
- SPDX header on every new file:
  ```
  # SPDX-License-Identifier: AGPL-3.0-only
  # Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
  ```

## Correction to the spec

The spec names four drafter flags. The real vocabulary in `core/inference/llama_cpp.py:3687-3688` is **seven**:

```python
_HF_DRAFT_FLAGS    = frozenset({"--spec-draft-hf", "-hfd", "-hfrd", "--hf-repo-draft"})
_LOCAL_DRAFT_FLAGS = frozenset({"--model-draft", "--spec-draft-model", "-md"})
```

Use these by import, never by retyping. This strengthens the spec's argument for backend-side composition rather than weakening it.

## File Structure

| File | Responsibility |
|---|---|
| `studio/backend/routes/draft_model.py` | **New.** Flag composition, validation, candidate enumeration, HTTP routes. |
| `studio/backend/main.py` | **Modify, 1 line.** Router registration. The entire seam. |
| `studio/backend/tests/test_draft_model_compose.py` | **New.** Composition + upstream-import canaries. |
| `studio/backend/tests/test_draft_model_validate.py` | **New.** Validation checks and their negative controls. |
| `studio/frontend/src/features/chat/api/draft-model-api.ts` | **New.** Typed client for the two endpoints. |
| `studio/frontend/src/features/chat/components/draft-model-picker.tsx` | **New.** The shared control, used by both surfaces. |
| `studio/frontend/src/features/chat/chat-settings-sheet.tsx` | **Modify.** Mount the picker (per-chat). |
| `studio/frontend/src/features/api-monitor/components/saved-model-settings.tsx` | **Modify.** Mount the picker (saved-per-model). |

---

### Task 1: Flag composition, with upstream canaries

**Files:**
- Create: `studio/backend/routes/draft_model.py`
- Test: `studio/backend/tests/test_draft_model_compose.py`

**Interfaces:**
- Consumes: `_HF_DRAFT_FLAGS`, `_LOCAL_DRAFT_FLAGS` from `core.inference.llama_cpp`; `_flag_name` from `core.inference.llama_server_args`.
- Produces: `compose_draft_args(existing: list[str], choice: DraftChoice | None) -> list[str]`, and `DraftChoice = tuple[str, str]` where the first element is `"local"` or `"hf"` and the second is the path or repo id.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_draft_model_compose.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Composition of llama_extra_args for a pinned drafter.

The flag vocabulary is imported from upstream rather than retyped, so the
canaries here are load-bearing: if upstream renames or moves those sets, these
tests fail loudly instead of the composer silently missing a spelling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from routes.draft_model import compose_draft_args


class TestUpstreamCanaries:
    """These pin the upstream names the composer depends on. A rename upstream
    must break a test here, not leak into production as an unrecognised flag."""

    def test_flag_sets_are_importable_and_populated(self):
        from core.inference.llama_cpp import _HF_DRAFT_FLAGS, _LOCAL_DRAFT_FLAGS
        assert "-md" in _LOCAL_DRAFT_FLAGS
        assert "--model-draft" in _LOCAL_DRAFT_FLAGS
        assert "--spec-draft-model" in _LOCAL_DRAFT_FLAGS
        assert "-hfd" in _HF_DRAFT_FLAGS
        assert "--spec-draft-hf" in _HF_DRAFT_FLAGS
        assert not (_HF_DRAFT_FLAGS & _LOCAL_DRAFT_FLAGS)

    def test_flag_name_helper_is_importable(self):
        from core.inference.llama_server_args import _flag_name
        assert _flag_name("--model-draft=/x.gguf") == "--model-draft"
        assert _flag_name("-md") == "-md"


class TestComposition:
    def test_pins_a_local_drafter_on_empty_args(self):
        assert compose_draft_args([], ("local", "/models/d.gguf")) == [
            "--model-draft", "/models/d.gguf"
        ]

    def test_pins_an_hf_drafter_on_empty_args(self):
        assert compose_draft_args([], ("hf", "unsloth/Qwen3-0.6B-GGUF")) == [
            "--spec-draft-hf", "unsloth/Qwen3-0.6B-GGUF"
        ]

    def test_unrelated_args_survive_untouched_and_in_order(self):
        existing = ["--threads", "8", "--flash-attn", "-c", "4096"]
        out = compose_draft_args(existing, ("local", "/d.gguf"))
        assert out[:5] == existing, "hand-written args must not be reordered or dropped"
        assert out[5:] == ["--model-draft", "/d.gguf"]

    def test_an_existing_drafter_flag_is_replaced_not_duplicated(self):
        existing = ["--threads", "8", "-md", "/old.gguf", "--flash-attn"]
        out = compose_draft_args(existing, ("local", "/new.gguf"))
        assert out == ["--threads", "8", "--flash-attn", "--model-draft", "/new.gguf"]

    def test_the_equals_spelling_is_recognised_and_removed(self):
        existing = ["--model-draft=/old.gguf", "--threads", "8"]
        out = compose_draft_args(existing, ("local", "/new.gguf"))
        assert "--model-draft=/old.gguf" not in out
        assert out == ["--threads", "8", "--model-draft", "/new.gguf"]

    def test_every_spelling_is_removed_including_the_rare_ones(self):
        existing = ["-hfrd", "a/b", "--hf-repo-draft", "c/d", "--spec-draft-model", "/e.gguf"]
        out = compose_draft_args(existing, ("hf", "x/y"))
        assert out == ["--spec-draft-hf", "x/y"]

    def test_clearing_removes_the_drafter_and_restores_auto_discovery(self):
        existing = ["--threads", "8", "-md", "/old.gguf"]
        assert compose_draft_args(existing, None) == ["--threads", "8"]

    # --- negative controls -------------------------------------------------
    # Each must FAIL against a naive substring implementation. Verified in Step 2.

    def test_control_a_value_that_merely_contains_md_is_not_stripped(self):
        """`--cache-type-k` with value `md_something` must survive. A composer
        that removes any token containing "-md" eats it."""
        existing = ["--alias", "my-md-model", "--threads", "8"]
        out = compose_draft_args(existing, ("local", "/d.gguf"))
        assert "--alias" in out and "my-md-model" in out

    def test_control_a_drafter_flags_value_is_removed_with_it(self):
        """Removing `-md` but leaving `/old.gguf` behind would hand llama-server
        a stray positional argument."""
        out = compose_draft_args(["-md", "/old.gguf"], None)
        assert out == []
```

- [ ] **Step 2: Run the tests and confirm they fail for the right reason**

```bash
cd studio/backend && python -m pytest tests/test_draft_model_compose.py -v
```

Expected: every test fails with `ModuleNotFoundError: No module named 'routes.draft_model'`.

**Also prove the two controls are real.** After Step 3 passes, temporarily replace the removal predicate with the naive version below, re-run, and confirm `test_control_a_value_that_merely_contains_md_is_not_stripped` FAILS. Then revert. If it passes, the control is inert — say so and fix it.

```python
# NAIVE VERSION — for the control check only, do not keep
drop = [i for i, a in enumerate(args) if "-md" in a or "draft" in a]
```

- [ ] **Step 3: Write the implementation**

Create `studio/backend/routes/draft_model.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pinning a specific speculative draft model.

Studio already resolves a drafter automatically: `auto` ranks colocated
MTP/DSpark/DFlash sidecars through utils/models/drafters/preference.py and
launches the winner. That is safe by construction -- only name-matched
neighbours are eligible -- but it is invisible and not reproducible, and there
is no way to name a drafter that lives elsewhere.

This module lets the user pin one, by composing the drafter flags into the
caller's existing ``llama_extra_args``. The load path in routes/inference.py
already honours a caller-named drafter completely (VRAM charged, last-wins
leaving exactly one resident, native path rules applied to split shards), so
nothing there changes.

Composition lives here rather than in the frontend on purpose. The flag
vocabulary is seven spellings across two families, each accepting ``-f v`` and
``-f=v``, and _extra_args_mtp_draft_source already parses exactly that set.
Rebuilding it in TypeScript would mean two parsers with nothing binding them
together -- the shape of the unpinned-dependency failure this project has
already paid for once. The sets are imported, never retyped; tests/
test_draft_model_compose.py pins their names so an upstream rename fails there.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from core.inference.llama_cpp import _HF_DRAFT_FLAGS, _LOCAL_DRAFT_FLAGS
from core.inference.llama_server_args import _flag_name

# ("local", path) or ("hf", repo_id).
DraftChoice = Tuple[str, str]

# The spelling written when pinning. Reading accepts all seven; writing picks
# the canonical long form of each family so the raw-args box stays legible.
_CANONICAL_LOCAL_FLAG = "--model-draft"
_CANONICAL_HF_FLAG = "--spec-draft-hf"

_ALL_DRAFT_FLAGS = _HF_DRAFT_FLAGS | _LOCAL_DRAFT_FLAGS


def _strip_draft_flags(args: Sequence[str]) -> list[str]:
    """Every drafter-naming flag and its value removed; everything else kept in
    order.

    Matching is on the parsed flag NAME, never on a substring: an ``--alias``
    value of ``my-md-model`` contains "-md" and must survive.
    """
    out: list[str] = []
    skip_next = False
    for i, raw in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        token = str(raw)
        flag = _flag_name(token)
        if flag in _ALL_DRAFT_FLAGS:
            _, eq, _inline = token.partition("=")
            if not eq:
                # Separate value form: drop the following token too, unless it
                # is itself a flag (a malformed trailing "-md" owns no value).
                nxt = str(args[i + 1]) if i + 1 < len(args) else ""
                if nxt and not nxt.startswith("-"):
                    skip_next = True
            continue
        out.append(token)
    return out


def compose_draft_args(
    existing: Optional[Sequence[str]], choice: Optional[DraftChoice]
) -> list[str]:
    """``existing`` with any pinned drafter replaced by ``choice``.

    ``choice = None`` clears the pin, which restores auto-discovery: with no
    drafter flag present the load path falls back to its own sidecar ranking.
    """
    args = _strip_draft_flags(existing or [])
    if choice is None:
        return args
    kind, ref = choice
    flag = _CANONICAL_HF_FLAG if kind == "hf" else _CANONICAL_LOCAL_FLAG
    return args + [flag, str(ref)]
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
cd studio/backend && python -m pytest tests/test_draft_model_compose.py -v
```

Expected: all PASS. Then perform the control check described in Step 2 and record the result.

- [ ] **Step 5: Commit**

```bash
git add studio/backend/routes/draft_model.py studio/backend/tests/test_draft_model_compose.py
git commit -m "feat(draft-model): compose pinned drafter flags into llama_extra_args"
```

---

### Task 2: Validation — existence, confinement, size

**Files:**
- Modify: `studio/backend/routes/draft_model.py`
- Test: `studio/backend/tests/test_draft_model_validate.py`

**Interfaces:**
- Consumes: `compose_draft_args` from Task 1.
- Produces: `validate_choice(target_path: str | None, choice: DraftChoice) -> DraftVerdict`, where `DraftVerdict` is a dataclass with fields `ok: bool`, `reason: str`, `detail: str`, `size_bytes: int | None`, `vocab_target: int | None`, `vocab_draft: int | None`. `reason` is a stable machine token from `VERDICT_OK`, `VERDICT_MISSING`, `VERDICT_OUTSIDE`, `VERDICT_VOCAB_MISMATCH`, `VERDICT_VOCAB_UNKNOWN`.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_draft_model_validate.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Validation of a pinned drafter.

Auto-discovery is safe by construction: it only selects colocated,
name-matched sidecars. Pinning removes that protection, so these are the
checks that replace it. Each has a negative control proving it is the check
that fires, not a neighbour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from routes.draft_model import (
    VERDICT_MISSING,
    VERDICT_OK,
    VERDICT_OUTSIDE,
    validate_choice,
)


def _fake_gguf(path: Path, size: int = 2048) -> Path:
    """A file that is not a real GGUF. Enough for existence/size/confinement,
    which run before any header parse."""
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"\0" * size)
    return path


class TestExistence:
    def test_a_missing_local_drafter_is_rejected_with_a_stated_reason(self, tmp_path):
        v = validate_choice(str(tmp_path / "target.gguf"), ("local", str(tmp_path / "nope.gguf")))
        assert not v.ok
        assert v.reason == VERDICT_MISSING
        assert "nope.gguf" in v.detail, "the reason must name the file, not just fail"

    def test_an_existing_local_drafter_passes_existence(self, tmp_path):
        target = _fake_gguf(tmp_path / "target.gguf")
        draft = _fake_gguf(tmp_path / "draft.gguf", size = 4096)
        v = validate_choice(str(target), ("local", str(draft)))
        assert v.reason != VERDICT_MISSING
        assert v.size_bytes == 4096, "size must be reported so the UI can warn before a 409"


class TestConfinement:
    def test_a_drafter_outside_the_target_tree_is_rejected(self, tmp_path):
        target = _fake_gguf(tmp_path / "models" / "target.gguf")
        outside = _fake_gguf(tmp_path / "elsewhere" / "secret.gguf")
        v = validate_choice(str(target), ("local", str(outside)))
        assert not v.ok
        assert v.reason == VERDICT_OUTSIDE

    def test_a_sibling_of_the_target_is_allowed(self, tmp_path):
        target = _fake_gguf(tmp_path / "models" / "target.gguf")
        sibling = _fake_gguf(tmp_path / "models" / "draft.gguf")
        v = validate_choice(str(target), ("local", str(sibling)))
        assert v.reason != VERDICT_OUTSIDE

    # --- negative control --------------------------------------------------
    def test_control_a_symlink_escape_is_rejected(self, tmp_path):
        """A path INSIDE the tree that resolves outside it. A confinement check
        using string prefixes on the unresolved path admits this."""
        target = _fake_gguf(tmp_path / "models" / "target.gguf")
        outside = _fake_gguf(tmp_path / "elsewhere" / "secret.gguf")
        link = tmp_path / "models" / "innocent.gguf"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable (Windows without developer mode)")
        v = validate_choice(str(target), ("local", str(link)))
        assert not v.ok, "a symlink out of the tree must not be admitted"
        assert v.reason == VERDICT_OUTSIDE


class TestRemote:
    def test_an_hf_repo_is_not_subjected_to_local_path_checks(self, tmp_path):
        target = _fake_gguf(tmp_path / "target.gguf")
        v = validate_choice(str(target), ("hf", "unsloth/Qwen3-0.6B-GGUF"))
        assert v.reason not in (VERDICT_MISSING, VERDICT_OUTSIDE)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd studio/backend && python -m pytest tests/test_draft_model_validate.py -v
```

Expected: `ImportError: cannot import name 'VERDICT_MISSING' from 'routes.draft_model'`.

**Control proof:** after Step 3 passes, change `_resolve_real` to return `Path(p)` (no `.resolve()`), re-run, and confirm `test_control_a_symlink_escape_is_rejected` FAILS. Revert. Record the result.

- [ ] **Step 3: Write the implementation**

Append to `studio/backend/routes/draft_model.py`:

```python
import os
from dataclasses import dataclass

VERDICT_OK = "ok"
VERDICT_MISSING = "missing"
VERDICT_OUTSIDE = "outside_permitted_directory"
VERDICT_VOCAB_MISMATCH = "vocab_mismatch"
VERDICT_VOCAB_UNKNOWN = "vocab_unknown"


@dataclass
class DraftVerdict:
    ok: bool
    reason: str
    detail: str = ""
    size_bytes: Optional[int] = None
    vocab_target: Optional[int] = None
    vocab_draft: Optional[int] = None


def _resolve_real(p: str) -> Path:
    """Fully resolved path. `.resolve()` is what makes the confinement check
    survive a symlink pointing out of the tree; a string-prefix test on the
    unresolved path admits exactly that escape."""
    return Path(p).resolve()


def _is_confined(target: Path, draft: Path) -> bool:
    """A pinned local drafter must live in the target's directory tree.

    Same rule auto-discovery obeys implicitly by only ever considering
    colocated files, made explicit here because pinning can name anything.
    """
    root = target.parent
    try:
        draft.relative_to(root)
        return True
    except ValueError:
        return False


def validate_choice(target_path: Optional[str], choice: DraftChoice) -> DraftVerdict:
    """Whether ``choice`` is a usable drafter for ``target_path``.

    Remote repositories skip the local filesystem checks: there is no path to
    resolve and the load path prices them from their own listing.
    """
    kind, ref = choice
    if kind == "hf":
        return DraftVerdict(ok = True, reason = VERDICT_OK, detail = str(ref))

    draft = _resolve_real(str(ref))
    if not draft.is_file():
        return DraftVerdict(
            ok = False, reason = VERDICT_MISSING,
            detail = f"no file at {ref}",
        )
    if target_path:
        target = _resolve_real(target_path)
        if not _is_confined(target, draft):
            return DraftVerdict(
                ok = False, reason = VERDICT_OUTSIDE,
                detail = f"{draft.name} is outside {target.parent}",
            )
    return DraftVerdict(
        ok = True, reason = VERDICT_OK, detail = draft.name,
        size_bytes = draft.stat().st_size,
    )
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
cd studio/backend && python -m pytest tests/test_draft_model_validate.py -v
```

Expected: all PASS (the symlink control may SKIP on Windows without developer mode — if it skips, say so rather than counting it as passing). Then run the control proof from Step 2.

- [ ] **Step 5: Commit**

```bash
git add studio/backend/routes/draft_model.py studio/backend/tests/test_draft_model_validate.py
git commit -m "feat(draft-model): validate a pinned drafter's existence, confinement and size"
```

---

### Task 3: Vocabulary compatibility

**Files:**
- Modify: `studio/backend/routes/draft_model.py`
- Test: `studio/backend/tests/test_draft_model_validate.py`

**Interfaces:**
- Consumes: `DraftVerdict`, `validate_choice` from Task 2.
- Produces: `read_gguf_vocab_size(path: str) -> int | None`; `validate_choice` now also populates `vocab_target` / `vocab_draft` and can return `VERDICT_VOCAB_MISMATCH` or `VERDICT_VOCAB_UNKNOWN`.

**Why this task exists:** nothing in the backend compares a drafter against its target today. llama.cpp requires a shared vocabulary for speculative decoding; a mismatched pair fails inside llama-server with an error the user never asked for. Vocabulary size is the cheapest signal that catches the catastrophic case. It is **necessary, not sufficient** — two models can share a vocab size and still draft poorly — and the UI copy in Task 6 must not overclaim.

- [ ] **Step 1: Write the failing test**

Append to `studio/backend/tests/test_draft_model_validate.py`:

```python
import struct

from routes.draft_model import (
    VERDICT_VOCAB_MISMATCH,
    VERDICT_VOCAB_UNKNOWN,
    read_gguf_vocab_size,
)

_GGUF_MAGIC = 0x46554747
_TYPE_ARRAY = 9
_TYPE_STRING = 8


def _write_gguf_with_vocab(path: Path, n_tokens: int) -> Path:
    """A minimal but REAL GGUF header whose tokenizer.ggml.tokens array has
    ``n_tokens`` entries. Built rather than mocked: the parser under test reads
    bytes, so a mock would test nothing about the parsing."""
    path.parent.mkdir(parents = True, exist_ok = True)
    key = b"tokenizer.ggml.tokens"
    body = struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, 1)
    body += struct.pack("<Q", len(key)) + key
    body += struct.pack("<I", _TYPE_ARRAY)
    body += struct.pack("<I", _TYPE_STRING)
    body += struct.pack("<Q", n_tokens)
    for i in range(n_tokens):
        tok = f"t{i}".encode()
        body += struct.pack("<Q", len(tok)) + tok
    path.write_bytes(body)
    return path


class TestVocabulary:
    def test_vocab_size_is_read_from_the_token_array_length(self, tmp_path):
        g = _write_gguf_with_vocab(tmp_path / "m.gguf", 7)
        assert read_gguf_vocab_size(str(g)) == 7

    def test_a_non_gguf_file_reads_as_unknown_not_zero(self, tmp_path):
        p = tmp_path / "not.gguf"
        p.write_bytes(b"\0" * 512)
        assert read_gguf_vocab_size(str(p)) is None

    def test_matching_vocabularies_pass(self, tmp_path):
        t = _write_gguf_with_vocab(tmp_path / "m" / "target.gguf", 32)
        d = _write_gguf_with_vocab(tmp_path / "m" / "draft.gguf", 32)
        v = validate_choice(str(t), ("local", str(d)))
        assert v.ok
        assert v.vocab_target == 32 and v.vocab_draft == 32

    def test_mismatched_vocabularies_are_rejected(self, tmp_path):
        t = _write_gguf_with_vocab(tmp_path / "m" / "target.gguf", 32)
        d = _write_gguf_with_vocab(tmp_path / "m" / "draft.gguf", 64)
        v = validate_choice(str(t), ("local", str(d)))
        assert not v.ok
        assert v.reason == VERDICT_VOCAB_MISMATCH
        assert "32" in v.detail and "64" in v.detail

    def test_an_unreadable_vocabulary_is_reported_not_silently_passed(self, tmp_path):
        """Fail visibly. A silent pass here is the exact shape of the
        sqlite-vec and update_flow.py defects this project has already paid
        for: a feature that quietly does not work."""
        t = _write_gguf_with_vocab(tmp_path / "m" / "target.gguf", 32)
        d = _fake_gguf(tmp_path / "m" / "draft.gguf")   # not a GGUF
        v = validate_choice(str(t), ("local", str(d)))
        assert v.reason == VERDICT_VOCAB_UNKNOWN
        assert not v.ok

    # --- negative control --------------------------------------------------
    def test_control_the_vocab_gate_fires_on_its_own(self, tmp_path):
        """The mismatched pair must pass existence, confinement and size, so a
        VOCAB_MISMATCH verdict cannot be a neighbouring check misfiring."""
        t = _write_gguf_with_vocab(tmp_path / "m" / "target.gguf", 32)
        d = _write_gguf_with_vocab(tmp_path / "m" / "draft.gguf", 64)
        assert d.is_file()                              # existence would pass
        assert d.parent == Path(str(t)).parent          # confinement would pass
        v = validate_choice(str(t), ("local", str(d)))
        assert v.reason == VERDICT_VOCAB_MISMATCH
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd studio/backend && python -m pytest tests/test_draft_model_validate.py -v -k Vocabulary
```

Expected: `ImportError: cannot import name 'read_gguf_vocab_size'`.

- [ ] **Step 3: Write the implementation**

Append to `studio/backend/routes/draft_model.py`:

```python
import struct

_GGUF_MAGIC = 0x46554747  # b"GGUF" LE u32
_VOCAB_KEY = b"tokenizer.ggml.tokens"
_TYPE_ARRAY = 9

# Byte widths of the fixed-size GGUF value types, indexed by type id. Mirrors
# _FIXED_VTYPE_SIZES in utils/models/gguf_metadata.py; kept local so this module
# depends on one upstream private set (the draft flags) rather than two.
_FIXED_VTYPE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_gguf_vocab_size(path: str) -> Optional[int]:
    """Token count from ``tokenizer.ggml.tokens``, or None when unreadable.

    The array LENGTH is the vocabulary size: many GGUFs carry no vocab_size
    key, which is the same approach the loader itself takes. Only the length is
    read -- the tokens are skipped without being materialised, so a six-figure
    vocabulary costs no memory.

    Returns None rather than 0 for "could not determine". Callers must treat
    None as unknown and say so; a 0 would compare equal to another 0 and
    silently admit a mismatched pair.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(24)
            if len(head) < 24:
                return None
            magic, _version, _tcount, kv_count = struct.unpack("<IIQQ", head)
            if magic != _GGUF_MAGIC:
                return None
            for _ in range(kv_count):
                raw = f.read(8)
                if len(raw) < 8:
                    return None
                (klen,) = struct.unpack("<Q", raw)
                key = f.read(klen)
                vtype_raw = f.read(4)
                if len(vtype_raw) < 4:
                    return None
                (vtype,) = struct.unpack("<I", vtype_raw)
                if key == _VOCAB_KEY and vtype == _TYPE_ARRAY:
                    atype_raw = f.read(4)
                    alen_raw = f.read(8)
                    if len(atype_raw) < 4 or len(alen_raw) < 8:
                        return None
                    (alen,) = struct.unpack("<Q", alen_raw)
                    return int(alen)
                if not _skip_value(f, vtype):
                    return None
    except (OSError, struct.error, ValueError):
        return None
    return None


def _skip_value(f, vtype: int) -> bool:
    """Advance past one GGUF value. False when the type is unknown, which ends
    the walk rather than misreading the rest of the header."""
    if vtype in _FIXED_VTYPE_SIZES:
        return len(f.read(_FIXED_VTYPE_SIZES[vtype])) == _FIXED_VTYPE_SIZES[vtype]
    if vtype == 8:  # string
        raw = f.read(8)
        if len(raw) < 8:
            return False
        (n,) = struct.unpack("<Q", raw)
        f.seek(n, os.SEEK_CUR)
        return True
    if vtype == _TYPE_ARRAY:
        head = f.read(12)
        if len(head) < 12:
            return False
        atype, alen = struct.unpack("<IQ", head)
        for _ in range(alen):
            if not _skip_value(f, atype):
                return False
        return True
    return False
```

Then extend `validate_choice`'s local branch — replace its final `return` with:

```python
    size_bytes = draft.stat().st_size
    v_target = read_gguf_vocab_size(str(target)) if target_path else None
    v_draft = read_gguf_vocab_size(str(draft))
    if v_target is None or v_draft is None:
        return DraftVerdict(
            ok = False, reason = VERDICT_VOCAB_UNKNOWN,
            detail = "could not read the vocabulary from one of the files",
            size_bytes = size_bytes, vocab_target = v_target, vocab_draft = v_draft,
        )
    if v_target != v_draft:
        return DraftVerdict(
            ok = False, reason = VERDICT_VOCAB_MISMATCH,
            detail = f"target vocabulary is {v_target}, drafter is {v_draft}",
            size_bytes = size_bytes, vocab_target = v_target, vocab_draft = v_draft,
        )
    return DraftVerdict(
        ok = True, reason = VERDICT_OK, detail = draft.name,
        size_bytes = size_bytes, vocab_target = v_target, vocab_draft = v_draft,
    )
```

Note the earlier tests in Task 2 used non-GGUF fixtures, so they now land on `VERDICT_VOCAB_UNKNOWN`. Update `test_an_existing_local_drafter_passes_existence` and `test_a_sibling_of_the_target_is_allowed` to assert only that the reason is not `VERDICT_MISSING` / `VERDICT_OUTSIDE` respectively — which they already do. Confirm this rather than assuming it.

- [ ] **Step 4: Run the tests and verify they pass**

```bash
cd studio/backend && python -m pytest tests/test_draft_model_validate.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add studio/backend/routes/draft_model.py studio/backend/tests/test_draft_model_validate.py
git commit -m "feat(draft-model): reject vocabulary-mismatched drafters before load"
```

---

### Task 4: Candidate enumeration and HTTP routes

**Files:**
- Modify: `studio/backend/routes/draft_model.py`
- Modify: `studio/backend/main.py` (**the one-line seam**)
- Test: `studio/backend/tests/test_draft_model_routes.py`

**Interfaces:**
- Consumes: `compose_draft_args`, `validate_choice`, `DraftVerdict`.
- Produces: `GET /api/draft-model/candidates?model_path=…` → `{"candidates": [{"kind","ref","label","source"}]}` where `source` is `"sidecar"` or `"local"`; `POST /api/draft-model/select` with body `{"model_path", "existing_args", "choice"}` → `{"ok","reason","detail","size_bytes","vocab_target","vocab_draft","llama_extra_args"}`. `choice` is `null` to clear, else `{"kind","ref"}`.

- [ ] **Step 1: Write the failing test**

Create `studio/backend/tests/test_draft_model_routes.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTP contract for the draft-model chooser."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from routes.draft_model import router

_GGUF_MAGIC = 0x46554747


def _gguf(path: Path, n_tokens: int = 32) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    key = b"tokenizer.ggml.tokens"
    body = struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, 1)
    body += struct.pack("<Q", len(key)) + key
    body += struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", n_tokens)
    for i in range(n_tokens):
        tok = f"t{i}".encode()
        body += struct.pack("<Q", len(tok)) + tok
    path.write_bytes(body)
    return path


@pytest.fixture
def client():
    """Auth is replaced through dependency_overrides, NOT monkeypatch.

    `Depends(get_current_subject)` captures the function object when the route
    is defined at import time, so reassigning the module attribute afterwards
    changes nothing and every request would still hit real auth. FastAPI's
    override map is keyed on that captured object, which is why it works.
    """
    from auth.authentication import get_current_subject

    app = FastAPI()
    app.include_router(router, prefix = "/api/draft-model")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    return TestClient(app)


class TestCandidates:
    def test_colocated_ggufs_are_offered_and_the_target_itself_is_not(self, tmp_path, client):
        target = _gguf(tmp_path / "m" / "target.gguf")
        _gguf(tmp_path / "m" / "draft.gguf")
        r = client.get("/api/draft-model/candidates", params = {"model_path": str(target)})
        assert r.status_code == 200
        refs = [c["ref"] for c in r.json()["candidates"]]
        assert any("draft.gguf" in x for x in refs)
        assert not any("target.gguf" in x for x in refs), (
            "a model must never be offered as its own drafter"
        )


class TestSelect:
    def test_a_valid_choice_returns_composed_args(self, tmp_path, client):
        target = _gguf(tmp_path / "m" / "target.gguf", 32)
        draft = _gguf(tmp_path / "m" / "draft.gguf", 32)
        r = client.post("/api/draft-model/select", json = {
            "model_path": str(target),
            "existing_args": ["--threads", "8"],
            "choice": {"kind": "local", "ref": str(draft)},
        })
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["llama_extra_args"] == ["--threads", "8", "--model-draft", str(draft)]

    def test_an_invalid_choice_returns_the_reason_and_no_args(self, tmp_path, client):
        target = _gguf(tmp_path / "m" / "target.gguf", 32)
        draft = _gguf(tmp_path / "m" / "draft.gguf", 64)
        r = client.post("/api/draft-model/select", json = {
            "model_path": str(target),
            "existing_args": ["--threads", "8"],
            "choice": {"kind": "local", "ref": str(draft)},
        })
        assert r.status_code == 200, "a rejected choice is a verdict, not a server error"
        body = r.json()
        assert body["ok"] is False
        assert body["reason"] == "vocab_mismatch"
        assert body["llama_extra_args"] is None, (
            "a rejected choice must not hand back args the UI might save anyway"
        )

    def test_clearing_removes_the_drafter_flags(self, tmp_path, client):
        target = _gguf(tmp_path / "m" / "target.gguf")
        r = client.post("/api/draft-model/select", json = {
            "model_path": str(target),
            "existing_args": ["--threads", "8", "-md", "/old.gguf"],
            "choice": None,
        })
        assert r.json()["llama_extra_args"] == ["--threads", "8"]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd studio/backend && python -m pytest tests/test_draft_model_routes.py -v
```

Expected: `ImportError: cannot import name 'router' from 'routes.draft_model'`.

- [ ] **Step 3: Write the implementation**

Append to `studio/backend/routes/draft_model.py`:

```python
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from auth.authentication import get_current_subject
from loggers import get_logger

router = APIRouter()
logger = get_logger(__name__)


class ChoiceIn(BaseModel):
    kind: str
    ref: str


class SelectIn(BaseModel):
    model_path: Optional[str] = None
    existing_args: list[str] = []
    choice: Optional[ChoiceIn] = None


@router.get("/candidates")
def list_candidates(
    model_path: str = Query("", max_length = 4096),
    current_subject: str = Depends(get_current_subject),
) -> dict:
    """Drafters selectable for ``model_path``.

    Colocated GGUFs only. The target itself is excluded: a model cannot draft
    for itself, and offering it invites a load that wastes VRAM on a second
    copy of the same weights.
    """
    out: list[dict] = []
    if model_path:
        target = _resolve_real(model_path)
        if target.parent.is_dir():
            for f in sorted(target.parent.glob("*.gguf")):
                if _resolve_real(str(f)) == target:
                    continue
                out.append({
                    "kind": "local",
                    "ref": str(f),
                    "label": f.name,
                    "source": "sidecar" if _looks_like_sidecar(f.name) else "local",
                })
    return {"candidates": out}


def _looks_like_sidecar(name: str) -> bool:
    """Whether auto-discovery would treat this as a drafter sidecar. Labelling
    only -- it changes how the entry is presented, never whether it is offered,
    so a naming convention that drifts upstream cannot hide a valid choice."""
    low = name.lower()
    return low.startswith(("mtp-", "dspark-", "dflash-")) or "-mtp-" in low


@router.post("/select")
def select_draft_model(
    payload: SelectIn,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    """Validate a choice and return the llama_extra_args that pin it.

    A rejected choice is a 200 carrying a verdict, not an error status: the UI
    renders the reason inline next to the picker, and an HTTP error would make
    a normal, expected outcome look like a fault.
    """
    if payload.choice is None:
        return {
            "ok": True, "reason": VERDICT_OK, "detail": "",
            "size_bytes": None, "vocab_target": None, "vocab_draft": None,
            "llama_extra_args": compose_draft_args(payload.existing_args, None),
        }
    verdict = validate_choice(payload.model_path, (payload.choice.kind, payload.choice.ref))
    args = (
        compose_draft_args(payload.existing_args, (payload.choice.kind, payload.choice.ref))
        if verdict.ok else None
    )
    return {
        "ok": verdict.ok, "reason": verdict.reason, "detail": verdict.detail,
        "size_bytes": verdict.size_bytes,
        "vocab_target": verdict.vocab_target, "vocab_draft": verdict.vocab_draft,
        "llama_extra_args": args,
    }
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
cd studio/backend && python -m pytest tests/test_draft_model_routes.py -v
```

Expected: all PASS.

- [ ] **Step 5: Register the router — the entire upstream seam**

In `studio/backend/main.py`, beside the other `include_router` calls (around line 1367), add exactly one line:

```python
app.include_router(draft_model_router, prefix = "/api/draft-model", tags = ["draft-model"])
```

and its import alongside the existing route imports:

```python
from routes.draft_model import router as draft_model_router
```

**That import plus that line is the whole seam for this sub-project.** Verify nothing else in `main.py` changed:

```bash
git diff --stat studio/backend/main.py
```

Expected: `1 file changed, 2 insertions(+)`. If it is more, revert and redo.

- [ ] **Step 6: Verify the app still starts**

```bash
cd studio/backend && python -c "import main; print('app imports OK')"
```

Expected: `app imports OK`.

- [ ] **Step 7: Commit**

```bash
git add studio/backend/routes/draft_model.py studio/backend/tests/test_draft_model_routes.py studio/backend/main.py
git commit -m "feat(draft-model): candidates and select endpoints, registered on one line"
```

---

### Task 5: Frontend API client

**Files:**
- Create: `studio/frontend/src/features/chat/api/draft-model-api.ts`

**Interfaces:**
- Consumes: the two endpoints from Task 4.
- Produces: `type DraftChoice = { kind: "local" | "hf"; ref: string } | null`; `type DraftCandidate = { kind: string; ref: string; label: string; source: "sidecar" | "local" }`; `type SelectResult = { ok: boolean; reason: string; detail: string; sizeBytes: number | null; vocabTarget: number | null; vocabDraft: number | null; llamaExtraArgs: string[] | null }`; `fetchDraftCandidates(modelPath: string): Promise<DraftCandidate[]>`; `selectDraftModel(modelPath: string, existingArgs: string[], choice: DraftChoice): Promise<SelectResult>`.

- [ ] **Step 1: Write the implementation**

Follow the fetch/auth conventions of the sibling files in `studio/frontend/src/features/chat/api/` — read `chat-settings-api.ts` first and match how it builds URLs and attaches credentials rather than inventing a new pattern.

```ts
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Client for the draft-model chooser.
 *
 * This module deliberately does NOT build llama-server flags. The drafter flag
 * vocabulary is seven spellings across two families, each accepting `-f v` and
 * `-f=v`, with last-wins ordering; the backend already parses that set, and a
 * second implementation here would be free to drift from it. We send a choice
 * and receive a finished argument array.
 */

export type DraftChoice = { kind: "local" | "hf"; ref: string } | null;

export type DraftCandidate = {
  kind: string;
  ref: string;
  label: string;
  source: "sidecar" | "local";
};

export type SelectResult = {
  ok: boolean;
  reason: string;
  detail: string;
  sizeBytes: number | null;
  vocabTarget: number | null;
  vocabDraft: number | null;
  llamaExtraArgs: string[] | null;
};

export async function fetchDraftCandidates(modelPath: string): Promise<DraftCandidate[]> {
  const url = `/api/draft-model/candidates?model_path=${encodeURIComponent(modelPath)}`;
  const res = await fetch(url, { credentials: "include" });
  if (!res.ok) return [];
  const body = await res.json();
  return Array.isArray(body?.candidates) ? body.candidates : [];
}

export async function selectDraftModel(
  modelPath: string,
  existingArgs: string[],
  choice: DraftChoice,
): Promise<SelectResult> {
  const res = await fetch("/api/draft-model/select", {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_path: modelPath,
      existing_args: existingArgs,
      choice: choice ? { kind: choice.kind, ref: choice.ref } : null,
    }),
  });
  const b = await res.json();
  return {
    ok: Boolean(b?.ok),
    reason: String(b?.reason ?? ""),
    detail: String(b?.detail ?? ""),
    sizeBytes: b?.size_bytes ?? null,
    vocabTarget: b?.vocab_target ?? null,
    vocabDraft: b?.vocab_draft ?? null,
    llamaExtraArgs: Array.isArray(b?.llama_extra_args) ? b.llama_extra_args : null,
  };
}
```

- [ ] **Step 2: Typecheck**

```bash
cd studio/frontend && npx tsc -b --noEmit
```

Expected: no errors introduced by this file.

- [ ] **Step 3: Commit**

```bash
git add studio/frontend/src/features/chat/api/draft-model-api.ts
git commit -m "feat(draft-model): typed frontend client for the chooser endpoints"
```

---

### Task 6: The shared picker component

**Files:**
- Create: `studio/frontend/src/features/chat/components/draft-model-picker.tsx`

**Interfaces:**
- Consumes: everything exported by `draft-model-api.ts`.
- Produces: `DraftModelPicker`, props `{ modelPath: string; speculativeType: string; existingArgs: string[]; onArgsChange: (args: string[]) => void }`.

**Behaviour:** renders nothing when `speculativeType` is `"ngram"`, `"ngram-simple"` or `"off"` — those launch no separate drafter, so a chooser there would be a control with no effect. Shows the current pin (parsed from `existingArgs` by asking the backend, never by parsing flags locally), a candidate list, an HF free-text field, and a Clear action. On any change it calls `selectDraftModel` and only calls `onArgsChange` when `ok` is true; otherwise it renders `detail` inline.

**Copy requirement:** the compatibility message must not overclaim. Vocabulary equality is necessary, not sufficient — say "vocabulary matches" or "vocabularies differ (32000 vs 151936)", never "compatible" / "will work".

- [ ] **Step 1: Read the surrounding conventions before writing**

Read `studio/frontend/src/features/chat/chat-settings-sheet.tsx` around the existing speculative-type control and match its component vocabulary (Select, Label, helper-text classes). Do not introduce a new UI primitive or styling approach.

- [ ] **Step 2: Write the component**

Write this, substituting the `Select` / `Label` / helper-text primitives observed in Step 1 for the plain elements below. The logic must survive that substitution unchanged.

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";
import {
  type DraftCandidate,
  type DraftChoice,
  fetchDraftCandidates,
  selectDraftModel,
} from "../api/draft-model-api";

/** Modes that launch no separate drafter. A picker here would be a control
 *  with no effect, which is worse than no control. */
const NO_DRAFTER_MODES = new Set(["off", "ngram", "ngram-simple"]);

export type DraftModelPickerProps = {
  modelPath: string;
  speculativeType: string;
  existingArgs: string[];
  onArgsChange: (args: string[]) => void;
};

export function DraftModelPicker({
  modelPath,
  speculativeType,
  existingArgs,
  onArgsChange,
}: DraftModelPickerProps) {
  const [candidates, setCandidates] = useState<DraftCandidate[]>([]);
  const [hfRepo, setHfRepo] = useState("");
  const [problem, setProblem] = useState("");
  const [note, setNote] = useState("");

  const hidden = NO_DRAFTER_MODES.has(speculativeType);

  useEffect(() => {
    if (hidden || !modelPath) return;
    let cancelled = false;
    fetchDraftCandidates(modelPath).then((c) => {
      if (!cancelled) setCandidates(c);
    });
    return () => {
      cancelled = true;
    };
  }, [hidden, modelPath]);

  const apply = useCallback(
    async (choice: DraftChoice) => {
      setProblem("");
      setNote("");
      const r = await selectDraftModel(modelPath, existingArgs, choice);
      if (!r.ok || r.llamaExtraArgs === null) {
        // Rejected: surface the reason and leave the caller's args untouched.
        setProblem(r.detail || r.reason);
        return;
      }
      if (r.vocabTarget !== null && r.vocabDraft !== null) {
        // Deliberately "vocabulary matches", never "compatible": equal vocab
        // size is necessary, not sufficient, for a good speculative pair.
        setNote(`vocabulary matches (${r.vocabTarget})`);
      }
      if (r.sizeBytes !== null) {
        const gb = (r.sizeBytes / 1024 ** 3).toFixed(2);
        setNote((n) => (n ? `${n} · ${gb} GB` : `${gb} GB`));
      }
      onArgsChange(r.llamaExtraArgs);
    },
    [modelPath, existingArgs, onArgsChange],
  );

  // Re-validate an existing pin on mount (Task 8 extends this). A pinned
  // drafter that has since been deleted fails open at load time, so this is
  // the only place it becomes visible.
  useEffect(() => {
    if (hidden || !modelPath) return;
    const i = existingArgs.findIndex(
      (a) => a === "--model-draft" || a === "--spec-draft-hf",
    );
    if (i === -1 || i + 1 >= existingArgs.length) return;
    const kind = existingArgs[i] === "--spec-draft-hf" ? "hf" : "local";
    selectDraftModel(modelPath, existingArgs, {
      kind: kind as "local" | "hf",
      ref: existingArgs[i + 1],
    }).then((r) => {
      if (!r.ok) setProblem(`pinned drafter unusable: ${r.detail || r.reason}`);
    });
    // Mount-only: re-running on every args change would re-validate our own writes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hidden, modelPath]);

  if (hidden) return null;

  return (
    <div className="space-y-2">
      <label htmlFor="draft-model">Draft model</label>
      <select
        id="draft-model"
        onChange={(e) => {
          const v = e.target.value;
          void apply(v ? { kind: "local", ref: v } : null);
        }}
      >
        <option value="">Automatic</option>
        {candidates.map((c) => (
          <option key={c.ref} value={c.ref}>
            {c.label}
            {c.source === "sidecar" ? " (sidecar)" : ""}
          </option>
        ))}
      </select>

      {candidates.length === 0 && <p>No colocated drafters found.</p>}

      <div>
        <input
          value={hfRepo}
          placeholder="Hugging Face repo, e.g. unsloth/Qwen3-0.6B-GGUF"
          onChange={(e) => setHfRepo(e.target.value)}
        />
        <button
          type="button"
          disabled={!hfRepo.trim()}
          onClick={() => void apply({ kind: "hf", ref: hfRepo.trim() })}
        >
          Use repo
        </button>
      </div>

      <button type="button" onClick={() => void apply(null)}>
        Clear (use automatic)
      </button>

      {problem && <p role="alert">{problem}</p>}
      {!problem && note && <p>{note}</p>}
    </div>
  );
}
```

- [ ] **Step 3: Typecheck and lint**

```bash
cd studio/frontend && npx tsc -b --noEmit && npx biome check src/features/chat/components/draft-model-picker.tsx
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add studio/frontend/src/features/chat/components/draft-model-picker.tsx
git commit -m "feat(draft-model): shared picker component for both surfaces"
```

---

### Task 7: Mount on both surfaces

**Files:**
- Modify: `studio/frontend/src/features/chat/chat-settings-sheet.tsx`
- Modify: `studio/frontend/src/features/api-monitor/components/saved-model-settings.tsx`

**Interfaces:**
- Consumes: `DraftModelPicker` from Task 6.
- Produces: no new exports.

Both files already read and write `llama_extra_args`, so the picker's `onArgsChange` feeds the existing setter in each. Find that setter before writing — do not add a second write path for the same field.

- [ ] **Step 1: Mount in the chat settings sheet**

Place `<DraftModelPicker …/>` directly below the existing speculative-type control, passing the sheet's current model path, its `speculativeType` value, and its existing `llamaExtraArgs` state plus setter.

- [ ] **Step 2: Mount in saved model settings**

Same component, wired to that panel's per-model `llama_extra_args` state and its saved `speculative_type`.

- [ ] **Step 3: Typecheck, lint, build**

```bash
cd studio/frontend && npx tsc -b --noEmit && npm run build
```

Expected: build succeeds and emits `dist/index.html`.

- [ ] **Step 4: Manual verification against a running backend**

Start the backend and confirm, stating which you observed rather than asserting all of them:

1. The picker is hidden when the mode is `off` or `ngram`.
2. Choosing a colocated GGUF writes `--model-draft <path>` into the raw-args box and nothing else changes there.
3. Choosing a mismatched-vocabulary GGUF shows the reason and leaves the args unchanged.
4. Clear removes the flag and leaves hand-written args intact.
5. The saved-per-model choice survives a reload.

- [ ] **Step 5: Commit**

```bash
git add studio/frontend/src/features/chat/chat-settings-sheet.tsx studio/frontend/src/features/api-monitor/components/saved-model-settings.tsx
git commit -m "feat(draft-model): mount the picker on the chat and saved-model surfaces"
```

---

### Task 8: Re-validate on open, and record the known limitation

**Files:**
- Modify: `studio/frontend/src/features/chat/components/draft-model-picker.tsx`
- Create: `studio/backend/routes/draft_model.py` docstring addition (no new file)

**Interfaces:**
- Consumes: `selectDraftModel` from Task 5.
- Produces: no new exports.

**Why:** a pinned drafter that later disappears from disk is treated by the load path as *"a local `--model-draft` that is not on disk, so no drafter loads and none is charged"* — silently, with speculation quietly reverting to none. Selection-time validation cannot prevent that at load time, and fixing it properly means editing the upstream load path, which the seam budget forbids.

- [ ] **Step 1: Re-validate the current pin when the picker mounts**

When `existingArgs` already contains a pin, call `selectDraftModel` with that same choice on mount. If the verdict is not `ok`, render the reason prominently — a missing pinned drafter must be visible here, because it will be invisible at load time.

- [ ] **Step 2: Record the limitation in the module docstring**

Append to the module docstring of `studio/backend/routes/draft_model.py`:

```
Known limitation: a pinned drafter that is later deleted fails OPEN at load
time. routes/inference.py treats a local --model-draft that is not on disk as
"no drafter loads and none is charged", so speculation silently reverts to
none. The picker re-validates when it opens and marks a missing pin, but that
only covers the UI path. Closing this properly means editing the upstream load
path, which this sub-project's seam budget (one router-registration line)
deliberately declines. Revisit if the seam constraint is ever relaxed.
```

- [ ] **Step 3: Typecheck and commit**

```bash
cd studio/frontend && npx tsc -b --noEmit
git add studio/frontend/src/features/chat/components/draft-model-picker.tsx studio/backend/routes/draft_model.py
git commit -m "feat(draft-model): re-validate a pin on open; record the load-time fail-open"
```

---

## Final verification

- [ ] Run all three backend test files (never the full suite):

```bash
cd studio/backend
python -m pytest tests/test_draft_model_compose.py tests/test_draft_model_validate.py tests/test_draft_model_routes.py -v
```

- [ ] Confirm the seam is exactly two lines:

```bash
git diff --stat main..HEAD -- studio/backend/main.py
```

Expected: `1 file changed, 2 insertions(+)`.

- [ ] Confirm no forbidden file was touched:

```bash
git diff --name-only main..HEAD -- studio/backend/core/inference/tools.py \
    studio/backend/routes/inference.py studio/backend/core/inference/llama_cpp.py pyproject.toml
```

Expected: empty output.

- [ ] State explicitly which negative controls were proven to fail before their fix, and which (if any) turned out inert.
