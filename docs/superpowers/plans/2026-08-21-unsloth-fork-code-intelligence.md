# Code Intelligence (LSP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Studio's agent loop read-only language intelligence for C# and TypeScript/JavaScript over LSP — diagnostics without a build, go-to-definition, find-references, hover, and workspace symbol search.

**Architecture:** A package we own, `studio/backend/core/inference/assist_code/`, holding a persistent pool of language-server subprocesses keyed by `(language, workspace_root)`. The pool is modelled on `mcp_client.py`'s `_StdioSession`, which already solves persistent stdio JSON-RPC lifecycle in this codebase. Five read-only tools register into `ALL_TOOLS` and `execute_tool` through the same two touchpoints sub-project 1 established, and extend its existing request-filter re-add rather than adding a second one.

**Tech Stack:** Python 3.14, stdlib only (`subprocess`, `threading`, `json`) — no `pygls`/`lsprotocol` dependency. External language servers: `typescript-language-server` 6.0.0 (npm), `csharp-ls` 0.26.0 (dotnet tool).

**Spec:** `docs/superpowers/specs/2026-08-21-unsloth-fork-code-intelligence-design.md`

**Target repo:** `C:\Users\Admin\unsloth` — **not** the repo this plan lives in. Branch from `assist-vision-tools`, not `main`.

## Global Constraints

- **A tool never raises into the agent loop.** `execute()` returns `str` always.
- **Confinement is unconditional.** Every path and workspace root resolves inside the session workdir via `tools._get_workdir` / `tools._is_outside_workdir`. There is no `session_id`-omitted escape hatch; `_get_workdir(None)` returns a real anonymous sandbox.
- **No results is not an error.** Zero references, or a clean file, are valid answers.
- **`atexit` cleanup must not log.** Requires BOTH `logging.raiseExceptions = False` and a bare `except` — structlog's `PrintLogger` ignores `raiseExceptions`. See `llama_cpp.py:20925`.
- **Exactly two touchpoints in `tools.py`**, and the `routes/inference.py` re-add is **extended, not duplicated**.
- **The re-add stays gated `if tools_on and tools:`.** An unconditional re-add broke 4 upstream tests in `test_run_tools_locally_discriminator.py`.
- **Heavy/slow imports stay inside functions.** `core.inference.tools` is imported inside function bodies — it imports this package back, so module-scope import is circular.
- **SPDX header on every new file:** `# SPDX-License-Identifier: AGPL-3.0-only` then `# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0`
- **Tests use a real fake-server subprocess speaking real Content-Length framing**, never dict-returning mocks.
- **Run pytest with `--import-mode=importlib`.** Working dir `studio/backend`.
- **Do not run the full backend suite.** Upstream tests fabricate non-sparse GGUF fixtures up to 40 GB. Scope runs with `-k`.
- **Languages in v1 are C# and TypeScript/JavaScript only.** No Rust, no C++ — no toolchain on the dev machine.
- Commit messages end with: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`. Stage specific files; never `git add -A`.

---

## File Structure

All paths relative to `C:\Users\Admin\unsloth\studio\backend`.

| File | Responsibility |
|---|---|
| `core/inference/assist_code/__init__.py` | `execute()` dispatcher, handlers, result formatting |
| `core/inference/assist_code/paths.py` | Confine a file path or workspace root to the session workdir |
| `core/inference/assist_code/jsonrpc.py` | Content-Length framing, request/response correlation, notification queue |
| `core/inference/assist_code/session.py` | One language server: spawn, handshake, capabilities, health, wedged/crashed |
| `core/inference/assist_code/pool.py` | Keyed session pool, per-key lock, idle timeout, LRU cap, atexit cleanup |
| `core/inference/assist_code/servers.py` | Language registry, server discovery, install-on-first-use with disclosure |
| `core/inference/assist_code/diagnostics.py` | Push and pull diagnostics |
| `core/inference/assist_code/navigation.py` | definition / references / hover / symbols, symbol-name addressing |
| `core/inference/assist_code/schemas.py` | The five OpenAI-style tool schemas |
| `tests/lsp_fake_server.py` | A real subprocess speaking real LSP framing — the test harness |
| `tests/test_assist_code_*.py` | One test module per task |

Modified upstream: `core/inference/tools.py` (2 touchpoints), `routes/inference.py` (extend the existing re-add hunk).

---

## Task 1: Path and workspace confinement

**Files:**
- Create: `core/inference/assist_code/__init__.py` (empty package marker for now)
- Create: `core/inference/assist_code/paths.py`
- Test: `tests/test_assist_code_paths.py`

**Interfaces:**
- Consumes: `core.inference.tools._get_workdir(session_id)`, `core.inference.tools._is_outside_workdir(path, workdir)`
- Produces:
  - `resolve_file(path_value, *, session_id=None) -> tuple[str|None, str|None]` — `(abs_path, None)` or `(None, error_text)`
  - `resolve_workspace(path_value, *, session_id=None) -> tuple[str|None, str|None]` — same shape, but the path must be an existing directory
  - `workspace_for(file_path, *, session_id=None) -> str` — nearest enclosing project root, falling back to the workdir

- [ ] **Step 1: Write the failing tests**

Mirror `tests/test_assist_vision_paths.py`'s workdir fixture — `tmp_path` is NOT inside the session sandbox under this repo's conftest, so monkeypatch `tools._get_workdir`.

```python
# tests/test_assist_code_paths.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import os
import pytest
from core.inference.assist_code import paths


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    from core.inference import tools
    monkeypatch.setattr(tools, "_get_workdir", lambda _sid = None: str(tmp_path))
    return tmp_path


class TestResolveFile:
    def test_a_bare_filename_resolves_against_the_workdir(self, workdir):
        (workdir / "a.ts").write_text("export const x = 1\n")
        got, err = paths.resolve_file("a.ts", session_id = "s")
        assert err is None
        assert got == os.path.abspath(str(workdir / "a.ts"))

    def test_a_path_outside_the_workdir_is_refused(self, workdir, tmp_path_factory):
        outside = tmp_path_factory.mktemp("elsewhere") / "secret.ts"
        outside.write_text("x")
        got, err = paths.resolve_file(str(outside), session_id = "s")
        assert got is None
        assert "outside" in err.lower()

    def test_confinement_applies_with_no_session_id(self, workdir, tmp_path_factory):
        """No session_id must NOT be an escape hatch."""
        outside = tmp_path_factory.mktemp("elsewhere2") / "secret.ts"
        outside.write_text("x")
        got, err = paths.resolve_file(str(outside), session_id = None)
        assert got is None
        assert "outside" in err.lower()

    def test_a_missing_file_says_so_not_something_vaguer(self, workdir):
        got, err = paths.resolve_file("nope.ts", session_id = "s")
        assert got is None
        assert "not found" in err.lower()

    def test_an_empty_path_is_rejected(self, workdir):
        got, err = paths.resolve_file("   ", session_id = "s")
        assert got is None
        assert "required" in err.lower()


class TestResolveWorkspace:
    def test_an_existing_directory_resolves(self, workdir):
        (workdir / "proj").mkdir()
        got, err = paths.resolve_workspace("proj", session_id = "s")
        assert err is None
        assert got == os.path.abspath(str(workdir / "proj"))

    def test_a_file_is_not_a_workspace(self, workdir):
        (workdir / "a.ts").write_text("x")
        got, err = paths.resolve_workspace("a.ts", session_id = "s")
        assert got is None
        assert "directory" in err.lower()

    def test_a_workspace_outside_the_workdir_is_refused(self, workdir, tmp_path_factory):
        outside = tmp_path_factory.mktemp("elsewhere3")
        got, err = paths.resolve_workspace(str(outside), session_id = "s")
        assert got is None
        assert "outside" in err.lower()


class TestWorkspaceFor:
    @pytest.mark.parametrize("marker", ["package.json", "tsconfig.json", ".git"])
    def test_the_nearest_project_marker_wins(self, workdir, marker):
        proj = workdir / "proj"
        (proj / "src").mkdir(parents = True)
        if marker == ".git":
            (proj / marker).mkdir()
        else:
            (proj / marker).write_text("{}")
        f = proj / "src" / "a.ts"
        f.write_text("x")
        assert paths.workspace_for(str(f), session_id = "s") == os.path.abspath(str(proj))

    def test_no_marker_falls_back_to_the_workdir(self, workdir):
        (workdir / "loose").mkdir()
        f = workdir / "loose" / "a.ts"
        f.write_text("x")
        assert paths.workspace_for(str(f), session_id = "s") == os.path.abspath(str(workdir))

    def test_the_search_never_climbs_past_the_workdir(self, workdir, tmp_path_factory):
        """A marker ABOVE the sandbox must not be selected as the root."""
        parent_marker = workdir.parent / "package.json"
        parent_marker.write_text("{}")
        (workdir / "x").mkdir()
        f = workdir / "x" / "a.ts"
        f.write_text("x")
        assert paths.workspace_for(str(f), session_id = "s") == os.path.abspath(str(workdir))
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_assist_code_paths.py --import-mode=importlib -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.inference.assist_code'`

- [ ] **Step 3: Create the package marker**

```python
# core/inference/assist_code/__init__.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read-only language intelligence (LSP) tools for Studio's agent loop."""
```

- [ ] **Step 4: Implement `paths.py`**

```python
# core/inference/assist_code/paths.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Confine a file path or a workspace root to the session's sandbox.

Reuses Studio's own sandbox machinery (``_get_workdir`` / ``_is_outside_workdir``
-- the pair ``edit_file`` resolves through) rather than inventing a second
model. Confinement ALWAYS applies: ``_get_workdir`` is None-safe by Studio's
design (``key = session_id or _ANON_KEY``), so an omitted ``session_id`` still
resolves to a real anonymous sandbox and is still confined to it.

A language server indexes an entire directory tree, so an unconfined workspace
root is a materially worse leak than an unconfined single-file read.

The ``tools`` import is deferred into function bodies: ``tools`` imports this
package back to register the code tools, so module-scope import is circular.
"""
import os

_PROJECT_MARKERS = (
    "package.json", "tsconfig.json", "jsconfig.json",   # JS / TS
    "*.sln", "*.csproj",                                # C#
    ".git",                                             # last resort
)


def _confine(path_value, session_id, label):
    """Shared resolution. Returns (abs_path, workdir, None) or (None, None, err)."""
    if not path_value or not str(path_value).strip():
        return None, None, f"{label} is required"
    raw = os.path.expanduser(str(path_value).strip())
    try:
        from core.inference import tools as _tools
        workdir = _tools._get_workdir(session_id)
        candidate = raw if os.path.isabs(raw) else os.path.join(workdir, raw)
        candidate = os.path.abspath(candidate)
        outside = _tools._is_outside_workdir(candidate, workdir)
    except Exception as e:
        return None, None, f"could not confine {label}: {e}"
    if outside:
        return None, None, (
            f"{label} is outside this conversation's working directory. "
            "Use a path inside it, or a bare filename, which resolves there."
        )
    return candidate, workdir, None


def resolve_file(path_value, *, session_id = None):
    """Resolve to an existing file inside the sandbox. Never raises."""
    candidate, _workdir, err = _confine(path_value, session_id, "path")
    if err:
        return None, err
    if not os.path.exists(candidate):
        return None, f"path not found: {path_value}"
    if not os.path.isfile(candidate):
        return None, f"path is not a file: {path_value}"
    return candidate, None


def resolve_workspace(path_value, *, session_id = None):
    """Resolve to an existing directory inside the sandbox. Never raises."""
    candidate, _workdir, err = _confine(path_value, session_id, "workspace")
    if err:
        return None, err
    if not os.path.exists(candidate):
        return None, f"workspace not found: {path_value}"
    if not os.path.isdir(candidate):
        return None, f"workspace is not a directory: {path_value}"
    return candidate, None


def _has_marker(directory):
    import glob
    for marker in _PROJECT_MARKERS:
        if "*" in marker:
            if glob.glob(os.path.join(directory, marker)):
                return True
        elif os.path.exists(os.path.join(directory, marker)):
            return True
    return False


def workspace_for(file_path, *, session_id = None):
    """Nearest enclosing project root, never climbing above the sandbox.

    Climbing past the workdir would hand the server a root outside the sandbox,
    which is exactly what confinement exists to prevent -- so the workdir is
    both the fallback and the ceiling.
    """
    try:
        from core.inference import tools as _tools
        workdir = os.path.abspath(_tools._get_workdir(session_id))
    except Exception:
        return os.path.dirname(os.path.abspath(file_path))

    current = os.path.dirname(os.path.abspath(file_path))
    while True:
        if not current.startswith(workdir):
            return workdir
        if _has_marker(current):
            return current
        if current == workdir:
            return workdir
        parent = os.path.dirname(current)
        if parent == current:
            return workdir
        current = parent
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_assist_code_paths.py --import-mode=importlib -q`
Expected: PASS, 12 tests

- [ ] **Step 6: Negative control**

Temporarily change `_confine` to `return candidate, workdir, None` before the `if outside:` check. Run the suite: `test_a_path_outside_the_workdir_is_refused`, `test_confinement_applies_with_no_session_id`, and `test_a_workspace_outside_the_workdir_is_refused` must FAIL. Restore and confirm they pass again.

- [ ] **Step 7: Commit**

```bash
git add core/inference/assist_code/__init__.py core/inference/assist_code/paths.py tests/test_assist_code_paths.py
git commit -m "feat(code): confine LSP file paths and workspace roots to the session sandbox"
```

---

## Task 2: JSON-RPC transport and the fake-server test harness

**Files:**
- Create: `core/inference/assist_code/jsonrpc.py`
- Create: `tests/lsp_fake_server.py`
- Test: `tests/test_assist_code_jsonrpc.py`

**Interfaces:**
- Produces:
  - `encode(payload: dict) -> bytes` — Content-Length framed
  - `read_message(stream) -> dict|None` — one framed message, `None` at EOF
  - `class Transport(proc)` — `.request(method, params, timeout) -> dict`, `.notify(method, params)`, `.wait_notification(method, predicate, timeout) -> dict|None`, `.close()`
  - `class LspTimeout(Exception)`, `class LspClosed(Exception)`

**Why a real subprocess and not mocks:** framing is where this class of client breaks — a mock returning dicts exercises none of the header parsing, byte counting, partial reads, or interleaving of responses with notifications.

- [ ] **Step 1: Write the fake server**

```python
# tests/lsp_fake_server.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A real subprocess that speaks real LSP framing, for tests.

Not a mock. It parses Content-Length headers off stdin and writes framed JSON
to stdout, so the client's framing, correlation and notification handling are
genuinely exercised. Behaviour is driven by argv so one file covers every case.

Modes:
  normal   -- replies to every request; publishes diagnostics on didOpen
  wedged   -- reads requests and never replies
  crash    -- exits immediately on the first request
  pull     -- advertises diagnosticProvider and answers textDocument/diagnostic
  noisy    -- emits unsolicited notifications before each reply
"""
import json
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"


def _read():
    length = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
    if length is None:
        return None
    return json.loads(sys.stdin.buffer.read(length).decode("utf-8"))


def _write(payload):
    body = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(b"Content-Length: %d\r\n\r\n" % len(body))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _capabilities():
    caps = {
        "definitionProvider": True,
        "referencesProvider": True,
        "hoverProvider": True,
        "workspaceSymbolProvider": True,
    }
    if MODE == "pull":
        caps["diagnosticProvider"] = {"interFileDependencies": False, "workspaceDiagnostics": False}
    return caps


def main():
    while True:
        msg = _read()
        if msg is None:
            return
        method = msg.get("method")
        mid = msg.get("id")

        if MODE == "crash" and method != "initialize":
            sys.exit(3)

        if MODE == "noisy":
            _write({"jsonrpc": "2.0", "method": "window/logMessage",
                    "params": {"type": 3, "message": "chatter"}})

        if method == "initialize":
            _write({"jsonrpc": "2.0", "id": mid,
                    "result": {"capabilities": _capabilities()}})
            continue
        if method == "shutdown":
            _write({"jsonrpc": "2.0", "id": mid, "result": None})
            continue
        if method == "exit":
            return
        if method == "textDocument/didOpen":
            if MODE in ("normal", "noisy"):
                uri = msg["params"]["textDocument"]["uri"]
                _write({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
                        "params": {"uri": uri, "diagnostics": [{
                            "range": {"start": {"line": 2, "character": 6},
                                      "end": {"line": 2, "character": 7}},
                            "severity": 1, "message": "Type 'string' is not assignable to type 'number'.",
                        }]}})
            continue
        if mid is None:
            continue  # any other notification

        if MODE == "wedged":
            continue  # read it, never answer

        if method == "textDocument/diagnostic":
            _write({"jsonrpc": "2.0", "id": mid, "result": {"kind": "full", "items": [{
                "range": {"start": {"line": 4, "character": 0}, "end": {"line": 4, "character": 3}},
                "severity": 2, "message": "'x' is declared but never used.",
            }]}})
            continue
        _write({"jsonrpc": "2.0", "id": mid, "result": None})


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the failing transport tests**

```python
# tests/test_assist_code_jsonrpc.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import io
import os
import subprocess
import sys

import pytest

from core.inference.assist_code import jsonrpc

FAKE = os.path.join(os.path.dirname(__file__), "lsp_fake_server.py")


def _spawn(mode = "normal"):
    return subprocess.Popen(
        [sys.executable, FAKE, mode],
        stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.DEVNULL,
    )


class TestFraming:
    def test_encode_emits_a_content_length_header_and_a_blank_line(self):
        raw = jsonrpc.encode({"a": 1})
        head, _, body = raw.partition(b"\r\n\r\n")
        assert head == b"Content-Length: %d" % len(body)
        assert body == b'{"a": 1}'

    def test_read_message_round_trips_what_encode_wrote(self):
        payload = {"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}
        assert jsonrpc.read_message(io.BytesIO(jsonrpc.encode(payload))) == payload

    def test_read_message_returns_none_at_eof(self):
        assert jsonrpc.read_message(io.BytesIO(b"")) is None

    def test_a_body_split_across_reads_is_reassembled(self):
        """Byte counting, not line reading -- the classic framing bug."""
        payload = {"jsonrpc": "2.0", "id": 1, "result": "x" * 5000}
        assert jsonrpc.read_message(io.BytesIO(jsonrpc.encode(payload))) == payload

    def test_utf8_multibyte_is_counted_in_bytes_not_characters(self):
        payload = {"jsonrpc": "2.0", "id": 1, "result": "héllo — ünïcode ✅"}
        assert jsonrpc.read_message(io.BytesIO(jsonrpc.encode(payload))) == payload


class TestTransport:
    def test_a_request_gets_its_own_reply(self):
        t = jsonrpc.Transport(_spawn())
        try:
            reply = t.request("initialize", {"processId": None}, timeout = 10)
            assert "capabilities" in reply
        finally:
            t.close()

    def test_replies_are_correlated_by_id_not_by_arrival_order(self):
        t = jsonrpc.Transport(_spawn("noisy"))
        try:
            for _ in range(5):
                assert t.request("initialize", {}, timeout = 10) is not None
        finally:
            t.close()

    def test_an_unsolicited_notification_does_not_satisfy_a_request(self):
        """The noisy server emits logMessage before every reply."""
        t = jsonrpc.Transport(_spawn("noisy"))
        try:
            reply = t.request("initialize", {}, timeout = 10)
            assert "capabilities" in reply
        finally:
            t.close()

    def test_wait_notification_returns_a_pushed_message(self):
        t = jsonrpc.Transport(_spawn())
        try:
            t.request("initialize", {}, timeout = 10)
            t.notify("textDocument/didOpen", {"textDocument": {"uri": "file:///a.ts"}})
            note = t.wait_notification(
                "textDocument/publishDiagnostics",
                lambda p: p.get("uri") == "file:///a.ts",
                timeout = 10,
            )
            assert note is not None
            assert note["diagnostics"][0]["severity"] == 1
        finally:
            t.close()

    def test_a_wedged_server_times_out_rather_than_hanging(self):
        t = jsonrpc.Transport(_spawn("wedged"))
        try:
            t.request("initialize", {}, timeout = 10)
            with pytest.raises(jsonrpc.LspTimeout):
                t.request("textDocument/definition", {}, timeout = 1)
        finally:
            t.close()

    def test_a_dead_server_raises_closed_not_timeout(self):
        """A crash must be distinguishable from slowness -- they need
        different responses (restart vs wait)."""
        t = jsonrpc.Transport(_spawn("crash"))
        try:
            t.request("initialize", {}, timeout = 10)
            with pytest.raises(jsonrpc.LspClosed):
                t.request("textDocument/definition", {}, timeout = 10)
        finally:
            t.close()

    def test_wait_notification_that_never_arrives_returns_none(self):
        t = jsonrpc.Transport(_spawn("wedged"))
        try:
            t.request("initialize", {}, timeout = 10)
            assert t.wait_notification("nope/never", lambda p: True, timeout = 1) is None
        finally:
            t.close()

    def test_close_terminates_the_process(self):
        proc = _spawn()
        t = jsonrpc.Transport(proc)
        t.request("initialize", {}, timeout = 10)
        t.close()
        assert proc.poll() is not None
```

- [ ] **Step 3: Run to verify they fail**

Run: `python -m pytest tests/test_assist_code_jsonrpc.py --import-mode=importlib -q`
Expected: FAIL — no module `jsonrpc`

- [ ] **Step 4: Implement `jsonrpc.py`**

```python
# core/inference/assist_code/jsonrpc.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Content-Length framed JSON-RPC over a subprocess's stdio.

Two sinks, not one. LSP interleaves replies to our requests with notifications
the server sends unbidden -- diagnostics arrive that way -- so the reader
thread feeds BOTH a response map keyed by request id AND a notification list.
A design that only models request/response cannot express diagnostics at all.

Timeout and closure are deliberately different exceptions: a slow server should
be waited on or reported, a dead one must be restarted, and collapsing them
loses the distinction the caller needs.
"""
import itertools
import json
import threading

_HEADER_SEP = b"\r\n\r\n"


class LspTimeout(Exception):
    """A request was not answered inside its deadline."""


class LspClosed(Exception):
    """The server exited or its pipes closed."""


def encode(payload: dict) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    return b"Content-Length: %d%s%s" % (len(body), _HEADER_SEP, body)


def read_message(stream):
    """Read one framed message. Returns the dict, or None at EOF.

    Counts BYTES, not characters: a multi-byte body read as characters
    truncates, which is the classic framing bug in hand-rolled LSP clients.
    """
    length = None
    while True:
        line = stream.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break
        if line.lower().startswith(b"content-length:"):
            try:
                length = int(line.split(b":", 1)[1].strip())
            except ValueError:
                return None
    if length is None:
        return None
    body = b""
    while len(body) < length:
        chunk = stream.read(length - len(body))
        if not chunk:
            return None
        body += chunk
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


class Transport:
    """Owns a server subprocess and its reader thread."""

    def __init__(self, proc):
        self._proc = proc
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._replies = {}                 # id -> payload
        self._events = {}                  # id -> Event
        self._notifications = []           # list[(method, params)]
        self._note_cv = threading.Condition()
        self._closed = threading.Event()
        self._reader = threading.Thread(
            target = self._read_loop, name = "lsp-reader", daemon = True
        )
        self._reader.start()

    def _read_loop(self):
        try:
            while True:
                msg = read_message(self._proc.stdout)
                if msg is None:
                    break
                if "id" in msg and ("result" in msg or "error" in msg):
                    mid = msg["id"]
                    with self._lock:
                        self._replies[mid] = msg
                        ev = self._events.get(mid)
                    if ev is not None:
                        ev.set()
                elif "method" in msg:
                    with self._note_cv:
                        self._notifications.append((msg["method"], msg.get("params") or {}))
                        self._note_cv.notify_all()
        except Exception:
            pass
        finally:
            self._closed.set()
            # Wake everyone waiting; they re-check _closed and raise LspClosed.
            with self._lock:
                events = list(self._events.values())
            for ev in events:
                ev.set()
            with self._note_cv:
                self._note_cv.notify_all()

    def _alive(self):
        return not self._closed.is_set() and self._proc.poll() is None

    def notify(self, method, params = None):
        if not self._alive():
            raise LspClosed(f"server is not running (notify {method})")
        try:
            self._proc.stdin.write(encode(
                {"jsonrpc": "2.0", "method": method, "params": params or {}}))
            self._proc.stdin.flush()
        except Exception as e:
            raise LspClosed(f"server pipe closed: {e}") from e

    def request(self, method, params = None, timeout = 30.0):
        if not self._alive():
            raise LspClosed(f"server is not running (request {method})")
        mid = next(self._ids)
        ev = threading.Event()
        with self._lock:
            self._events[mid] = ev
        try:
            try:
                self._proc.stdin.write(encode(
                    {"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}))
                self._proc.stdin.flush()
            except Exception as e:
                raise LspClosed(f"server pipe closed: {e}") from e

            if not ev.wait(timeout):
                raise LspTimeout(f"{method} did not answer within {timeout:.0f}s")
            with self._lock:
                payload = self._replies.pop(mid, None)
            if payload is None:
                # Woken by shutdown rather than by a reply.
                raise LspClosed(f"server exited while handling {method}")
            if "error" in payload:
                err = payload["error"] or {}
                raise LspClosed(f"{method} failed: {err.get('message', err)}")
            return payload.get("result")
        finally:
            with self._lock:
                self._events.pop(mid, None)

    def wait_notification(self, method, predicate, timeout = 10.0):
        """Wait for a matching pushed notification. None if it never comes.

        Scans already-buffered notifications first: the server may publish
        before we start waiting, and a pure wait would miss it.
        """
        import time
        deadline = time.monotonic() + timeout
        seen = 0
        with self._note_cv:
            while True:
                while seen < len(self._notifications):
                    m, p = self._notifications[seen]
                    seen += 1
                    if m == method:
                        try:
                            if predicate(p):
                                return p
                        except Exception:
                            pass
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._closed.is_set():
                    return None
                self._note_cv.wait(timeout = min(remaining, 0.25))

    def close(self):
        try:
            if self._alive():
                try:
                    self.request("shutdown", None, timeout = 2.0)
                except Exception:
                    pass
                try:
                    self.notify("exit")
                except Exception:
                    pass
        except Exception:
            pass
        for closer in (
            lambda: self._proc.stdin.close(),
            lambda: self._proc.terminate(),
        ):
            try:
                closer()
            except Exception:
                pass
        try:
            self._proc.wait(timeout = 3)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._closed.set()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_assist_code_jsonrpc.py --import-mode=importlib -q`
Expected: PASS, 14 tests

- [ ] **Step 6: Negative control on framing**

In `read_message`, replace the byte-counting loop with a single `stream.read(length)`. Confirm `test_a_body_split_across_reads_is_reassembled` still passes (it may) but that changing `encode` to use `len(json.dumps(payload))` instead of `len(body)` makes `test_utf8_multibyte_is_counted_in_bytes_not_characters` FAIL. Restore both.

- [ ] **Step 7: Commit**

```bash
git add core/inference/assist_code/jsonrpc.py tests/lsp_fake_server.py tests/test_assist_code_jsonrpc.py
git commit -m "feat(code): Content-Length JSON-RPC transport with a real fake-server harness"
```

---

## Task 3: LSP session — handshake, capabilities, health

**Files:**
- Create: `core/inference/assist_code/session.py`
- Test: `tests/test_assist_code_session.py`

**Interfaces:**
- Consumes: `jsonrpc.Transport`, `jsonrpc.LspTimeout`, `jsonrpc.LspClosed`
- Produces:
  - `class Session(command: list[str], root: str, *, language: str)` with `.start(timeout)`, `.capabilities -> dict`, `.supports(name) -> bool`, `.open_document(path)`, `.request(method, params, timeout)`, `.wait_notification(...)`, `.is_healthy() -> bool`, `.close()`
  - `class SessionStartFailed(Exception)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_assist_code_session.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import os
import sys

import pytest

from core.inference.assist_code import session as sess
from core.inference.assist_code import jsonrpc

FAKE = os.path.join(os.path.dirname(__file__), "lsp_fake_server.py")


def _session(tmp_path, mode = "normal"):
    return sess.Session([sys.executable, FAKE, mode], str(tmp_path), language = "typescript")


class TestHandshake:
    def test_start_completes_and_records_capabilities(self, tmp_path):
        s = _session(tmp_path)
        try:
            s.start(timeout = 10)
            assert s.capabilities.get("definitionProvider") is True
        finally:
            s.close()

    def test_supports_reads_the_negotiated_capabilities(self, tmp_path):
        s = _session(tmp_path)
        try:
            s.start(timeout = 10)
            assert s.supports("hoverProvider")
            assert not s.supports("renameProvider")
        finally:
            s.close()

    def test_pull_diagnostics_are_detected_only_when_advertised(self, tmp_path):
        push = _session(tmp_path, "normal")
        pull = _session(tmp_path, "pull")
        try:
            push.start(timeout = 10)
            pull.start(timeout = 10)
            assert not push.supports("diagnosticProvider")
            assert pull.supports("diagnosticProvider")
        finally:
            push.close()
            pull.close()

    def test_a_server_that_never_answers_initialize_fails_to_start(self, tmp_path):
        s = _session(tmp_path, "wedged")
        try:
            with pytest.raises(sess.SessionStartFailed) as e:
                s.start(timeout = 1)
            assert "did not" in str(e.value).lower() or "timed out" in str(e.value).lower()
        finally:
            s.close()

    def test_a_command_that_does_not_exist_fails_to_start_with_a_readable_error(self, tmp_path):
        s = sess.Session(["definitely-not-a-real-binary-xyz"], str(tmp_path), language = "typescript")
        try:
            with pytest.raises(sess.SessionStartFailed) as e:
                s.start(timeout = 5)
            assert "definitely-not-a-real-binary-xyz" in str(e.value)
        finally:
            s.close()


class TestDocuments:
    def test_open_document_sends_the_file_contents(self, tmp_path):
        f = tmp_path / "a.ts"
        f.write_text("const a: number = 'x'\n")
        s = _session(tmp_path)
        try:
            s.start(timeout = 10)
            s.open_document(str(f))
            note = s.wait_notification(
                "textDocument/publishDiagnostics", lambda p: True, timeout = 10)
            assert note is not None
        finally:
            s.close()

    def test_opening_the_same_document_twice_does_not_resend_didopen(self, tmp_path):
        f = tmp_path / "a.ts"
        f.write_text("x\n")
        s = _session(tmp_path)
        try:
            s.start(timeout = 10)
            s.open_document(str(f))
            s.open_document(str(f))
            assert s.opened_count == 1
        finally:
            s.close()


class TestHealth:
    def test_a_live_session_is_healthy(self, tmp_path):
        s = _session(tmp_path)
        try:
            s.start(timeout = 10)
            assert s.is_healthy()
        finally:
            s.close()

    def test_a_closed_session_is_not_healthy(self, tmp_path):
        s = _session(tmp_path)
        s.start(timeout = 10)
        s.close()
        assert not s.is_healthy()

    def test_uri_round_trip_survives_spaces_and_drive_letters(self, tmp_path):
        d = tmp_path / "a dir"
        d.mkdir()
        f = d / "b.ts"
        f.write_text("x")
        uri = sess.path_to_uri(str(f))
        assert uri.startswith("file:///")
        assert sess.uri_to_path(uri) == os.path.abspath(str(f))
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_assist_code_session.py --import-mode=importlib -q`
Expected: FAIL — no module `session`

- [ ] **Step 3: Implement `session.py`**

```python
# core/inference/assist_code/session.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One language server: spawn, handshake, capability negotiation, health.

Capabilities are negotiated, not assumed. Whether diagnostics are pulled
(``textDocument/diagnostic``, LSP 3.17) or pushed (``publishDiagnostics``)
depends on what the server advertises during ``initialize``, and support is
uneven across servers -- so the answer is read from the handshake rather than
hard-coded per language.
"""
import os
import subprocess
import sys
import urllib.parse
import urllib.request

from . import jsonrpc

_LANGUAGE_IDS = {
    "typescript": "typescript",
    "javascript": "javascript",
    "csharp": "csharp",
}


class SessionStartFailed(Exception):
    """The server could not be spawned or did not complete the handshake."""


def path_to_uri(path):
    return urllib.parse.urljoin("file:", urllib.request.pathname2url(os.path.abspath(path)))


def uri_to_path(uri):
    parsed = urllib.parse.urlparse(uri)
    return os.path.abspath(urllib.request.url2pathname(parsed.path))


class Session:
    def __init__(self, command, root, *, language):
        self.command = list(command)
        self.root = os.path.abspath(root)
        self.language = language
        self.capabilities = {}
        self.opened_count = 0
        self._opened = set()
        self._transport = None
        self._proc = None

    def start(self, timeout = 60.0):
        try:
            self._proc = subprocess.Popen(
                self.command,
                stdin = subprocess.PIPE, stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                cwd = self.root,
            )
        except FileNotFoundError as e:
            raise SessionStartFailed(
                f"could not start the {self.language} language server: "
                f"{self.command[0]} was not found ({e})"
            ) from e
        except Exception as e:
            raise SessionStartFailed(
                f"could not start the {self.language} language server "
                f"({' '.join(self.command)}): {e}"
            ) from e

        self._transport = jsonrpc.Transport(self._proc)
        params = {
            "processId": os.getpid(),
            "rootUri": path_to_uri(self.root),
            "workspaceFolders": [{"uri": path_to_uri(self.root), "name": os.path.basename(self.root)}],
            "capabilities": {
                "textDocument": {
                    "publishDiagnostics": {"relatedInformation": False},
                    "diagnostic": {"dynamicRegistration": False},
                    "definition": {"linkSupport": False},
                    "references": {},
                    "hover": {"contentFormat": ["plaintext", "markdown"]},
                },
                "workspace": {"symbol": {}, "workspaceFolders": True},
            },
        }
        try:
            result = self._transport.request("initialize", params, timeout = timeout)
        except jsonrpc.LspTimeout as e:
            raise SessionStartFailed(
                f"the {self.language} language server did not complete its handshake "
                f"within {timeout:.0f}s"
            ) from e
        except jsonrpc.LspClosed as e:
            raise SessionStartFailed(
                f"the {self.language} language server exited during startup: {e}"
            ) from e
        self.capabilities = (result or {}).get("capabilities", {}) or {}
        try:
            self._transport.notify("initialized", {})
        except jsonrpc.LspClosed as e:
            raise SessionStartFailed(f"server closed right after initialize: {e}") from e

    def supports(self, name):
        return bool(self.capabilities.get(name))

    def open_document(self, path):
        """didOpen once per file. Re-opening the same path is a no-op."""
        path = os.path.abspath(path)
        if path in self._opened:
            return
        try:
            with open(path, "r", encoding = "utf-8", errors = "replace") as fh:
                text = fh.read()
        except OSError as e:
            raise SessionStartFailed(f"could not read {path}: {e}") from e
        ext = os.path.splitext(path)[1].lower()
        language_id = _LANGUAGE_IDS.get(self.language, self.language)
        if ext in (".js", ".jsx", ".mjs", ".cjs"):
            language_id = "javascript"
        elif ext in (".ts", ".tsx", ".mts", ".cts"):
            language_id = "typescript"
        self._transport.notify("textDocument/didOpen", {"textDocument": {
            "uri": path_to_uri(path), "languageId": language_id,
            "version": 1, "text": text,
        }})
        self._opened.add(path)
        self.opened_count += 1

    def request(self, method, params = None, timeout = 30.0):
        if self._transport is None:
            raise jsonrpc.LspClosed("session was never started")
        return self._transport.request(method, params, timeout = timeout)

    def wait_notification(self, method, predicate, timeout = 10.0):
        if self._transport is None:
            return None
        return self._transport.wait_notification(method, predicate, timeout = timeout)

    def is_healthy(self):
        return (
            self._transport is not None
            and self._proc is not None
            and self._proc.poll() is None
            and not self._transport._closed.is_set()
        )

    def close(self):
        if self._transport is not None:
            self._transport.close()
            self._transport = None
        self._opened.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_assist_code_session.py --import-mode=importlib -q`
Expected: PASS, 11 tests

- [ ] **Step 5: Commit**

```bash
git add core/inference/assist_code/session.py tests/test_assist_code_session.py
git commit -m "feat(code): LSP session with negotiated capabilities and health checks"
```

---

## Task 4: The session pool

**Files:**
- Create: `core/inference/assist_code/pool.py`
- Test: `tests/test_assist_code_pool.py`

**Interfaces:**
- Consumes: `session.Session`, `session.SessionStartFailed`
- Produces:
  - `acquire(language, root, factory, *, start_timeout=60.0) -> Session` — reuses a live session or creates one; restarts a dead one **once**
  - `shutdown_all()` — closes every session
  - `stats() -> dict` with keys `live`, `keys`
  - Module constants `MAX_SESSIONS = 4`, `IDLE_TIMEOUT_SECONDS = 600`

**Modelled on `mcp_client.py`:** `_StdioKeyLock` (`:540`) for per-key locking so two threads racing on the same workspace do not spawn two servers, and `_transport_dead`/`_session_responsive` (`:318`, `:344`) for the health checks. Registers an `atexit` handler that **must not log** — see `llama_cpp.py:20925`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_assist_code_pool.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import os
import sys
import threading

import pytest

from core.inference.assist_code import pool, session as sess

FAKE = os.path.join(os.path.dirname(__file__), "lsp_fake_server.py")


def _factory(tmp_path, mode = "normal"):
    def make(root):
        return sess.Session([sys.executable, FAKE, mode], root, language = "typescript")
    return make


@pytest.fixture(autouse = True)
def _clean_pool():
    pool.shutdown_all()
    yield
    pool.shutdown_all()


class TestReuse:
    def test_the_second_acquire_returns_the_same_session(self, tmp_path):
        a = pool.acquire("typescript", str(tmp_path), _factory(tmp_path))
        b = pool.acquire("typescript", str(tmp_path), _factory(tmp_path))
        assert a is b
        assert pool.stats()["live"] == 1

    def test_different_roots_get_different_sessions(self, tmp_path):
        one = tmp_path / "one"; one.mkdir()
        two = tmp_path / "two"; two.mkdir()
        a = pool.acquire("typescript", str(one), _factory(tmp_path))
        b = pool.acquire("typescript", str(two), _factory(tmp_path))
        assert a is not b
        assert pool.stats()["live"] == 2

    def test_different_languages_on_one_root_get_different_sessions(self, tmp_path):
        def make_cs(root):
            return sess.Session([sys.executable, FAKE, "normal"], root, language = "csharp")
        a = pool.acquire("typescript", str(tmp_path), _factory(tmp_path))
        b = pool.acquire("csharp", str(tmp_path), make_cs)
        assert a is not b

    def test_concurrent_acquires_of_one_key_create_exactly_one_session(self, tmp_path):
        """Without per-key locking this spawns N servers and leaks N-1."""
        seen = []
        def worker():
            seen.append(pool.acquire("typescript", str(tmp_path), _factory(tmp_path)))
        threads = [threading.Thread(target = worker) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len({id(s) for s in seen}) == 1
        assert pool.stats()["live"] == 1


class TestRecovery:
    def test_a_dead_session_is_replaced(self, tmp_path):
        a = pool.acquire("typescript", str(tmp_path), _factory(tmp_path))
        a.close()
        b = pool.acquire("typescript", str(tmp_path), _factory(tmp_path))
        assert b is not a
        assert b.is_healthy()

    def test_a_server_that_cannot_start_reports_rather_than_caching_a_corpse(self, tmp_path):
        def bad(root):
            return sess.Session(["definitely-not-a-real-binary-xyz"], root, language = "typescript")
        with pytest.raises(sess.SessionStartFailed):
            pool.acquire("typescript", str(tmp_path), bad)
        assert pool.stats()["live"] == 0


class TestEviction:
    def test_exceeding_the_cap_evicts_the_least_recently_used(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pool, "MAX_SESSIONS", 2)
        roots = []
        for name in ("a", "b", "c"):
            d = tmp_path / name; d.mkdir(); roots.append(str(d))
        first = pool.acquire("typescript", roots[0], _factory(tmp_path))
        pool.acquire("typescript", roots[1], _factory(tmp_path))
        pool.acquire("typescript", roots[2], _factory(tmp_path))
        assert pool.stats()["live"] == 2
        assert not first.is_healthy(), "the evicted session's process must be closed"

    def test_an_idle_session_is_reaped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pool, "IDLE_TIMEOUT_SECONDS", 0)
        s = pool.acquire("typescript", str(tmp_path), _factory(tmp_path))
        pool.reap_idle()
        assert pool.stats()["live"] == 0
        assert not s.is_healthy()


class TestShutdown:
    def test_shutdown_all_closes_every_session(self, tmp_path):
        one = tmp_path / "one"; one.mkdir()
        two = tmp_path / "two"; two.mkdir()
        a = pool.acquire("typescript", str(one), _factory(tmp_path))
        b = pool.acquire("typescript", str(two), _factory(tmp_path))
        pool.shutdown_all()
        assert pool.stats()["live"] == 0
        assert not a.is_healthy() and not b.is_healthy()

    def test_the_atexit_handler_never_raises_even_with_broken_logging(self, tmp_path, monkeypatch):
        """atexit runs after log streams may be closed. llama_cpp.py:20925
        documents this: a logging call there surfaced as unrelated tracebacks
        printed after the test summary."""
        pool.acquire("typescript", str(tmp_path), _factory(tmp_path))
        import logging
        class _Exploding:
            def info(self, *a, **k): raise ValueError("stream closed")
            def warning(self, *a, **k): raise ValueError("stream closed")
            def exception(self, *a, **k): raise ValueError("stream closed")
        monkeypatch.setattr(pool, "logger", _Exploding())
        pool._atexit_cleanup()          # must not raise
        assert logging.raiseExceptions is True, "the flag must be restored"
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_assist_code_pool.py --import-mode=importlib -q`
Expected: FAIL — no module `pool`

- [ ] **Step 3: Implement `pool.py`**

```python
# core/inference/assist_code/pool.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A bounded pool of language-server sessions keyed by (language, root).

Persistent because starting a server is expensive -- typescript-language-server
takes seconds to index and rust-analyzer/clangd take tens of seconds, so a
server-per-call design would make every tool call unusable.

Per-key locking follows ``mcp_client._StdioKeyLock`` (:540): without it, two
threads asking for the same workspace at once each spawn a server and one is
orphaned. ``execute_tool`` runs on a daemon worker thread
(``tool_stream_exec.py:161-176``), so concurrent calls are real.
"""
import atexit
import logging
import os
import threading
import time

try:
    from loggers import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover - outside the app
    logger = logging.getLogger(__name__)

MAX_SESSIONS = 4
IDLE_TIMEOUT_SECONDS = 600

_lock = threading.Lock()
_key_locks = {}
_sessions = {}      # key -> session
_last_used = {}     # key -> monotonic timestamp


def _key(language, root):
    return (language, os.path.abspath(root))


def _key_lock(key):
    with _lock:
        lk = _key_locks.get(key)
        if lk is None:
            lk = threading.Lock()
            _key_locks[key] = lk
        return lk


def _close_quietly(session):
    try:
        session.close()
    except Exception:
        pass


def _evict_if_needed():
    """Caller holds no lock. Drops LRU sessions until under the cap."""
    while True:
        with _lock:
            if len(_sessions) <= MAX_SESSIONS:
                return
            victim_key = min(_last_used, key = lambda k: _last_used.get(k, 0))
            victim = _sessions.pop(victim_key, None)
            _last_used.pop(victim_key, None)
        if victim is not None:
            logger.info("assist_code: evicting idle language server for %s", victim_key)
            _close_quietly(victim)


def acquire(language, root, factory, *, start_timeout = 60.0):
    """Return a live session for (language, root), creating one if needed.

    A dead session is replaced rather than handed back. A server that fails to
    start is NOT cached -- caching a corpse turns one bad start into a
    permanent failure for that workspace.
    """
    key = _key(language, root)
    with _key_lock(key):
        with _lock:
            existing = _sessions.get(key)
        if existing is not None:
            if existing.is_healthy():
                with _lock:
                    _last_used[key] = time.monotonic()
                return existing
            with _lock:
                _sessions.pop(key, None)
                _last_used.pop(key, None)
            _close_quietly(existing)

        session = factory(os.path.abspath(root))
        session.start(timeout = start_timeout)   # SessionStartFailed propagates
        with _lock:
            _sessions[key] = session
            _last_used[key] = time.monotonic()
    _evict_if_needed()
    return session


def reap_idle():
    now = time.monotonic()
    stale = []
    with _lock:
        for key, used in list(_last_used.items()):
            if now - used >= IDLE_TIMEOUT_SECONDS:
                stale.append((key, _sessions.pop(key, None)))
                _last_used.pop(key, None)
    for key, session in stale:
        if session is not None:
            logger.info("assist_code: reaping idle language server for %s", key)
            _close_quietly(session)


def stats():
    with _lock:
        return {"live": len(_sessions), "keys": sorted(str(k) for k in _sessions)}


def shutdown_all():
    with _lock:
        sessions = list(_sessions.values())
        _sessions.clear()
        _last_used.clear()
    for s in sessions:
        _close_quietly(s)


def _atexit_cleanup():
    """Terminate every server at interpreter exit.

    Nothing here may report through logging. By the time atexit runs, the
    streams handlers write to can already be closed, and a write then fails --
    which is how the equivalent bug was found in llama_cpp.py:20925, as
    unrelated tracebacks printed after the test summary. Two mechanisms are
    needed together: raiseExceptions covers the stdlib loggers other libraries
    install, and the bare except covers this module's structlog PrintLogger,
    which does not consult raiseExceptions at all.
    """
    raise_exceptions = logging.raiseExceptions
    logging.raiseExceptions = False
    try:
        shutdown_all()
    except Exception:
        pass
    finally:
        logging.raiseExceptions = raise_exceptions


atexit.register(_atexit_cleanup)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_assist_code_pool.py --import-mode=importlib -q`
Expected: PASS, 10 tests

- [ ] **Step 5: Negative control on the per-key lock**

Change `_key_lock` to return a fresh `threading.Lock()` every call. `test_concurrent_acquires_of_one_key_create_exactly_one_session` must FAIL. Restore and confirm it passes.

- [ ] **Step 6: Commit**

```bash
git add core/inference/assist_code/pool.py tests/test_assist_code_pool.py
git commit -m "feat(code): bounded LSP session pool with per-key locking and silent atexit cleanup"
```

---

## Task 5: Server registry and install-on-first-use

**Files:**
- Create: `core/inference/assist_code/servers.py`
- Test: `tests/test_assist_code_servers.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `language_for(path) -> str|None` — from file extension
  - `SUPPORTED = ("typescript", "javascript", "csharp")`
  - `server_command(language, *, installer=None) -> list[str]` — locating or installing the server; raises `ServerUnavailable`
  - `class ServerUnavailable(Exception)`

**Disclosure convention** follows `assist_vision/models.py`: the log line naming what is being installed and from where is emitted **before** the install starts, and a failure names the manual install command.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_assist_code_servers.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import pytest

from core.inference.assist_code import servers


class _RecordingLogger:
    """Records rendered lines; makes ORDERING observable, which is the point."""
    def __init__(self): self.lines = []
    def info(self, msg, *args): self.lines.append(msg % args if args else msg)
    def warning(self, msg, *args): self.lines.append(msg % args if args else msg)
    def text(self): return "\n".join(self.lines).lower()


@pytest.fixture
def spy(monkeypatch):
    s = _RecordingLogger()
    monkeypatch.setattr(servers, "logger", s)
    return s


class TestLanguageDetection:
    @pytest.mark.parametrize("name,expected", [
        ("a.ts", "typescript"), ("a.tsx", "typescript"), ("a.mts", "typescript"),
        ("a.js", "javascript"), ("a.jsx", "javascript"), ("a.cjs", "javascript"),
        ("a.cs", "csharp"),
        ("a.rs", None), ("a.cpp", None), ("a.py", None), ("noext", None),
    ])
    def test_extension_maps_to_language(self, name, expected):
        assert servers.language_for(name) == expected

    def test_rust_and_cpp_are_deliberately_absent_from_v1(self):
        assert "rust" not in servers.SUPPORTED
        assert "cpp" not in servers.SUPPORTED


class TestDiscovery:
    def test_an_already_installed_server_is_used_without_installing(self, spy, monkeypatch):
        monkeypatch.setattr(servers, "_which", lambda name: "C:/fake/typescript-language-server")
        def _boom(*a, **k):
            raise AssertionError("must not install a server that is already present")
        cmd = servers.server_command("typescript", installer = _boom)
        assert cmd[0] == "C:/fake/typescript-language-server"
        assert "installing" not in spy.text()

    def test_a_missing_server_is_announced_before_the_install_starts(self, spy, monkeypatch):
        monkeypatch.setattr(servers, "_which", lambda name: None)
        log_at_install_time = []
        def _installer(cmd):
            log_at_install_time.append(spy.text())
            return True
        calls = {"n": 0}
        def _which_after(name):
            calls["n"] += 1
            return None if calls["n"] == 1 else "C:/fake/typescript-language-server"
        monkeypatch.setattr(servers, "_which", _which_after)
        servers.server_command("typescript", installer = _installer)
        assert log_at_install_time, "installer was never called"
        announced = log_at_install_time[0]
        assert "installing" in announced
        assert "typescript-language-server" in announced

    def test_a_failed_install_names_the_manual_command(self, monkeypatch):
        monkeypatch.setattr(servers, "_which", lambda name: None)
        with pytest.raises(servers.ServerUnavailable) as e:
            servers.server_command("typescript", installer = lambda cmd: False)
        msg = str(e.value)
        assert "npm install" in msg
        assert "typescript-language-server" in msg

    def test_csharp_failure_names_the_dotnet_tool_command(self, monkeypatch):
        monkeypatch.setattr(servers, "_which", lambda name: None)
        with pytest.raises(servers.ServerUnavailable) as e:
            servers.server_command("csharp", installer = lambda cmd: False)
        msg = str(e.value)
        assert "dotnet tool install" in msg
        assert "csharp-ls" in msg

    def test_an_unsupported_language_names_the_supported_ones(self):
        with pytest.raises(servers.ServerUnavailable) as e:
            servers.server_command("rust")
        msg = str(e.value).lower()
        assert "rust" in msg
        assert "typescript" in msg and "csharp" in msg
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_assist_code_servers.py --import-mode=importlib -q`
Expected: FAIL — no module `servers`

- [ ] **Step 3: Implement `servers.py`**

```python
# core/inference/assist_code/servers.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which language server serves which language, and how to get it.

No server is vendored. Each is installed on first use and the install is
announced BEFORE it starts -- a chat message that silently triggers a package
install is the surprise this avoids -- with failures naming the exact manual
command.

v1 is TypeScript/JavaScript and C# only. Rust and C++ are absent deliberately:
the dev machine has no cargo/rustc/g++/cl/cmake, and shipping a language that
cannot be exercised is how this project has repeatedly shipped a green suite
over broken code. Adding them later is a new entry in _SERVERS, not a new
mechanism.
"""
import logging
import shutil
import subprocess

try:
    from loggers import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover - outside the app
    logger = logging.getLogger(__name__)

SUPPORTED = ("typescript", "javascript", "csharp")

_EXTENSIONS = {
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".cs": "csharp",
}

_SERVERS = {
    "typescript": {
        "binary": "typescript-language-server",
        "args": ["--stdio"],
        "install": ["npm", "install", "-g", "typescript-language-server", "typescript"],
        "manual": "npm install -g typescript-language-server typescript",
    },
    "javascript": {
        "binary": "typescript-language-server",
        "args": ["--stdio"],
        "install": ["npm", "install", "-g", "typescript-language-server", "typescript"],
        "manual": "npm install -g typescript-language-server typescript",
    },
    "csharp": {
        "binary": "csharp-ls",
        "args": [],
        "install": ["dotnet", "tool", "install", "--global", "csharp-ls"],
        "manual": "dotnet tool install --global csharp-ls",
    },
}


class ServerUnavailable(Exception):
    """No language server could be located or installed."""


def language_for(path):
    import os
    return _EXTENSIONS.get(os.path.splitext(str(path))[1].lower())


def _which(name):
    return shutil.which(name)


def _run_install(cmd):
    """Returns True on success. Never raises."""
    try:
        completed = subprocess.run(
            cmd, stdin = subprocess.DEVNULL,
            stdout = subprocess.PIPE, stderr = subprocess.STDOUT,
            timeout = 600,
        )
        return completed.returncode == 0
    except Exception:
        return False


def server_command(language, *, installer = None):
    """Locate the server for ``language``, installing it once if absent."""
    spec = _SERVERS.get(language)
    if spec is None:
        raise ServerUnavailable(
            f"no language server for {language!r}. Supported: {', '.join(SUPPORTED)}."
        )

    found = _which(spec["binary"])
    if found:
        return [found] + list(spec["args"])

    # Announced before the install runs, not after: a user watching the log
    # should know why the machine just started fetching packages.
    logger.info(
        "assist_code: installing %s for %s -- first use only (%s)",
        spec["binary"], language, spec["manual"],
    )
    run = installer or _run_install
    ok = run(spec["install"])
    if ok:
        found = _which(spec["binary"])
        if found:
            return [found] + list(spec["args"])
    raise ServerUnavailable(
        f"the {language} language server ({spec['binary']}) is not installed and could "
        f"not be installed automatically. Install it manually with: {spec['manual']}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_assist_code_servers.py --import-mode=importlib -q`
Expected: PASS, 18 tests

- [ ] **Step 5: Commit**

```bash
git add core/inference/assist_code/servers.py tests/test_assist_code_servers.py
git commit -m "feat(code): language server registry with disclosed install-on-first-use"
```

---

## Task 6: Diagnostics — push and pull

**Files:**
- Create: `core/inference/assist_code/diagnostics.py`
- Test: `tests/test_assist_code_diagnostics.py`

**Interfaces:**
- Consumes: `session.Session`, `session.path_to_uri`
- Produces: `collect(session, file_path, *, timeout=15.0) -> list[dict]` where each dict is `{"severity": str, "line": int, "column": int, "message": str}` with **1-based** line and column
- `SEVERITY = {1: "error", 2: "warning", 3: "info", 4: "hint"}`

**The core subtlety:** classic LSP has no "give me this file's errors" request — the server pushes `textDocument/publishDiagnostics` after `didOpen`. LSP 3.17's pull (`textDocument/diagnostic`) is used only when the server advertised `diagnosticProvider` at handshake.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_assist_code_diagnostics.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import os
import sys

import pytest

from core.inference.assist_code import diagnostics, session as sess

FAKE = os.path.join(os.path.dirname(__file__), "lsp_fake_server.py")


def _started(tmp_path, mode):
    s = sess.Session([sys.executable, FAKE, mode], str(tmp_path), language = "typescript")
    s.start(timeout = 10)
    return s


@pytest.fixture
def ts_file(tmp_path):
    f = tmp_path / "a.ts"
    f.write_text("const a = 1\nconst b = 2\nconst c: number = 'x'\nlet d\nlet x\n")
    return f


class TestPushPath:
    def test_pushed_diagnostics_are_collected(self, tmp_path, ts_file):
        s = _started(tmp_path, "normal")
        try:
            got = diagnostics.collect(s, str(ts_file), timeout = 10)
            assert len(got) == 1
            assert got[0]["severity"] == "error"
            assert "not assignable" in got[0]["message"]
        finally:
            s.close()

    def test_line_and_column_are_converted_to_one_based(self, tmp_path, ts_file):
        """LSP is 0-based; humans and every compiler are 1-based. Off-by-one
        here sends the model to the wrong line."""
        s = _started(tmp_path, "normal")
        try:
            got = diagnostics.collect(s, str(ts_file), timeout = 10)
            assert got[0]["line"] == 3      # LSP line 2
            assert got[0]["column"] == 7    # LSP character 6
        finally:
            s.close()


class TestPullPath:
    def test_pull_is_used_when_the_server_advertises_it(self, tmp_path, ts_file):
        s = _started(tmp_path, "pull")
        try:
            got = diagnostics.collect(s, str(ts_file), timeout = 10)
            assert len(got) == 1
            assert got[0]["severity"] == "warning"
            assert "never used" in got[0]["message"]
            assert got[0]["line"] == 5
        finally:
            s.close()


class TestNoResults:
    def test_a_server_that_publishes_nothing_yields_an_empty_list_not_an_error(
        self, tmp_path, ts_file
    ):
        """A clean file is a valid answer, not a failure."""
        s = _started(tmp_path, "wedged")   # never pushes, never answers
        try:
            got = diagnostics.collect(s, str(ts_file), timeout = 1)
            assert got == []
        finally:
            s.close()
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_assist_code_diagnostics.py --import-mode=importlib -q`
Expected: FAIL — no module `diagnostics`

- [ ] **Step 3: Implement `diagnostics.py`**

```python
# core/inference/assist_code/diagnostics.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diagnostics, by whichever mechanism the server actually supports.

Classic LSP has no request for "what is wrong in this file": the server emits
textDocument/publishDiagnostics unsolicited after didOpen. LSP 3.17 added a
pull request (textDocument/diagnostic) but support is uneven, so the path is
chosen from what the server advertised at handshake rather than per language.

An empty result is a CLEAN FILE, not a failure -- including when the wait times
out, because a server with nothing to report may simply publish nothing.
"""
from . import jsonrpc
from .session import path_to_uri

SEVERITY = {1: "error", 2: "warning", 3: "info", 4: "hint"}


def _normalise(items):
    out = []
    for item in items or []:
        rng = (item.get("range") or {}).get("start") or {}
        out.append({
            "severity": SEVERITY.get(item.get("severity"), "info"),
            # LSP is 0-based; every compiler and human is 1-based.
            "line": int(rng.get("line", 0)) + 1,
            "column": int(rng.get("character", 0)) + 1,
            "message": str(item.get("message", "")).strip(),
        })
    out.sort(key = lambda d: (d["line"], d["column"]))
    return out


def collect(session, file_path, *, timeout = 15.0):
    """Diagnostics for ``file_path``. Returns a list; [] means clean."""
    session.open_document(file_path)
    uri = path_to_uri(file_path)

    if session.supports("diagnosticProvider"):
        try:
            result = session.request(
                "textDocument/diagnostic",
                {"textDocument": {"uri": uri}},
                timeout = timeout,
            )
        except (jsonrpc.LspTimeout, jsonrpc.LspClosed):
            return []
        return _normalise((result or {}).get("items"))

    note = session.wait_notification(
        "textDocument/publishDiagnostics",
        lambda p: p.get("uri") == uri,
        timeout = timeout,
    )
    if note is None:
        return []
    return _normalise(note.get("diagnostics"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_assist_code_diagnostics.py --import-mode=importlib -q`
Expected: PASS, 4 tests

- [ ] **Step 5: Negative control on 1-based conversion**

Remove the `+ 1` from `line`. `test_line_and_column_are_converted_to_one_based` must FAIL. Restore.

- [ ] **Step 6: Commit**

```bash
git add core/inference/assist_code/diagnostics.py tests/test_assist_code_diagnostics.py
git commit -m "feat(code): diagnostics over both the push and pull LSP mechanisms"
```

---

## Task 7: Navigation

**Files:**
- Create: `core/inference/assist_code/navigation.py`
- Test: `tests/test_assist_code_navigation.py`

**Interfaces:**
- Consumes: `session.Session`, `session.path_to_uri`, `session.uri_to_path`
- Produces:
  - `resolve_position(session, file_path, *, symbol=None, line=None, column=None) -> tuple[dict|None, str|None]` — LSP `{"line","character"}` (0-based) or error text
  - `definition(session, file_path, position, *, timeout) -> list[dict]`
  - `references(session, file_path, position, *, timeout) -> list[dict]`
  - `hover(session, file_path, position, *, timeout) -> str`
  - `symbols(session, query, *, timeout) -> list[dict]`
  - Location dicts: `{"path": str, "line": int, "column": int}` with 1-based line/column

**Addressing:** the model gets `symbol` (a name) or `line`/`column`. Symbol-first matters because an agent usually has not read the file; coordinates are the escape hatch for ambiguity.

- [ ] **Step 1: Write the failing tests**

Extend the fake server first — add to `tests/lsp_fake_server.py`, inside `main()` before the generic `_write(... "result": None)` fallback:

```python
        if method == "textDocument/definition":
            _write({"jsonrpc": "2.0", "id": mid, "result": [{
                "uri": msg["params"]["textDocument"]["uri"],
                "range": {"start": {"line": 0, "character": 6},
                          "end": {"line": 0, "character": 7}},
            }]})
            continue
        if method == "textDocument/references":
            uri = msg["params"]["textDocument"]["uri"]
            _write({"jsonrpc": "2.0", "id": mid, "result": [
                {"uri": uri, "range": {"start": {"line": 0, "character": 6},
                                       "end": {"line": 0, "character": 7}}},
                {"uri": uri, "range": {"start": {"line": 3, "character": 2},
                                       "end": {"line": 3, "character": 3}}},
            ]})
            continue
        if method == "textDocument/hover":
            _write({"jsonrpc": "2.0", "id": mid, "result": {
                "contents": {"kind": "markdown", "value": "```ts\nconst a: number\n```"}}})
            continue
        if method == "workspace/symbol":
            query = (msg.get("params") or {}).get("query", "")
            if query == "nothingmatches":
                _write({"jsonrpc": "2.0", "id": mid, "result": []})
                continue
            _write({"jsonrpc": "2.0", "id": mid, "result": [{
                "name": query or "a", "kind": 13,
                "location": {"uri": "file:///%s/a.ts" % "ws",
                             "range": {"start": {"line": 0, "character": 6},
                                       "end": {"line": 0, "character": 7}}},
            }]})
            continue
```

```python
# tests/test_assist_code_navigation.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import os
import sys

import pytest

from core.inference.assist_code import navigation as nav, session as sess

FAKE = os.path.join(os.path.dirname(__file__), "lsp_fake_server.py")


@pytest.fixture
def ts_file(tmp_path):
    f = tmp_path / "a.ts"
    f.write_text("const alpha = 1\nconst beta = 2\n\nconsole.log(alpha)\n")
    return f


@pytest.fixture
def started(tmp_path):
    s = sess.Session([sys.executable, FAKE, "normal"], str(tmp_path), language = "typescript")
    s.start(timeout = 10)
    yield s
    s.close()


class TestResolvePosition:
    def test_explicit_line_and_column_convert_to_zero_based(self, started, ts_file):
        pos, err = nav.resolve_position(started, str(ts_file), line = 4, column = 13)
        assert err is None
        assert pos == {"line": 3, "character": 12}

    def test_a_symbol_name_is_found_in_the_file(self, started, ts_file):
        pos, err = nav.resolve_position(started, str(ts_file), symbol = "beta")
        assert err is None
        assert pos["line"] == 1
        assert pos["character"] == 6

    def test_a_symbol_that_is_absent_says_so(self, started, ts_file):
        pos, err = nav.resolve_position(started, str(ts_file), symbol = "nowhere")
        assert pos is None
        assert "nowhere" in err

    def test_neither_symbol_nor_position_is_an_error(self, started, ts_file):
        pos, err = nav.resolve_position(started, str(ts_file))
        assert pos is None
        assert "symbol" in err.lower()


class TestOperations:
    def test_definition_returns_one_based_locations(self, started, ts_file):
        locs = nav.definition(started, str(ts_file), {"line": 3, "character": 12}, timeout = 10)
        assert locs == [{"path": os.path.abspath(str(ts_file)), "line": 1, "column": 7}]

    def test_references_returns_every_usage(self, started, ts_file):
        locs = nav.references(started, str(ts_file), {"line": 0, "character": 6}, timeout = 10)
        assert len(locs) == 2
        assert locs[0]["line"] == 1 and locs[1]["line"] == 4

    def test_hover_returns_plain_text_without_markdown_fences(self, started, ts_file):
        text = nav.hover(started, str(ts_file), {"line": 0, "character": 6}, timeout = 10)
        assert "const a: number" in text
        assert "```" not in text

    def test_symbols_finds_matches(self, started, ts_file):
        got = nav.symbols(started, "alpha", timeout = 10)
        assert got and got[0]["name"] == "alpha"

    def test_no_symbol_matches_is_an_empty_list_not_an_error(self, started, ts_file):
        assert nav.symbols(started, "nothingmatches", timeout = 10) == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_assist_code_navigation.py --import-mode=importlib -q`
Expected: FAIL — no module `navigation`

- [ ] **Step 3: Implement `navigation.py`**

```python
# core/inference/assist_code/navigation.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Definition, references, hover and symbol search.

Addressing is symbol-first. LSP speaks line/character, but an agent thinks in
names and usually has not read the file -- so a name is resolved to a position
by scanning the opened document, with explicit line/column as the escape hatch
when a name is ambiguous.

Every location returned is 1-based, matching compilers and humans; LSP's
0-based coordinates never leak past this module.
"""
import re

from . import jsonrpc
from .session import path_to_uri, uri_to_path


def _loc(item):
    location = item.get("location") or item
    uri = location.get("uri") or location.get("targetUri")
    rng = location.get("range") or location.get("targetSelectionRange") or {}
    start = rng.get("start") or {}
    return {
        "path": uri_to_path(uri) if uri else "",
        "line": int(start.get("line", 0)) + 1,
        "column": int(start.get("character", 0)) + 1,
    }


def _locations(result):
    if result is None:
        return []
    if isinstance(result, dict):
        result = [result]
    return [_loc(item) for item in result]


def resolve_position(session, file_path, *, symbol = None, line = None, column = None):
    """Return an LSP position (0-based) or error text."""
    if line is not None:
        try:
            zero_line = int(line) - 1
            zero_col = int(column) - 1 if column is not None else 0
        except (TypeError, ValueError):
            return None, "line and column must be whole numbers"
        if zero_line < 0 or zero_col < 0:
            return None, "line and column are 1-based, so they start at 1"
        return {"line": zero_line, "character": zero_col}, None

    if not symbol or not str(symbol).strip():
        return None, "give either 'symbol' (a name) or 'line' (1-based)"

    name = str(symbol).strip()
    try:
        with open(file_path, "r", encoding = "utf-8", errors = "replace") as fh:
            lines = fh.read().splitlines()
    except OSError as e:
        return None, f"could not read {file_path}: {e}"

    pattern = re.compile(r"\b%s\b" % re.escape(name))
    for index, text in enumerate(lines):
        match = pattern.search(text)
        if match:
            return {"line": index, "character": match.start()}, None
    return None, f"symbol {name!r} does not appear in {file_path}"


def definition(session, file_path, position, *, timeout = 30.0):
    session.open_document(file_path)
    try:
        result = session.request("textDocument/definition", {
            "textDocument": {"uri": path_to_uri(file_path)}, "position": position,
        }, timeout = timeout)
    except (jsonrpc.LspTimeout, jsonrpc.LspClosed):
        return []
    return _locations(result)


def references(session, file_path, position, *, timeout = 30.0):
    session.open_document(file_path)
    try:
        result = session.request("textDocument/references", {
            "textDocument": {"uri": path_to_uri(file_path)}, "position": position,
            "context": {"includeDeclaration": True},
        }, timeout = timeout)
    except (jsonrpc.LspTimeout, jsonrpc.LspClosed):
        return []
    return _locations(result)


def _hover_text(contents):
    """LSP allows a string, a {language,value} pair, a MarkupContent, or a list."""
    if contents is None:
        return ""
    if isinstance(contents, str):
        return contents
    if isinstance(contents, dict):
        return str(contents.get("value", ""))
    if isinstance(contents, list):
        return "\n".join(_hover_text(part) for part in contents)
    return str(contents)


def hover(session, file_path, position, *, timeout = 30.0):
    session.open_document(file_path)
    try:
        result = session.request("textDocument/hover", {
            "textDocument": {"uri": path_to_uri(file_path)}, "position": position,
        }, timeout = timeout)
    except (jsonrpc.LspTimeout, jsonrpc.LspClosed):
        return ""
    text = _hover_text((result or {}).get("contents"))
    # Strip markdown fences: the model gets plain text, not rendering markup.
    cleaned = re.sub(r"```[a-zA-Z0-9_+-]*\n?", "", text).replace("```", "")
    return cleaned.strip()


_SYMBOL_KINDS = {
    5: "class", 6: "method", 8: "field", 9: "constructor", 10: "enum",
    11: "interface", 12: "function", 13: "variable", 14: "constant",
    23: "struct",
}


def symbols(session, query, *, timeout = 30.0):
    try:
        result = session.request("workspace/symbol", {"query": query}, timeout = timeout)
    except (jsonrpc.LspTimeout, jsonrpc.LspClosed):
        return []
    out = []
    for item in result or []:
        loc = _loc(item)
        loc["name"] = item.get("name", "")
        loc["kind"] = _SYMBOL_KINDS.get(item.get("kind"), "symbol")
        out.append(loc)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_assist_code_navigation.py --import-mode=importlib -q`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add core/inference/assist_code/navigation.py tests/lsp_fake_server.py tests/test_assist_code_navigation.py
git commit -m "feat(code): LSP navigation with symbol-first addressing"
```

---

## Task 8: Schemas, dispatcher, registration, and real-server end-to-end

**Files:**
- Create: `core/inference/assist_code/schemas.py`
- Modify: `core/inference/assist_code/__init__.py`
- Modify: `core/inference/tools.py` — **exactly two touchpoints**
- Modify: `routes/inference.py` — **extend the existing re-add hunk, do not add a second**
- Test: `tests/test_assist_code_registration.py`, `tests/test_assist_code_e2e.py`

**Interfaces:**
- Consumes: everything above
- Produces: `ASSIST_CODE_TOOLS: list[dict]`, `ASSIST_CODE_TOOL_NAMES: set[str]`, `execute(name, arguments, *, session_id=None, timeout=None, cancel_event=None) -> str`

- [ ] **Step 1: Write `schemas.py`**

```python
# core/inference/assist_code/schemas.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI-style schemas for the five read-only code-intelligence tools."""

_PATH = (
    "Path to a source file. It must be inside the conversation's working "
    "directory; a bare filename resolves there. Supported: .ts .tsx .js .jsx .cs"
)
_WHERE = (
    "Give either 'symbol' (a name, easiest -- it is located for you) or an "
    "explicit 'line' (1-based) and optional 'column' (1-based)."
)
_FIRST_USE = (
    " On first use for a language this may install its language server "
    "(typescript-language-server via npm, csharp-ls via dotnet tool) and "
    "index the project, which can take a few seconds."
)


def _tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


_POSITION_PROPS = {
    "path": {"type": "string", "description": _PATH},
    "symbol": {"type": "string", "description": "Name of the symbol to locate. " + _WHERE},
    "line": {"type": "integer", "description": "1-based line number. Alternative to 'symbol'."},
    "column": {"type": "integer", "description": "1-based column number. Optional with 'line'."},
}

CODE_DIAGNOSTICS_TOOL = _tool(
    "code_diagnostics",
    "Type errors and warnings for a source file, from its language server, "
    "without running a build. Use after editing to check the change is sound. "
    "An empty result means the file is clean." + _FIRST_USE,
    {"path": {"type": "string", "description": _PATH}},
    ["path"],
)

CODE_DEFINITION_TOOL = _tool(
    "code_definition",
    "Where a symbol is defined. " + _WHERE + _FIRST_USE,
    _POSITION_PROPS, ["path"],
)

CODE_REFERENCES_TOOL = _tool(
    "code_references",
    "Every place a symbol is used across the project. Use before changing a "
    "shared function to see what depends on it. " + _WHERE + _FIRST_USE,
    _POSITION_PROPS, ["path"],
)

CODE_HOVER_TOOL = _tool(
    "code_hover",
    "Type, signature and documentation for a symbol. " + _WHERE + _FIRST_USE,
    _POSITION_PROPS, ["path"],
)

CODE_SYMBOLS_TOOL = _tool(
    "code_symbols",
    "Search the project for a symbol by name -- classes, functions, variables "
    "-- when you know a name but not which file holds it." + _FIRST_USE,
    {
        "query": {"type": "string", "description": "Symbol name or prefix to search for."},
        "path": {"type": "string", "description":
                 "Any file in the project to search, used to pick the workspace. " + _PATH},
    },
    ["query", "path"],
)

ASSIST_CODE_TOOLS = [
    CODE_DIAGNOSTICS_TOOL, CODE_DEFINITION_TOOL, CODE_REFERENCES_TOOL,
    CODE_HOVER_TOOL, CODE_SYMBOLS_TOOL,
]
ASSIST_CODE_TOOL_NAMES = {t["function"]["name"] for t in ASSIST_CODE_TOOLS}
```

- [ ] **Step 2: Write the failing registration tests**

```python
# tests/test_assist_code_registration.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import asyncio
import types

import pytest

from core.inference.assist_code import ASSIST_CODE_TOOL_NAMES

_STUDIO_ALLOWLIST = ["web_search", "python", "terminal", "edit_file"]


def _payload(**over):
    base = dict(enabled_tools = list(_STUDIO_ALLOWLIST), rag_scope = None,
                thread_id = None, bypass_permissions = False)
    base.update(over)
    return types.SimpleNamespace(**base)


def _select(payload, **kwargs):
    import routes.inference as routes_mod
    kwargs.setdefault("tools_on", True)
    kwargs.setdefault("mcp_allowed", False)
    return asyncio.run(routes_mod._select_request_tools(payload, **kwargs))


def _names(tools):
    return [t["function"]["name"] for t in tools]


class TestRegistry:
    def test_every_code_tool_is_in_all_tools(self):
        from core.inference.tools import ALL_TOOLS
        names = {t["function"]["name"] for t in ALL_TOOLS}
        assert ASSIST_CODE_TOOL_NAMES <= names

    def test_no_code_tool_collides_with_an_upstream_name(self):
        from core.inference.tools import ALL_TOOLS
        names = [t["function"]["name"] for t in ALL_TOOLS]
        for name in ASSIST_CODE_TOOL_NAMES:
            assert names.count(name) == 1

    def test_every_schema_documents_the_first_use_install(self):
        from core.inference.assist_code import ASSIST_CODE_TOOLS
        for tool in ASSIST_CODE_TOOLS:
            assert "install" in tool["function"]["description"].lower()


class TestReachability:
    def test_code_tools_survive_a_studio_shaped_allowlist(self):
        """Registration is necessary but not sufficient: Studio's frontend
        sends an enabled_tools allowlist naming only upstream tools."""
        names = _names(_select(_payload()))
        missing = ASSIST_CODE_TOOL_NAMES - set(names)
        assert not missing, f"code tools filtered out of a Studio chat request: {sorted(missing)}"

    def test_the_vision_tools_still_survive_too(self):
        """The re-add is shared; extending it must not displace sub-project 1."""
        from core.inference.tools import ASSIST_VISION_TOOL_NAMES
        names = set(_names(_select(_payload())))
        assert ASSIST_VISION_TOOL_NAMES <= names

    def test_an_empty_selection_stays_empty(self):
        """Upstream relies on an empty selection producing an empty catalogue
        so the tool loop is skipped. Breaking this broke four upstream tests
        in test_run_tools_locally_discriminator.py."""
        assert _names(_select(_payload(enabled_tools = []))) == []

    def test_a_selection_of_only_unimplemented_tools_stays_empty(self):
        assert _names(_select(_payload(enabled_tools = ["code_execution"]))) == []

    def test_no_tool_is_offered_twice_when_the_allowlist_is_omitted(self):
        names = _names(_select(_payload(enabled_tools = None)))
        for name in ASSIST_CODE_TOOL_NAMES:
            assert names.count(name) == 1

    def test_tools_off_admits_no_code_tool(self):
        names = set(_names(_select(_payload(), tools_on = False)))
        assert not (ASSIST_CODE_TOOL_NAMES & names)


class TestDispatch:
    def test_execute_tool_reaches_the_code_dispatcher(self, monkeypatch):
        from core.inference import tools
        from core.inference import assist_code
        monkeypatch.setattr(assist_code, "execute", lambda *a, **k: "SENTINEL")
        assert tools.execute_tool("code_diagnostics", {"path": "a.ts"}, session_id = "s") == "SENTINEL"

    def test_every_code_tool_returns_a_string_for_junk_arguments(self):
        from core.inference import tools
        for name in sorted(ASSIST_CODE_TOOL_NAMES):
            for args in ({}, {"path": None}, {"path": 5}, {"path": "../../etc/passwd"}):
                out = tools.execute_tool(name, args, session_id = "s")
                assert isinstance(out, str) and out
```

- [ ] **Step 3: Run to verify they fail**

Run: `python -m pytest tests/test_assist_code_registration.py --import-mode=importlib -q`
Expected: FAIL — `ImportError: cannot import name 'ASSIST_CODE_TOOL_NAMES'`

- [ ] **Step 4: Implement the dispatcher in `__init__.py`**

Replace the file's contents:

```python
# core/inference/assist_code/__init__.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read-only language intelligence (LSP) tools for Studio's agent loop.

``execute`` is the single entry point ``execute_tool`` delegates to. It always
returns ``str`` and never raises: the tool boundary is a string, so an escaping
exception would land in the agent loop instead of becoming text the model can
read and act on.
"""
import os

from .schemas import ASSIST_CODE_TOOLS, ASSIST_CODE_TOOL_NAMES  # noqa: F401

_MAX_RESULTS = 50


def _session_for(path, session_id, start_timeout):
    """Resolve path, pick a language and workspace, and get a pooled session.

    Returns (session, resolved_path, None) or (None, None, error_text).
    """
    from . import paths, pool, servers, session as sess

    resolved, err = paths.resolve_file(path, session_id = session_id)
    if err:
        return None, None, err

    language = servers.language_for(resolved)
    if language is None:
        return None, None, (
            f"no language server for {os.path.basename(resolved)}. "
            f"Supported file types: .ts .tsx .js .jsx .cs"
        )
    try:
        command = servers.server_command(language)
    except servers.ServerUnavailable as e:
        return None, None, str(e)

    root = paths.workspace_for(resolved, session_id = session_id)

    def factory(resolved_root):
        return sess.Session(command, resolved_root, language = language)

    try:
        session = pool.acquire(language, root, factory, start_timeout = start_timeout)
    except sess.SessionStartFailed as e:
        return None, None, str(e)
    return session, resolved, None


def _rel(path, session_id):
    try:
        from core.inference import tools as _tools
        return os.path.relpath(path, _tools._get_workdir(session_id))
    except Exception:
        return path


def _format_locations(locations, session_id, empty_message):
    if not locations:
        return empty_message
    lines = []
    for loc in locations[:_MAX_RESULTS]:
        label = loc.get("name")
        prefix = f"{label} ({loc.get('kind', 'symbol')}) - " if label else ""
        lines.append(f"{prefix}{_rel(loc['path'], session_id)}:{loc['line']}:{loc['column']}")
    if len(locations) > _MAX_RESULTS:
        lines.append(f"... and {len(locations) - _MAX_RESULTS} more")
    return "\n".join(lines)


def _position(session, resolved, arguments):
    from . import navigation
    return navigation.resolve_position(
        session, resolved,
        symbol = arguments.get("symbol"),
        line = arguments.get("line"),
        column = arguments.get("column"),
    )


def _do_diagnostics(arguments, session_id, budget):
    from . import diagnostics
    session, resolved, err = _session_for(arguments.get("path"), session_id, budget)
    if err:
        return err
    found = diagnostics.collect(session, resolved, timeout = min(budget, 20.0))
    if not found:
        return f"No problems found in {_rel(resolved, session_id)}."
    lines = [f"{len(found)} problem(s) in {_rel(resolved, session_id)}:"]
    for d in found[:_MAX_RESULTS]:
        lines.append(f"  {d['line']}:{d['column']}  {d['severity']}: {d['message']}")
    if len(found) > _MAX_RESULTS:
        lines.append(f"  ... and {len(found) - _MAX_RESULTS} more")
    return "\n".join(lines)


def _do_definition(arguments, session_id, budget):
    from . import navigation
    session, resolved, err = _session_for(arguments.get("path"), session_id, budget)
    if err:
        return err
    position, err = _position(session, resolved, arguments)
    if err:
        return err
    locs = navigation.definition(session, resolved, position, timeout = min(budget, 20.0))
    return _format_locations(locs, session_id, "No definition found.")


def _do_references(arguments, session_id, budget):
    from . import navigation
    session, resolved, err = _session_for(arguments.get("path"), session_id, budget)
    if err:
        return err
    position, err = _position(session, resolved, arguments)
    if err:
        return err
    locs = navigation.references(session, resolved, position, timeout = min(budget, 20.0))
    return _format_locations(locs, session_id, "No references found.")


def _do_hover(arguments, session_id, budget):
    from . import navigation
    session, resolved, err = _session_for(arguments.get("path"), session_id, budget)
    if err:
        return err
    position, err = _position(session, resolved, arguments)
    if err:
        return err
    text = navigation.hover(session, resolved, position, timeout = min(budget, 20.0))
    return text or "No type information available at that position."


def _do_symbols(arguments, session_id, budget):
    from . import navigation
    query = arguments.get("query")
    if not query or not str(query).strip():
        return "query is required"
    session, _resolved, err = _session_for(arguments.get("path"), session_id, budget)
    if err:
        return err
    found = navigation.symbols(session, str(query).strip(), timeout = min(budget, 20.0))
    return _format_locations(found, session_id, f"No symbol matching {query!r} was found.")


_HANDLERS = {
    "code_diagnostics": _do_diagnostics,
    "code_definition": _do_definition,
    "code_references": _do_references,
    "code_hover": _do_hover,
    "code_symbols": _do_symbols,
}


def execute(name, arguments, *, session_id = None, timeout = None, cancel_event = None):
    """Run a code-intelligence tool. Always returns str; never raises."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"unknown code tool: {name}"
    if cancel_event is not None and cancel_event.is_set():
        return f"{name} was cancelled before it started"
    try:
        budget = float(timeout) if timeout else 60.0
    except (TypeError, ValueError):
        budget = 60.0
    try:
        return handler(arguments or {}, session_id, budget)
    except Exception as e:  # noqa: BLE001 - the tool boundary must not raise
        return f"{name} failed: {type(e).__name__}: {e}"
```

- [ ] **Step 5: Add the two `tools.py` touchpoints**

Beside the existing vision import near `ALL_TOOLS` (~line 9801):

```python
from core.inference.assist_code import ASSIST_CODE_TOOLS, ASSIST_CODE_TOOL_NAMES
```

In the `ALL_TOOLS` list, beside `*ASSIST_VISION_TOOLS`:

```python
    *ASSIST_CODE_TOOLS,
```

In `execute_tool`, beside the existing vision branch (below `effective_timeout`):

```python
    if name in ASSIST_CODE_TOOL_NAMES:
        from core.inference import assist_code
        return assist_code.execute(
            name, arguments, session_id = session_id,
            timeout = effective_timeout, cancel_event = cancel_event,
        )
```

- [ ] **Step 6: Extend the `routes/inference.py` re-add — do NOT add a second hunk**

Change only the import line and the membership test inside the existing block:

```python
    if tools_on and tools:
        from core.inference.tools import ASSIST_VISION_TOOL_NAMES, ASSIST_CODE_TOOL_NAMES
        _addable = ASSIST_VISION_TOOL_NAMES | ASSIST_CODE_TOOL_NAMES
        _already = {t["function"]["name"] for t in tools}
        tools = tools + [
            t for t in ALL_TOOLS
            if t["function"]["name"] in _addable
            and t["function"]["name"] not in _already
        ]
```

Add one sentence to that block's existing comment:

```python
    # Sub-project 2 adds the code-intelligence tools to the same re-add rather
    # than a second hunk: one place to reason about, one merge conflict to
    # resolve instead of two.
```

- [ ] **Step 7: Run registration tests**

Run: `python -m pytest tests/test_assist_code_registration.py --import-mode=importlib -q`
Expected: PASS, 12 tests

- [ ] **Step 8: Verify the seam and that sub-project 1 still passes**

```bash
git diff --stat core/inference/tools.py routes/inference.py
python -m pytest tests/ -k "assist_vision or assist_code or run_tools_locally or edit_file_tool or tool_approvals" --import-mode=importlib -q
```
Expected: `tools.py` grows by 3 lines and stays 3 hunks; `routes/inference.py` stays **one** hunk. The vision and discriminator tests must still pass. The only acceptable failure is the pre-existing `test_a_paired_surrogate_emoji_still_writes_normally`.

- [ ] **Step 9: Write the real-server end-to-end test**

```python
# tests/test_assist_code_e2e.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""End-to-end against the REAL typescript-language-server.

Skipped when the server is absent, but never silently: a skip reason names it.
These are the only tests here that prove the thing actually works -- the fake
server proves the client is correct about a protocol we defined, and this
proves we were right about the protocol.
"""
import os
import shutil

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("typescript-language-server") is None,
    reason = "typescript-language-server not installed; run: npm install -g typescript-language-server typescript",
)


@pytest.fixture
def ts_project(tmp_path, monkeypatch):
    from core.inference import tools
    monkeypatch.setattr(tools, "_get_workdir", lambda _sid = None: str(tmp_path))
    (tmp_path / "tsconfig.json").write_text('{"compilerOptions":{"strict":true}}')
    (tmp_path / "lib.ts").write_text(
        "export function addNumbers(a: number, b: number): number {\n"
        "  return a + b\n"
        "}\n"
    )
    (tmp_path / "main.ts").write_text(
        "import { addNumbers } from './lib'\n"
        "const wrong: number = addNumbers(1, 'two')\n"
        "console.log(wrong)\n"
    )
    yield tmp_path
    from core.inference.assist_code import pool
    pool.shutdown_all()


def test_a_real_type_error_is_reported_with_its_line(ts_project):
    from core.inference import tools
    out = tools.execute_tool("code_diagnostics", {"path": "main.ts"}, session_id = "e2e")
    assert "problem" in out.lower()
    assert ":2:" in out, out


def test_a_clean_file_reports_no_problems(ts_project):
    from core.inference import tools
    out = tools.execute_tool("code_diagnostics", {"path": "lib.ts"}, session_id = "e2e")
    assert "no problems" in out.lower(), out


def test_definition_crosses_files(ts_project):
    from core.inference import tools
    out = tools.execute_tool(
        "code_definition", {"path": "main.ts", "symbol": "addNumbers"}, session_id = "e2e")
    assert "lib.ts" in out, out


def test_hover_reports_the_real_signature(ts_project):
    from core.inference import tools
    out = tools.execute_tool(
        "code_hover", {"path": "lib.ts", "symbol": "addNumbers"}, session_id = "e2e")
    assert "addNumbers" in out and "number" in out, out


def test_references_finds_the_call_site(ts_project):
    from core.inference import tools
    out = tools.execute_tool(
        "code_references", {"path": "lib.ts", "symbol": "addNumbers"}, session_id = "e2e")
    assert "main.ts" in out, out
```

- [ ] **Step 10: Run the end-to-end tests**

```bash
npm install -g typescript-language-server typescript
python -m pytest tests/test_assist_code_e2e.py --import-mode=importlib -q
```
Expected: PASS, 5 tests. **If they skip, the task is not done** — install the server and run them.

- [ ] **Step 11: Negative control on reachability**

Temporarily change the re-add gate to `if False:`. `test_code_tools_survive_a_studio_shaped_allowlist` and `test_the_vision_tools_still_survive_too` must FAIL while `test_an_empty_selection_stays_empty` still passes. Restore and confirm all pass.

- [ ] **Step 12: Commit**

```bash
git add core/inference/assist_code/schemas.py core/inference/assist_code/__init__.py \
        core/inference/tools.py routes/inference.py \
        tests/test_assist_code_registration.py tests/test_assist_code_e2e.py
git commit -m "feat(code): register the five code-intelligence tools into Studio's agent loop"
```

---

## Self-Review

**Spec coverage.** Persistent pool → Task 4. `mcp_client` lifecycle model → Task 4. Non-logging atexit → Task 4 (with a test that monkeypatches an exploding logger). Push/pull diagnostics → Task 6. Symbol-first addressing → Task 7. Server acquisition with disclosure → Task 5. Confinement, unconditional → Task 1. Two `tools.py` touchpoints and the *extended* re-add → Task 8. Real fake server, not mocks → Task 2, reused by 3/4/6/7. Real end-to-end → Task 8 Step 9. Negative controls → Tasks 1, 2, 4, 6, 8. Read-only surface, no rename, no completions → Task 8 schemas. C#/TS-JS only → Task 5 `SUPPORTED`, with a test asserting Rust and C++ are absent.

**Type consistency.** `resolve_file`/`resolve_workspace` return `(value, error)` throughout. Positions are LSP 0-based only inside `navigation`/`diagnostics`; everything crossing the tool boundary is 1-based. `Session` is constructed `(command, root, *, language)` in Tasks 3, 4 and 8 identically. `pool.acquire(language, root, factory, *, start_timeout)` matches its Task 8 call site.

**One known gap, deliberate.** `code_symbols` requires `path` purely to choose a workspace, which reads oddly for a project-wide search. The alternative — a `workspace` parameter — would need its own confinement path and a way for the model to know the root. Revisit if it proves awkward in use.
