# Multi-Persona System: Memory Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A chat session bound to a Crew persona only ever sees, searches, edits, or deletes memories created by that same persona, plus a shared "no persona" pool — never another persona's memories. A session with no persona bound keeps today's exact behavior.

**Architecture:** Tag each memory entry with an optional `persona_id`, resolved via the already-shared `resolve_crew_binding(db, session_id, owner)` (the same mechanism sub-project 2 used for TTS voice), and extend the memory system's existing owner-filtering with one more skip-check implementing "own persona + shared pool, never another persona's."

**IMPORTANT CORRECTION FROM THE DESIGN SPEC** (found during this plan's own live-code research —
read this before starting, it changes which files matter): the spec
(`docs/superpowers/specs/2026-08-06-crew-memory-isolation-design.md`) described the isolation
mechanism as living inside `NativeMemoryProvider.recall()`/`delete()` (`src/memory_provider.py`).
Tracing the actual call graph found that **neither of the two real live consumers of memory ever
calls that class**:
- `do_manage_memory` (`src/ai_interaction.py:296`, the agent tool) calls `_memory_manager` (a
  `MemoryManager` instance) **directly** — `_memory_manager.load(owner=owner)` for list/search,
  `_memory_manager.add_entry(...)` for add, and hand-written inline
  `if owner and m.get("owner") != owner:` ownership checks for edit/delete. It never touches
  `NativeMemoryProvider`.
- The auto-injection context-priming pass (`src/chat_processor.py`'s `build_context_preface`)
  also calls `self.memory_manager.load(owner=owner)` **directly**.
- `NativeMemoryProvider`/`MemoryProviderRegistry`/`MemoryService` are constructed in
  `src/app_initializer.py` (`memory_provider_registry = MemoryProviderRegistry([NativeMemoryProvider(...)])`)
  and stored under a services dict key (`"memory_provider_registry"`) — but nothing anywhere in
  `app.py`, any route file, or the MCP memory server ever reads that key. Confirmed via grep: zero
  consumers. **This is a built-but-unwired extensibility layer with no live production traffic.**

The spec's *requirements* (hard isolation, shared unscoped pool, fail-open, reuse
`resolve_crew_binding`) are completely unaffected by this correction — only *which files*
implement the owner-filtering pattern that gets extended. This plan targets the real call graph:
`src/memory.py` (`MemoryManager`), `src/ai_interaction.py` (`do_manage_memory`),
`src/chat_processor.py` (`build_context_preface`), and `routes/chat_helpers.py`
(`extract_preset`/`build_chat_context`, which already resolves a session's persona binding for
an unrelated purpose — see Task 4). **`src/memory_provider.py`, `services/memory/service.py`,
and `src/app_initializer.py` are NOT touched by this plan** — patching unreachable code would be
speculative work with nothing to verify it against.

**Tech Stack:** Python 3.14 (FastAPI backend), SQLAlchemy (for persona binding lookups only —
memory itself is a flat JSON file, not a DB table), pytest with `--import-mode=importlib`.

## Global Constraints

- Hard isolation: a persona-bound session sees `{its own persona's memories} ∪ {the shared/unscoped pool}`, never another persona's. No "search everyone's memories" override.
- Every memory that exists before this feature ships, and every memory created in a session with no persona bound, lands in the shared pool — visible to every persona and to unbound sessions alike.
- Resolve `persona_id` via `resolve_crew_binding(db, session_id, owner)` (`src/crew_helpers.py`) — reuse as-is, do not reimplement persona/owner scoping.
- Fail-open: no session, no owner, no persona binding, a dangling binding, or any lookup error must silently fall back to today's exact owner-only behavior — never an error surfaced to a memory caller.
- No schema migration — memory entries are flat JSON dicts, not fixed-column DB rows. Add one optional key, set only when present (mirroring exactly how `owner` is already handled).
- Editing or deleting a memory that belongs to a different persona is treated as "not found" — identical to how a different owner's memory is already treated today. No new error type.
- Commit directly to `dev` (this project's established convention — no feature branch). Stage specific files, never `git add -A`. Do not stage `installer/Output/Assist-Setup.exe`. Commits end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. pytest needs `--import-mode=importlib`.
- Tests use real `MemoryManager`/DB rows (not mocks) — this feature's core test surface is explicit cross-persona leak checks (persona A creates a memory, persona B must not see it via list/search/edit/delete; a shared-pool memory must be visible to both), not just happy-path coverage.

---

### Task 1: `MemoryManager` gains `persona_id` filtering

**Files:**
- Modify: `src/memory.py` (`add_entry`, `load`)
- Test: `tests/test_memory_persona_isolation.py` (new)

**Interfaces:**
- Produces: `MemoryManager.add_entry(text, source="user", category="fact", owner=None, persona_id=None) -> Dict` — sets `entry["persona_id"]` only when `persona_id` is truthy, mirroring the existing `owner` handling exactly. `MemoryManager.load(owner=None, persona_id=None) -> List[Dict]` — when `persona_id` is given, returns entries where `entry.get("persona_id") in (None, persona_id)` (own persona's entries + the shared/unscoped pool), applied on top of the existing owner filter.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_persona_isolation.py`:

```python
"""Persona-scoped memory isolation at the MemoryManager layer. Hard
isolation: a persona sees its own tagged entries plus the shared
(untagged) pool, never another persona's tagged entries. Mirrors the
sub-project-2 fail-open precedent: no persona_id given -> today's exact
owner-only behavior, unchanged."""
from src.memory import MemoryManager


def test_add_entry_sets_persona_id_when_given(tmp_path):
    mgr = MemoryManager(str(tmp_path))
    entry = mgr.add_entry("likes dark mode", owner="alice", persona_id="persona-a")
    assert entry["persona_id"] == "persona-a"


def test_add_entry_omits_persona_id_when_not_given(tmp_path):
    mgr = MemoryManager(str(tmp_path))
    entry = mgr.add_entry("likes dark mode", owner="alice")
    assert "persona_id" not in entry


def test_load_without_persona_id_returns_everything_for_the_owner(tmp_path):
    """Fail-open / backward compatibility: omitting persona_id must behave
    exactly like today -- owner-only filtering, no persona narrowing."""
    mgr = MemoryManager(str(tmp_path))
    e1 = mgr.add_entry("fact one", owner="alice", persona_id="persona-a")
    e2 = mgr.add_entry("fact two", owner="alice", persona_id="persona-b")
    e3 = mgr.add_entry("fact three", owner="alice")
    mgr.save([e1, e2, e3])

    loaded = mgr.load(owner="alice")
    assert {m["text"] for m in loaded} == {"fact one", "fact two", "fact three"}


def test_load_with_persona_id_hides_other_personas_entries(tmp_path):
    mgr = MemoryManager(str(tmp_path))
    e1 = mgr.add_entry("persona A's fact", owner="alice", persona_id="persona-a")
    e2 = mgr.add_entry("persona B's fact", owner="alice", persona_id="persona-b")
    e3 = mgr.add_entry("shared fact", owner="alice")
    mgr.save([e1, e2, e3])

    loaded = mgr.load(owner="alice", persona_id="persona-a")
    texts = {m["text"] for m in loaded}
    assert texts == {"persona A's fact", "shared fact"}
    assert "persona B's fact" not in texts


def test_load_with_persona_id_still_respects_owner(tmp_path):
    """Persona filtering is additive to owner filtering, not a replacement
    for it -- a different owner's same-persona-id entry must still be
    invisible (persona ids are only meaningful within one owner's CrewMember
    table, but this guards against any future ID-collision assumption)."""
    mgr = MemoryManager(str(tmp_path))
    e1 = mgr.add_entry("alice's fact", owner="alice", persona_id="persona-a")
    e2 = mgr.add_entry("bob's fact", owner="bob", persona_id="persona-a")
    mgr.save([e1, e2])

    loaded = mgr.load(owner="alice", persona_id="persona-a")
    assert {m["text"] for m in loaded} == {"alice's fact"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_memory_persona_isolation.py -v --import-mode=importlib`
Expected: FAIL — `add_entry()`/`load()` don't accept `persona_id` yet (`TypeError: unexpected keyword argument`).

- [ ] **Step 3: Implement `add_entry`'s `persona_id` param**

`src/memory.py`'s `add_entry` currently reads (lines 215-230):

```python
    def add_entry(self, text: str, source: str = "user", category: str = "fact", owner: str = None) -> Dict:
        """Add a new memory entry."""
        if not text.strip():
            raise ValueError("Memory text cannot be empty")

        entry = {
            "id": str(uuid.uuid4()),
            "text": text.strip(),
            "timestamp": int(time.time()),
            "source": source,
            "category": category,
            "uses": 0,
        }
        if owner:
            entry["owner"] = owner
        return entry
```

Change to:

```python
    def add_entry(self, text: str, source: str = "user", category: str = "fact", owner: str = None, persona_id: str = None) -> Dict:
        """Add a new memory entry."""
        if not text.strip():
            raise ValueError("Memory text cannot be empty")

        entry = {
            "id": str(uuid.uuid4()),
            "text": text.strip(),
            "timestamp": int(time.time()),
            "source": source,
            "category": category,
            "uses": 0,
        }
        if owner:
            entry["owner"] = owner
        if persona_id:
            entry["persona_id"] = persona_id
        return entry
```

- [ ] **Step 4: Implement `load`'s `persona_id` param**

`src/memory.py`'s `load` currently reads (lines 129-134):

```python
    def load(self, owner: str = None) -> List[Dict]:
        """Load memory entries, optionally filtered by owner."""
        entries = self.load_all()
        if owner is None:
            return entries
        return [e for e in entries if e.get("owner") == owner]
```

Change to:

```python
    def load(self, owner: str = None, persona_id: str = None) -> List[Dict]:
        """Load memory entries, optionally filtered by owner and/or persona.
        persona_id filtering is additive: an entry is visible when it has no
        persona_id (the shared pool) OR its persona_id matches exactly --
        never a different persona's entries. Hard isolation, fail-open when
        persona_id is not given (today's exact owner-only behavior)."""
        entries = self.load_all()
        if owner is not None:
            entries = [e for e in entries if e.get("owner") == owner]
        if persona_id is not None:
            entries = [e for e in entries if e.get("persona_id") in (None, persona_id)]
        return entries
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_memory_persona_isolation.py -v --import-mode=importlib`
Expected: PASS (all 5 tests).

Then run the existing memory test files to confirm no regression:

Run: `pytest tests/test_memory_imports.py tests/test_memory_fallback_dislike.py tests/test_memory_bullet_extraction.py tests/test_memory_extraction_parse.py tests/test_memory_extractor_rows.py tests/test_memory_extractor_vector_degraded.py tests/test_memory_recall_nondict_rows.py --import-mode=importlib -v`
Expected: PASS (full files — confirms the new optional params don't break any existing caller).

- [ ] **Step 6: Commit**

```bash
git add src/memory.py tests/test_memory_persona_isolation.py
git commit -m "feat(memory): MemoryManager gains persona_id isolation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `do_manage_memory` resolves and applies persona isolation

**Files:**
- Modify: `src/ai_interaction.py` (`do_manage_memory`)
- Test: `tests/test_manage_memory_persona_isolation.py` (new)

**Interfaces:**
- Consumes: `MemoryManager.add_entry(..., persona_id=...)` / `load(..., persona_id=...)` (Task 1). `resolve_crew_binding(db, session_id, owner)` (`src/crew_helpers.py`, unchanged). `SessionLocal` (`core.database`, unchanged).
- Produces: `do_manage_memory(content, session_id=None, owner=None)`'s five actions (list/add/edit/delete/search) all respect persona isolation — a persona-bound session's `session_id` restricts every action to that persona's memories plus the shared pool.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_manage_memory_persona_isolation.py`:

```python
"""Cross-persona isolation for the manage_memory agent tool -- the tool's
whole surface (list/add/edit/delete/search) must respect persona
boundaries once a session is bound to one. Uses a real MemoryManager
(tmp_path-backed) and real CrewMember/Session ORM rows (the app's default
SessionLocal, matching how tests/test_session_crew_binding.py already
exercises resolve_crew_binding against real rows)."""
import uuid

import pytest

import src.ai_interaction as ai
from src.memory import MemoryManager


@pytest.fixture
def memory_manager(tmp_path):
    mgr = MemoryManager(str(tmp_path))
    ai.set_memory_manager(mgr)
    yield mgr
    ai.set_memory_manager(None)


def _make_persona_session(owner="alice", name="Nav"):
    from core.database import SessionLocal, CrewMember, Session as DbSession
    db = SessionLocal()
    try:
        crew_id = str(uuid.uuid4())
        db.add(CrewMember(id=crew_id, owner=owner, name=name))
        sess_id = str(uuid.uuid4())
        db.add(DbSession(id=sess_id, name="s", endpoint_url="http://x", model="m",
                         owner=owner, crew_member_id=crew_id))
        db.commit()
        return sess_id, crew_id
    finally:
        db.close()


def _make_unbound_session(owner="alice"):
    from core.database import SessionLocal, Session as DbSession
    db = SessionLocal()
    try:
        sess_id = str(uuid.uuid4())
        db.add(DbSession(id=sess_id, name="s", endpoint_url="http://x", model="m", owner=owner))
        db.commit()
        return sess_id
    finally:
        db.close()


@pytest.mark.asyncio
async def test_add_then_list_is_scoped_to_the_creating_persona(memory_manager):
    sess_a, _ = _make_persona_session(name="Persona A")
    result = await ai.do_manage_memory("add\nA's private fact", session_id=sess_a, owner="alice")
    assert "memory_id" in result

    listed = await ai.do_manage_memory("list", session_id=sess_a, owner="alice")
    assert "A's private fact" in listed["results"]


@pytest.mark.asyncio
async def test_persona_b_cannot_list_persona_as_memory(memory_manager):
    sess_a, _ = _make_persona_session(owner="alice", name="Persona A")
    sess_b, _ = _make_persona_session(owner="alice", name="Persona B")
    await ai.do_manage_memory("add\nA's private fact", session_id=sess_a, owner="alice")

    listed = await ai.do_manage_memory("list", session_id=sess_b, owner="alice")
    assert "A's private fact" not in listed["results"]


@pytest.mark.asyncio
async def test_persona_b_cannot_search_persona_as_memory(memory_manager):
    sess_a, _ = _make_persona_session(owner="alice", name="Persona A")
    sess_b, _ = _make_persona_session(owner="alice", name="Persona B")
    await ai.do_manage_memory("add\nthe secret launch code is 42", session_id=sess_a, owner="alice")

    found = await ai.do_manage_memory("search\nlaunch code", session_id=sess_b, owner="alice")
    assert "42" not in found["results"]


@pytest.mark.asyncio
async def test_persona_b_cannot_edit_persona_as_memory(memory_manager):
    sess_a, _ = _make_persona_session(owner="alice", name="Persona A")
    sess_b, _ = _make_persona_session(owner="alice", name="Persona B")
    added = await ai.do_manage_memory("add\noriginal text", session_id=sess_a, owner="alice")
    memory_id = added["memory_id"]

    result = await ai.do_manage_memory(f"edit\n{memory_id}\nhijacked text", session_id=sess_b, owner="alice")
    assert "error" in result

    listed = await ai.do_manage_memory("list", session_id=sess_a, owner="alice")
    assert "original text" in listed["results"]
    assert "hijacked text" not in listed["results"]


@pytest.mark.asyncio
async def test_persona_b_cannot_delete_persona_as_memory(memory_manager):
    sess_a, _ = _make_persona_session(owner="alice", name="Persona A")
    sess_b, _ = _make_persona_session(owner="alice", name="Persona B")
    added = await ai.do_manage_memory("add\nsurvives deletion attempt", session_id=sess_a, owner="alice")
    memory_id = added["memory_id"]

    result = await ai.do_manage_memory(f"delete\n{memory_id}", session_id=sess_b, owner="alice")
    assert "error" in result

    listed = await ai.do_manage_memory("list", session_id=sess_a, owner="alice")
    assert "survives deletion attempt" in listed["results"]


@pytest.mark.asyncio
async def test_shared_pool_memory_visible_to_every_persona(memory_manager):
    sess_a, _ = _make_persona_session(owner="alice", name="Persona A")
    unbound = _make_unbound_session(owner="alice")
    await ai.do_manage_memory("add\nshared fact for everyone", session_id=unbound, owner="alice")

    listed = await ai.do_manage_memory("list", session_id=sess_a, owner="alice")
    assert "shared fact for everyone" in listed["results"]


@pytest.mark.asyncio
async def test_unbound_session_keeps_todays_exact_behavior(memory_manager):
    """Fail-open: no persona binding at all -> the full owner pool, exactly
    like before this feature existed."""
    sess_a, _ = _make_persona_session(owner="alice", name="Persona A")
    unbound = _make_unbound_session(owner="alice")
    await ai.do_manage_memory("add\nA's fact", session_id=sess_a, owner="alice")
    await ai.do_manage_memory("add\nunbound fact", session_id=unbound, owner="alice")

    listed = await ai.do_manage_memory("list", session_id=unbound, owner="alice")
    assert "A's fact" in listed["results"]
    assert "unbound fact" in listed["results"]


@pytest.mark.asyncio
async def test_dangling_session_id_falls_back_to_owner_only(memory_manager):
    """Fail-open: a session_id that doesn't resolve to any real session must
    not error -- falls back to owner-only behavior, same as no session_id."""
    await ai.do_manage_memory("add\na fact", session_id=None, owner="alice")
    listed = await ai.do_manage_memory("list", session_id="does-not-exist", owner="alice")
    assert "a fact" in listed["results"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_manage_memory_persona_isolation.py -v --import-mode=importlib`
Expected: FAIL — persona B currently sees/edits/deletes persona A's memories, since `do_manage_memory` never resolves or applies `persona_id` today.

- [ ] **Step 3: Resolve `persona_id` once at the top of `do_manage_memory`**

`src/ai_interaction.py`'s `do_manage_memory` currently starts (lines 296-317):

```python
async def do_manage_memory(content: str, session_id: Optional[str] = None, owner: Optional[str] = None) -> Dict:
    """Manage memories: list, add, edit, delete, search.
    ...
    """
    if not _memory_manager:
        return {"error": "Memory manager not available"}

    lines = content.strip().split("\n")
    if not lines:
        return {"error": "Need at least 1 line: action"}

    action = lines[0].strip().lower()
```

Change to (add persona resolution right after the `_memory_manager` guard, mirroring
`routes/chat_helpers.py`'s `extract_preset` pattern exactly — same `SessionLocal` +
`resolve_crew_binding` shape, fail-open on any missing piece):

```python
async def do_manage_memory(content: str, session_id: Optional[str] = None, owner: Optional[str] = None) -> Dict:
    """Manage memories: list, add, edit, delete, search.

    If session_id/owner resolve to a persona-bound session, every action is
    scoped to that persona's memories plus the shared (unscoped) pool --
    never another persona's memories. An unbound session, a dangling
    binding, or a lookup failure all fall back to today's exact
    owner-only behavior.
    ...
    """
    if not _memory_manager:
        return {"error": "Memory manager not available"}

    persona_id = None
    if session_id is not None and owner is not None:
        try:
            from core.database import SessionLocal
            from src.crew_helpers import resolve_crew_binding
            db = SessionLocal()
            try:
                crew = resolve_crew_binding(db, session_id, owner)
                if crew:
                    persona_id = crew.id
            finally:
                db.close()
        except Exception:
            persona_id = None

    lines = content.strip().split("\n")
    if not lines:
        return {"error": "Need at least 1 line: action"}

    action = lines[0].strip().lower()
```

(The outer `try/except` around the DB work is defense-in-depth beyond what `extract_preset`
has — `resolve_crew_binding` itself already never raises, but `SessionLocal()` failing to
construct, however unlikely, must not break the memory tool.)

- [ ] **Step 4: Apply `persona_id` to the `list` action**

Currently (lines 319-321):

```python
    if action == "list":
        category_filter = lines[1].strip().lower() if len(lines) > 1 and lines[1].strip() else None
        memories = _memory_manager.load(owner=owner)
```

Change to:

```python
    if action == "list":
        category_filter = lines[1].strip().lower() if len(lines) > 1 and lines[1].strip() else None
        memories = _memory_manager.load(owner=owner, persona_id=persona_id)
```

- [ ] **Step 5: Apply `persona_id` to the `add` action**

Currently (line 345):

```python
        entry = _memory_manager.add_entry(text, source="ai_agent", category=category, owner=owner)
```

Change to:

```python
        entry = _memory_manager.add_entry(text, source="ai_agent", category=category, owner=owner, persona_id=persona_id)
```

- [ ] **Step 6: Apply `persona_id` to the `edit` action**

Currently (lines 373-384):

```python
        memories = _memory_manager.load_all()
        found = False
        for m in memories:
            if m.get("id", "").startswith(memory_id):
                # Verify ownership
                if owner and m.get("owner") != owner:
                    return {"error": f"Memory '{memory_id}' not found"}
                m["text"] = new_text
                m["timestamp"] = int(time.time())
                found = True
                full_id = m["id"]
                break
```

Change to (a sibling check alongside the existing ownership check, same "not found" response
shape so a persona boundary and an owner boundary are indistinguishable to the caller):

```python
        memories = _memory_manager.load_all()
        found = False
        for m in memories:
            if m.get("id", "").startswith(memory_id):
                # Verify ownership
                if owner and m.get("owner") != owner:
                    return {"error": f"Memory '{memory_id}' not found"}
                # Verify persona scope: a different persona's memory is invisible,
                # same as a different owner's -- not a permission error, just "not found".
                if persona_id is not None and m.get("persona_id") not in (None, persona_id):
                    return {"error": f"Memory '{memory_id}' not found"}
                m["text"] = new_text
                m["timestamp"] = int(time.time())
                found = True
                full_id = m["id"]
                break
```

- [ ] **Step 7: Apply `persona_id` to the `delete` action**

Currently (lines 404-415):

```python
        memories = _memory_manager.load_all()
        original_len = len(memories)
        full_id = None
        delete_id = None
        for m in memories:
            if m.get("id", "").startswith(memory_id):
                # Verify ownership
                if owner and m.get("owner") != owner:
                    return {"error": f"Memory '{memory_id}' not found"}
                full_id = m["id"]
                delete_id = m["id"]
                break
```

Change to:

```python
        memories = _memory_manager.load_all()
        original_len = len(memories)
        full_id = None
        delete_id = None
        for m in memories:
            if m.get("id", "").startswith(memory_id):
                # Verify ownership
                if owner and m.get("owner") != owner:
                    return {"error": f"Memory '{memory_id}' not found"}
                # Verify persona scope -- same "not found" shape as the ownership check above.
                if persona_id is not None and m.get("persona_id") not in (None, persona_id):
                    return {"error": f"Memory '{memory_id}' not found"}
                full_id = m["id"]
                delete_id = m["id"]
                break
```

- [ ] **Step 8: Apply `persona_id` to the `search` action**

Currently (line 435):

```python
        memories = _memory_manager.load(owner=owner)
```

Change to:

```python
        memories = _memory_manager.load(owner=owner, persona_id=persona_id)
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `pytest tests/test_manage_memory_persona_isolation.py -v --import-mode=importlib`
Expected: PASS (all 8 tests).

Then run the existing `ai_interaction`/memory-tool test coverage to confirm no regression —
find it first:

Run: `grep -rl "do_manage_memory\|dispatch_ai_tool" tests/`

Run each file that search returns with `pytest <file> -v --import-mode=importlib` and confirm
PASS. (This plan does not enumerate them by name since `dispatch_ai_tool` is a shared dispatcher
touched by many unrelated tool tests — confirm none of them broke rather than assuming a fixed
list.)

- [ ] **Step 10: Commit**

```bash
git add src/ai_interaction.py tests/test_manage_memory_persona_isolation.py
git commit -m "feat(memory): manage_memory tool respects persona isolation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Auto-injected chat context respects persona isolation

**Files:**
- Modify: `src/chat_processor.py` (`build_context_preface`)
- Test: `tests/test_chat_processor_memory_persona_isolation.py` (new)

**Interfaces:**
- Consumes: `MemoryManager.load(..., persona_id=...)` (Task 1).
- Produces: `ChatProcessor.build_context_preface(..., persona_id: Optional[str] = None, ...)` — when given, the memory portion of the returned preface (pinned + extended/recalled) is scoped to that persona's memories plus the shared pool.

- [ ] **Step 1: Write the failing test**

Create `tests/test_chat_processor_memory_persona_isolation.py`:

```python
"""Auto-injected chat context (build_context_preface) must respect the
same persona isolation manage_memory does -- a persona-bound turn's
background memory priming pulls only from that persona's memories plus
the shared pool. This is a thin proof that persona_id reaches
memory_manager.load() correctly; the filtering logic itself is already
covered by tests/test_memory_persona_isolation.py."""
from unittest.mock import MagicMock

from src.chat_processor import ChatProcessor


def _make_processor(loaded_memories):
    memory_manager = MagicMock()
    memory_manager.load.return_value = loaded_memories
    memory_manager.increment_uses = MagicMock()
    processor = ChatProcessor(memory_manager, personal_docs_manager=None, memory_vector=None, skills_manager=None)
    return processor, memory_manager


def test_build_context_preface_passes_persona_id_to_memory_manager_load():
    processor, memory_manager = _make_processor(loaded_memories=[])
    processor.build_context_preface(
        message="hello", session=MagicMock(), use_memory=True, owner="alice", persona_id="persona-a",
    )
    memory_manager.load.assert_called_once_with(owner="alice", persona_id="persona-a")


def test_build_context_preface_without_persona_id_matches_todays_call_shape():
    """Fail-open / backward compatibility: omitting persona_id must call
    memory_manager.load() exactly as it's called today (owner-only)."""
    processor, memory_manager = _make_processor(loaded_memories=[])
    processor.build_context_preface(
        message="hello", session=MagicMock(), use_memory=True, owner="alice",
    )
    memory_manager.load.assert_called_once_with(owner="alice", persona_id=None)


def test_build_context_preface_still_injects_a_pinned_memory():
    """Regression guard: the persona_id plumbing must not disturb the
    existing pinned/extended memory injection behavior."""
    processor, _ = _make_processor(loaded_memories=[
        {"id": "m1", "text": "core fact", "pinned": True, "category": "fact"},
    ])
    preface, _, _ = processor.build_context_preface(
        message="hello", session=MagicMock(), use_memory=True, owner="alice", persona_id="persona-a",
    )
    joined = " ".join(m.get("content", "") for m in preface)
    assert "core fact" in joined
```

(`ChatProcessor`'s constructor signature — `memory_manager, personal_docs_manager, memory_vector=None, skills_manager=None` — is taken from `src/app_initializer.py`'s existing construction call; read the live constructor before writing this test in case its parameter order or defaults have shifted since this plan was written.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chat_processor_memory_persona_isolation.py -v --import-mode=importlib`
Expected: FAIL — `build_context_preface()` doesn't accept `persona_id` yet.

- [ ] **Step 3: Add the `persona_id` parameter**

`src/chat_processor.py`'s `build_context_preface` signature currently reads (lines 198-212):

```python
    def build_context_preface(
        self,
        message: str,
        session: Any,
        use_web: bool = False,
        use_rag: bool = True,
        use_memory: bool = True,
        time_filter: Optional[str] = None,
        preset_system_prompt: Optional[str] = None,
        owner: Optional[str] = None,
        character_name: Optional[str] = None,
        agent_mode: bool = False,
        incognito: bool = False,
        use_skills: bool = True,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], List[Dict[str, str]]]:
```

Add `persona_id` (placed near `owner`, since it's used at the exact same call site below):

```python
    def build_context_preface(
        self,
        message: str,
        session: Any,
        use_web: bool = False,
        use_rag: bool = True,
        use_memory: bool = True,
        time_filter: Optional[str] = None,
        preset_system_prompt: Optional[str] = None,
        owner: Optional[str] = None,
        persona_id: Optional[str] = None,
        character_name: Optional[str] = None,
        agent_mode: bool = False,
        incognito: bool = False,
        use_skills: bool = True,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], List[Dict[str, str]]]:
```

- [ ] **Step 4: Thread it into the memory load call**

Currently (line 248):

```python
            mem_entries = self.memory_manager.load(owner=owner)
```

Change to:

```python
            mem_entries = self.memory_manager.load(owner=owner, persona_id=persona_id)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_chat_processor_memory_persona_isolation.py -v --import-mode=importlib`
Expected: PASS (all 3 tests).

Then run this file's existing test coverage to confirm no regression — find it first:

Run: `grep -rl "build_context_preface\|ChatProcessor" tests/`

Run each file that search returns with `pytest <file> -v --import-mode=importlib` and confirm
PASS (same reasoning as Task 2 Step 9 — `build_context_preface` is a widely-used method, don't
assume a fixed list).

- [ ] **Step 6: Commit**

```bash
git add src/chat_processor.py tests/test_chat_processor_memory_persona_isolation.py
git commit -m "feat(memory): auto-injected chat context respects persona isolation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Wire `persona_id` from the chat request into `build_context_preface`

**Files:**
- Modify: `routes/chat_helpers.py` (`PresetInfo`, `extract_preset`, `build_chat_context`)
- Test: `tests/test_chat_helpers.py`

**Interfaces:**
- Consumes: `ChatProcessor.build_context_preface(..., persona_id=...)` (Task 3).
- Produces: `PresetInfo` gains a `persona_id: Optional[str] = None` field, populated by `extract_preset` from the same `resolve_crew_binding` call it already makes for personality override — no second DB round-trip. `build_chat_context` reads `preset.persona_id` and passes it into `_preface_kwargs`.

- [ ] **Step 1: Write the failing test**

`extract_preset` already has test coverage in `tests/test_chat_helpers.py` (it was added in the
Crew core sub-project). Read that file's existing `extract_preset` tests first, then add:

```python
def test_extract_preset_returns_persona_id_when_session_bound_to_a_persona(monkeypatch):
    import uuid
    from core.database import SessionLocal, CrewMember, Session as DbSession
    from routes.chat_helpers import extract_preset

    db = SessionLocal()
    try:
        crew_id = str(uuid.uuid4())
        db.add(CrewMember(id=crew_id, owner="alice", name="Nav", personality="Be terse."))
        sess_id = str(uuid.uuid4())
        db.add(DbSession(id=sess_id, name="s", endpoint_url="http://x", model="m",
                         owner="alice", crew_member_id=crew_id))
        db.commit()
    finally:
        db.close()

    class _FakeSess:
        id = sess_id

    class _FakeHandler:
        def validate_and_extract_preset(self, preset_id):
            return (0.7, 2000, "default prompt", "Default")

    preset = extract_preset(_FakeHandler(), "default", sess=_FakeSess(), owner="alice")
    assert preset.persona_id == crew_id


def test_extract_preset_persona_id_none_for_unbound_session(monkeypatch):
    import uuid
    from core.database import SessionLocal, Session as DbSession
    from routes.chat_helpers import extract_preset

    db = SessionLocal()
    try:
        sess_id = str(uuid.uuid4())
        db.add(DbSession(id=sess_id, name="s", endpoint_url="http://x", model="m", owner="alice"))
        db.commit()
    finally:
        db.close()

    class _FakeSess:
        id = sess_id

    class _FakeHandler:
        def validate_and_extract_preset(self, preset_id):
            return (0.7, 2000, "default prompt", "Default")

    preset = extract_preset(_FakeHandler(), "default", sess=_FakeSess(), owner="alice")
    assert preset.persona_id is None


def test_extract_preset_persona_id_none_when_sess_or_owner_missing():
    from routes.chat_helpers import extract_preset

    class _FakeHandler:
        def validate_and_extract_preset(self, preset_id):
            return (0.7, 2000, "default prompt", "Default")

    preset = extract_preset(_FakeHandler(), "default")
    assert preset.persona_id is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chat_helpers.py -k persona_id -v --import-mode=importlib`
Expected: FAIL — `PresetInfo` has no `persona_id` field yet (`AttributeError`).

- [ ] **Step 3: Add `persona_id` to `PresetInfo`**

`routes/chat_helpers.py`'s `PresetInfo` currently reads (lines 69-75):

```python
@dataclass
class PresetInfo:
    """Extracted preset parameters."""
    temperature: Optional[float]
    max_tokens: Optional[int]
    system_prompt: Optional[str]
    character_name: Optional[str]
```

Add `persona_id` as a 5th field with a default (other test files construct `PresetInfo`
directly with only the original 4 fields — a default keeps those call sites working
unchanged; confirmed via grep that `tests/test_kv_cache_invalidation_2927.py` and
`tests/test_review_regressions.py` both construct `PresetInfo` directly):

```python
@dataclass
class PresetInfo:
    """Extracted preset parameters."""
    temperature: Optional[float]
    max_tokens: Optional[int]
    system_prompt: Optional[str]
    character_name: Optional[str]
    persona_id: Optional[str] = None
```

- [ ] **Step 4: Populate `persona_id` in `extract_preset`**

Currently (lines 335-361):

```python
def extract_preset(chat_handler, preset_id, sess=None, owner=None) -> PresetInfo:
    """Extract preset parameters via chat_handler. If `sess`/`owner` are given
    and the session is bound to a persona (CrewMember) with a non-empty
    personality, the persona's personality/name REPLACE the preset's
    system_prompt/character_name for this turn -- the persona already
    defines the assistant's voice for a persona-bound conversation. A
    missing/dangling binding falls back to normal preset-only behavior."""
    temperature, max_tokens, system_prompt, char_name = (
        chat_handler.validate_and_extract_preset(preset_id)
    )
    if sess is not None and owner is not None:
        from core.database import SessionLocal
        from src.crew_helpers import resolve_crew_binding
        db = SessionLocal()
        try:
            crew = resolve_crew_binding(db, sess.id, owner)
            if crew and crew.personality:
                system_prompt = crew.personality
                char_name = crew.name
        finally:
            db.close()
    return PresetInfo(
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        character_name=char_name,
    )
```

Change to (capture `persona_id` from the SAME `resolve_crew_binding` call — no second lookup —
regardless of whether the persona has a personality set, since memory isolation applies to every
bound persona, not just ones with a custom personality):

```python
def extract_preset(chat_handler, preset_id, sess=None, owner=None) -> PresetInfo:
    """Extract preset parameters via chat_handler. If `sess`/`owner` are given
    and the session is bound to a persona (CrewMember), its id is returned
    as persona_id (used for memory isolation, regardless of whether it has
    a custom personality) and -- when it has a non-empty personality -- the
    persona's personality/name REPLACE the preset's system_prompt/
    character_name for this turn. A missing/dangling binding falls back to
    normal preset-only behavior with persona_id=None."""
    temperature, max_tokens, system_prompt, char_name = (
        chat_handler.validate_and_extract_preset(preset_id)
    )
    persona_id = None
    if sess is not None and owner is not None:
        from core.database import SessionLocal
        from src.crew_helpers import resolve_crew_binding
        db = SessionLocal()
        try:
            crew = resolve_crew_binding(db, sess.id, owner)
            if crew:
                persona_id = crew.id
                if crew.personality:
                    system_prompt = crew.personality
                    char_name = crew.name
        finally:
            db.close()
    return PresetInfo(
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        character_name=char_name,
        persona_id=persona_id,
    )
```

- [ ] **Step 5: Thread `persona_id` into `build_chat_context`'s `_preface_kwargs`**

Currently (lines 732-744):

```python
    _preface_kwargs = dict(
        message=_ctx_msg,
        session=sess,
        use_web=use_web and not skip_web,
        use_memory=mem_enabled,
        time_filter=time_filter,
        preset_system_prompt=preset.system_prompt,
        owner=user,
        character_name=preset.character_name,
        agent_mode=agent_mode,
        incognito=incognito,
        use_skills=skills_enabled,
    )
```

Add `persona_id=preset.persona_id` (placed near `owner=user`, matching Task 3's parameter
ordering):

```python
    _preface_kwargs = dict(
        message=_ctx_msg,
        session=sess,
        use_web=use_web and not skip_web,
        use_memory=mem_enabled,
        time_filter=time_filter,
        preset_system_prompt=preset.system_prompt,
        owner=user,
        persona_id=preset.persona_id,
        character_name=preset.character_name,
        agent_mode=agent_mode,
        incognito=incognito,
        use_skills=skills_enabled,
    )
```

Read the live file before editing — `preset = extract_preset(...)` must already be assigned
above this block (it is, at line 668, well before line 732) so `preset.persona_id` is in scope.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_chat_helpers.py -v --import-mode=importlib`
Expected: PASS (full file — confirms the 3 new tests pass and nothing else in this
already-much-modified file broke).

Then run the broader regression sweep this whole plan touches:

Run: `pytest --import-mode=importlib -k "memory or chat_helpers or chat_processor or manage_memory" -v`
Expected: PASS across the board.

- [ ] **Step 7: Commit**

```bash
git add routes/chat_helpers.py tests/test_chat_helpers.py
git commit -m "feat(memory): thread persona_id from chat requests into auto-injected memory

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** every section of `docs/superpowers/specs/2026-08-06-crew-memory-isolation-design.md` maps to a task — data model (Task 1's `persona_id` key), the two real consumers (`do_manage_memory` = Task 2, auto-injection = Tasks 3-4), fail-open error handling (every task's fail-open tests), the explicit cross-persona isolation test requirement (Task 2's 8-test battery is the core of this).
- **Mechanism correction from spec to plan:** documented prominently at the top of this plan (not just here) — the spec attributed the owner-filtering pattern to `NativeMemoryProvider`, which live-code tracing during this plan's own research found has zero production callers. The requirements are unaffected; only the target files changed, from `src/memory_provider.py`/`services/memory/service.py`/`src/app_initializer.py` (none touched by this plan) to `src/memory.py`/`src/ai_interaction.py`/`src/chat_processor.py`/`routes/chat_helpers.py` (all four tasks above).
- **Placeholder scan:** no TBDs. Task 2 Step 9 and Task 3 Step 5 both say "find the existing test coverage via grep, run it, confirm no regression" instead of naming files directly — deliberate, since `dispatch_ai_tool`/`build_context_preface` are shared infrastructure touched by many unrelated tool/chat tests; grepping and running whatever exists is more robust than a plan-time guess at file names that could already be stale by execution time.
- **Type consistency:** `persona_id` is the exact parameter/field name used everywhere across all 4 tasks — `MemoryManager.add_entry`/`load`, `do_manage_memory`'s local variable, `ChatProcessor.build_context_preface`, `PresetInfo.persona_id`, and `_preface_kwargs["persona_id"]`. No renaming drift between tasks.
