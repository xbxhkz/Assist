# Multi-Persona System sub-project 3: Memory Isolation — Design Spec

## Context

This is the third and final sub-project of the Multi-Persona initiative. Sub-project 1 (core
Crew system — personas with personality/model/tool overrides, `5a047739..4737ec43`) and
sub-project 2 (voice-per-persona, `981049d..8076be83`) have both shipped on `dev`. This spec
covers isolating a persona's *memory* — today, all of a user's memories live in one flat pool
regardless of which persona (if any) a conversation was bound to.

## Current State (as of this spec)

- **Storage**: `MemoryManager` (`src/memory.py`) — a flat JSON file (`memory.json`), one list of
  entries per install. `add_entry(text, source, category, owner=None)` creates an entry;
  `load(owner=None)` optionally filters by `owner`. An optional `MemoryVectorStore`
  (`src/memory_vector.py`, Chroma-backed) provides semantic search over the same entries, keyed
  by memory ID only — it has no owner/persona filtering built in.
- **Provider layer**: `NativeMemoryProvider` (`src/memory_provider.py`) sits between the tool/route
  layer and `MemoryManager`. Its `recall()` runs a *global* vector search, then intersects the
  results against an owner-pre-filtered candidate set (`memory_manager.load(owner=owner)`), plus
  a defense-in-depth per-entry check (`entry.get("owner") != owner` → skip). `delete()` has the
  identical pattern: an entry belonging to a different owner is treated as not found, not a
  permission error. `list_memories()` is a thin pass-through to `load(owner=...)`.
  `MemoryRecord`/`MemorySearchHit` already carry a `session_id` field, but it is provenance only
  — never used to filter anything today.
- **Two real consumers**:
  1. `do_manage_memory` (`src/ai_interaction.py:296`) — the explicit agent tool
     (`list`/`add`/`edit`/`delete`/`search`). Its signature already accepts `session_id` and
     `owner`; `session_id` is currently unused beyond being stamped onto new entries.
  2. Auto-injected context-priming in `routes/chat_helpers.py` — a background semantic-match
     pass that runs on (almost) every chat turn when memory is enabled (`mem_enabled`, a
     per-user preference, gated off for incognito/no-memory/research-spinoff/low-signal turns).
     Already has the session object in scope where this gate is evaluated.
  3. `services/memory/service.py` (`MemoryService`) is a third, thinner wrapper with only a
     `session_id` param and no `owner` concept at all. Confirmed via grep (`MemoryService(` —
     only 2 matches outside its own docstring example, both in `MemoryService`'s own test files)
     that it has **no production call site at all**. Dead code from this feature's perspective —
     not touched by this sub-project.
- **Confirmed dead code, not a live consumer**: `core.database.Memory` (a SQLAlchemy table,
  `core/database.py:675`) is never written to anywhere in this codebase (grepped for every
  constructor call — none exist). `builtin_actions.py`'s event-classifier feature queries this
  table for "personal context," but always gets an empty result in practice today. Out of scope
  for this sub-project — it isn't a live memory-isolation surface.

## Goal

A chat session bound to a persona only ever sees, searches, edits, or deletes memories created
by that same persona, plus a shared "no persona" pool — never another persona's memories. A
session with no persona bound keeps today's exact behavior (the full shared pool). Nothing
changes for any existing installation until a user actually starts using multiple personas.

## Architecture

**Hard isolation** (confirmed): a persona-bound session sees `{its own persona's memories} ∪
{the shared/unscoped pool}`, never another persona's. There is no "search everyone's memories"
override.

**Unscoped bucket** (confirmed): every memory that exists before this feature ships, and every
memory created in a session with no persona bound, lands in the shared pool — visible to every
persona and to unbound sessions alike. Isolation only activates once a session *is* bound to a
specific persona.

Resolve `persona_id` the same way sub-project 2 resolved a session's voice: via the already-shared
`resolve_crew_binding(db, session_id, owner)` (`src/crew_helpers.py`), at the moment memory is
read or written — not by threading a new concept through call chains that don't already have
`session_id`/`owner` in scope. Both real consumers already have both.

Filtering extends the *exact* pattern this codebase already uses for `owner` — a second
skip-check alongside the existing one, not a new mechanism:

```python
if owner is not None and entry.get("owner") != owner:
    continue
if persona_id is not None and entry.get("persona_id") not in (None, persona_id):
    continue
```

This one addition, applied consistently everywhere the existing owner check already appears
(`recall()`'s defense-in-depth loop, `delete()`'s match loop, and `load()`'s base filter), is
the entire isolation mechanism. No new storage engine, no new query layer.

## Data Model

No schema migration — `MemoryManager`'s entries are a flat JSON dict per entry, not a fixed-
column table. Add one optional key, set only when present (mirroring exactly how `owner` is
already handled in `add_entry`):

```python
if persona_id:
    entry["persona_id"] = persona_id
```

`MemoryRecord` (`src/memory_provider.py`) gains a `persona_id: Optional[str] = None` field
alongside its existing `owner`/`session_id` fields.

## API / Call-Signature Changes

- `src/memory.py`: `MemoryManager.add_entry(..., persona_id: str = None)`,
  `MemoryManager.load(..., persona_id: str = None)` — both mirror the existing `owner` parameter
  exactly (optional, `None` = no additional filtering).
- `src/memory_provider.py`: `NativeMemoryProvider.remember()` / `recall()` / `list_memories()` /
  `delete()` each gain a `persona_id: Optional[str] = None` keyword parameter, threaded straight
  through to the `MemoryManager` calls above and applied as the second skip-check shown above.
- `src/ai_interaction.py`: `do_manage_memory` resolves `persona_id` once, at the top of the
  function (it already receives `session_id` and `owner`), and passes it into every action.
- `routes/chat_helpers.py`: the auto-injection call site resolves and passes `persona_id` the
  same way.
- `services/memory/service.py` (`MemoryService`): **not touched** — confirmed no production call
  site (see Current State above). Patching it would be speculative work with no live consumer to
  verify against.

## Error Handling

Every failure mode reuses an established precedent from sub-project 2, applied to memory instead
of voice:

- No `session_id` passed to a memory operation → today's exact behavior, unchanged (owner-only
  filtering, no persona narrowing).
- `session_id` passed but the session has no persona binding → `persona_id` resolves to `None`
  → no additional filtering (the "unbound session" case, working exactly as it does today).
- Persona bound, but `resolve_crew_binding` fails for any reason (dangling reference, DB error)
  → fail-open to `persona_id = None` → today's owner-only behavior, never an error surfaced to
  the memory caller.
- A memory with no `persona_id` key (every pre-existing entry, plus every unbound-session entry
  going forward) is visible to every persona — this is the shared-pool behavior, by design, not
  a gap.
- Editing or deleting a memory that belongs to a *different* persona is treated as "not found,"
  identical to how a different owner's memory is already treated today — no new error type, no
  permission-denied response.

## Testing

- The same fail-open battery sub-project 2 established for `resolve_crew_binding`-based
  features (unbound session, dangling persona reference, lookup failure → falls back to
  owner-only behavior), applied to `add_entry`/`load`/`recall`/`delete`, against real
  `MemoryManager` entries and real DB-backed session/persona rows — not mocks.
- **Explicit cross-persona isolation tests are the core of this feature's test suite** — unlike
  the voice feature (getting it wrong just plays the wrong voice), a leak here is a real
  information-isolation bug. For each of list/search/recall/edit/delete: persona A creates a
  memory, persona B (same owner, different persona) must not see it via that action; a
  shared-pool (no-persona) memory must be visible to both.
- Auto-injection: a persona-bound session's context-priming pass must only pull from
  `{that persona} ∪ {shared pool}` — verified by asserting on the resolved `persona_id` filter
  reaching the same `recall()` path already covered by the provider-level tests above, not by
  re-testing the whole chat pipeline.

## Out of Scope

- `core.database.Memory` (the SQLAlchemy table) and the event-classifier feature that reads
  it — confirmed dead code, not a live memory-isolation surface.
- Migrating/backfilling existing memories to any persona — they stay in the shared pool
  permanently unless a user manually re-creates them under a specific persona.
- A UI affordance for moving a memory between personas or into/out of the shared pool.
- Any change to `MemoryVectorStore`'s indexing itself — the vector search stays global; isolation
  is enforced entirely by the existing post-search Python-side filtering layer, extended with one
  more condition.
