# Task Continuity Engine — Design

**Status:** approved in chat (3 sections), spec written 2026-09-29.
**Repo:** `C:\Users\Admin\unsloth`, on `unslothed-release`.
**Source:** requested via a master build prompt for a unified AI runtime
(`unsloth_unified_ai_memory_master_prompt.txt`, §17–28, §34.7–34.9, §35). This spec is
piece 1 of 3 in that initiative (Task Continuity → Unified Memory core → AirLLM),
chosen first and explicitly required to be **reconciled** with what already exists
rather than built as a fourth parallel state store.

**Platform:** this machine only — NVIDIA (RTX 5080 Laptop, 16 GiB VRAM, sm_120 Blackwell,
CUDA 13.0) and Windows 11. No portability requirement. Where the master prompt names Linux
as primary target, this spec overrides it.

---

## 1. Why this exists, and what it is not

The master prompt's stated purpose (§17): let a coding/agent system survive context-window
exhaustion without rereading the conversation, and let a task graph persist independent of
any one conversation. It is explicit that this is project execution state, **not chat
memory** (§17, §24 "Never claim work is complete merely because the AI remembers doing it").

**This project already has three live stores doing large parts of that job.** Phase 0
inspection (done in chat, not repeated here) found:

| existing store | what it does today | master-prompt artifact it already covers |
|---|---|---|
| `.superpowers/sdd/<project>/progress.md` | per-task status and numbered rulings, each with a reason and a cost-if-wrong, appended as work lands | §20 `decisions.json` |
| `C:\Users\Admin\.claude\projects\...\memory\*.md` + `MEMORY.md` | durable cross-session facts, one file per fact, an index loaded first (index-first = progressive disclosure) | §21 `context_summary.md` / progressive disclosure |
| `C:\Users\Admin\odysseus\.remember\` (`now.md`, `today-*.md`, `recent.md`, `archive.md`) | rolling session activity, daily rotation, older slices compacted | §20 `logs/activity.jsonl` |
| `core/inference/tool_audit/` (this repo, shipped) | every tool call this app runs, redacted, retained, attributed by `delegation_id` | the **app-side** half of §20 activity, for the shipped agent specifically |

**Evidence the gap is real, not cosmetic.** The model-roles-delegation plan
(`docs/superpowers/plans/2026-09-20-unslothed-model-roles-delegation.md`) has **59
unchecked checkboxes and 0 checked**, on work that is fully merged and shipped. The
checkbox mechanism is not merely unused — it actively lies about progress to whoever opens
it. `progress.md` for the same project is accurate, because it is prose written and read by
the same agent that also wrote the code, but it is not machine-queryable: nothing can ask
"which tasks are ready" or "did we try this fix before" without re-reading a document meant
for humans.

**Design rule for this piece:** add only what is structurally missing. Where a master-prompt
artifact is already served, this engine points at the existing store instead of duplicating
it. Building `decisions.json` and `logs/activity.jsonl` fresh would recreate, in a different
domain, the exact two-authorities failure the master prompt itself forbids for AirLLM
(§34.3–34.4: "AirLLM is an execution backend... UnifiedMemoryManager owns residency").

## 2. Reconciliation table

| master-prompt artifact (§17–28) | verdict | reasoning |
|---|---|---|
| `.ai/project_state.json` | **build** | nothing today is machine-readable current state; genuinely missing |
| `.ai/task_queue.json` | **build** | the real gap; dependencies, acceptance criteria and derived readiness do not exist anywhere today |
| `.ai/logs/errors.jsonl` | **build** | nothing records failed approaches; this project has hit the same class of mistake twice in one session (see §3) |
| `.ai/checkpoints/` | **build, narrow** | intent-at-a-moment, distinct from what git commits (code, not decision context) |
| `.ai/decisions.json` | **drop — point at `progress.md`** | already served, already accurate, already has reason + cost-if-wrong per entry |
| `.ai/logs/activity.jsonl` | **drop — point at `.remember` (dev) and tool_audit (app)** | already served on both sides; duplicating it risks drift between two records of the same events |
| `.ai/context_summary.md` | **generate, never hand-author** | `MEMORY.md` already does index-first disclosure; this file becomes a *rendered view* of `project_state.json` + `task_queue.json`, not a third thing to keep in sync |
| `.ai/project_state.md` | **generate** | same reasoning; a rendering, not a second source of truth |
| `.ai/locks/` | **defer to a later piece** | no concurrent writer exists yet in this project's actual usage (subagents are dispatched serially); build it when a second writer does, not speculatively |

## 3. Concrete motivation for `errors.jsonl`

Not hypothetical. In the model-roles-delegation work on this branch:

- **Sixteen inert negative controls** shipped across this project's history before being
  caught by mutation testing — a control that passes whether or not the guard it is meant to
  test exists. Several were the same *shape* of mistake (asserting on a message string
  instead of on the protected behaviour) recurring across unrelated tasks.
- **`git checkout -- <file>` used to restore a mutation destroyed uncommitted work — twice**,
  once on a fix wave a rate-limited subagent left uncommitted, once on the controller's own
  freshly-written validator check. Both were recorded afterward in a memory file
  (`unslothed-sdd-controller-blind-spots.md`) as a lesson — but that file is read at session
  start, not queried mid-task before a risky operation.

`errors.jsonl` exists so "have we tried this, and did it fail" is a query answerable *before*
repeating the mistake a third time, not a memory file someone has to remember to open.

## 4. Architecture: one core, two consumers

```
studio/backend/core/continuity/
    __init__.py        -- public API (see §5)
    schemas.py          -- ProjectState, Task, TaskQueue, ErrorEntry, dataclasses
    storage.py           -- atomic read/write, schema versioning
    render.py             -- project_state.md / context_summary.md generation
```

Pure-Python, dependency-free beyond the standard library and `dataclasses` — same reasoning
as `delegation/subagent.py`'s isolation: importable and testable with no backend, no model,
no network.

**Dev consumer:** `studio/backend/tools/continuity_cli.py`, invoked directly with the venv
Python the way `packaging/check_branding.py` and `studio/install_llama_prebuilt.py` already
are. Operates on a `.ai/` directory the caller names explicitly (typically the repo root).
This is what a controller session runs at a context reset.

**App consumer:** a `continuity_task` built-in tool inside the running Unslothed agent,
registered like any other tool (five points, additive seam — see §7). Operates on a `.ai/`
directory confined to the conversation's sandbox workdir, exactly as `resolve_image_bytes`
confines image paths.

**Why one core:** a single implementation means `validate`/`repair` behave identically
whichever caller wrote the state, and a future consumer (§34's `memory-agent`,
`testing-agent`) needs no new code, only a new caller of the same functions.

## 5. Schemas

### `project_state.json`

```json
{
  "schema_version": 1,
  "project": "string",
  "status": "active | paused | complete",
  "current_phase": "string",
  "current_task": "task-id or null",
  "completion_percent": 0,
  "last_checkpoint": "checkpoint filename or null",
  "completed": ["task-id", "..."],
  "in_progress": ["task-id", "..."],
  "remaining": ["task-id", "..."],
  "next_action": "string, one sentence",
  "modified_files": ["path", "..."],
  "tests": {"cmd": "string", "last_result": "string", "at": "iso8601"},
  "blocking_issues": ["string", "..."],
  "important_decisions": ["pointer strings into progress.md, e.g. Ruling 15, not copies"]
}
```

`important_decisions` holds **pointers**, never copied text — the reconciliation rule from
§2 applies inside the schema itself, not just at the file-existence level.

### `task_queue.json`

```json
{
  "schema_version": 1,
  "tasks": [
    {
      "id": "string",
      "title": "string",
      "status": "pending | ready | in_progress | blocked | complete | failed | abandoned",
      "depends_on": ["task-id", "..."],
      "acceptance_criteria": ["string", "..."],
      "owner": "string or null"
    }
  ]
}
```

`status = "ready"` is **derived**, not hand-set: a `pending` task whose every `depends_on`
entry is `complete` is ready. Storing "ready" as an independent field would be a second
source of truth for a fact computable from the others — the exact anti-pattern the
reconciliation itself is meant to avoid.

Legal status transitions (checked by `set_task_status`, illegal ones raise
`ContinuityError` rather than silently applying):

```
pending -> in_progress -> {complete, failed, blocked}
blocked -> pending
failed -> abandoned, or -> pending (retry)
any -> abandoned (explicit give-up)
complete is terminal
```

**`"ready"` is not a state in this table, on purpose — it is never a legal `set_task_status`
target.** An earlier draft of this table wrote `pending -> ready` and `blocked -> ready` as
transitions, directly contradicting the paragraph immediately above stating "ready" is derived
and never hand-set: if it can also be written by `set_task_status`, it is a second source of
truth for the same fact, able to disagree with the derived one. Caught before implementation
reached this state, during the plan's Task 3 review — a caller decides a `pending` task is
unblocked by consulting `ready_tasks()` (or the rendered summary), then calls `set_task_status`
straight to `in_progress`; there is no intermediate stored state to pass through. A `blocked`
task returns to `pending`, not to a "ready" it could only otherwise reach by having been blocked.
`set_task_status(id, "ready")` is refused with "unknown status", the same as any other invalid
target, not by a special case — "ready" is absent from the valid-status set entirely.

Per §19: never mark `complete` because code was written. `set_task_status(..., "complete")`
requires `acceptance_criteria` to be non-empty on that task — an empty list refuses the
transition rather than allowing a criterion-free "done".

### `logs/errors.jsonl`

Append-only, one JSON object per line:

```json
{"ts": "iso8601", "project": "string", "what_was_tried": "string", "why_it_failed": "string", "symptom": "string, short, for matching", "do_not_repeat": true}
```

`symptom` is a short free-text tag (e.g. `"inert negative control"`, `"git checkout destroys uncommitted work"`)
used by `prior_failures(project_dir, symptom_query)` for substring/keyword matching — no
embedding search, no external dependency, consistent with "avoid unnecessary dependencies"
(§30).

### `checkpoints/<timestamp>.json`

A snapshot of `project_state.json` plus a free-text `note`, written by explicit call
(`continuity checkpoint "..."`) or by the app tool before a risky operation. Pruned to the
most recent N (default 20) on write — checkpoints are for "what was the intent a moment
ago", not an unbounded history (git already is the unbounded history, for code).

## 6. Core API

```python
# studio/backend/core/continuity/__init__.py

def load_state(project_dir: str) -> Optional[ProjectState]: ...
def write_state(project_dir: str, state: ProjectState) -> None: ...      # atomic: tmp + os.replace
def update_state(project_dir: str, **fields) -> ProjectState: ...        # read-modify-write, same atomicity

def load_tasks(project_dir: str) -> TaskQueue: ...
def add_task(project_dir: str, task: Task) -> None: ...
def set_task_status(project_dir: str, task_id: str, status: str) -> None:  # raises ContinuityError on an illegal transition (see §5) or a missing id
def ready_tasks(project_dir: str) -> list[Task]: ...                      # derived; never read from a stored field

def record_error(project_dir: str, *, what_tried: str, why_failed: str, symptom: str) -> None:
    # Never-raises internally: a failure to write here falls back to appending into
    # .remember's buffer (now.md) rather than losing the record of a failure.
def prior_failures(project_dir: str, symptom_query: str) -> list[ErrorEntry]: ...

def checkpoint(project_dir: str, note: str) -> str:                       # returns the written filename; prunes to the most recent N
def validate(project_dir: str) -> list[Problem]: ...                      # invalid JSON, orphan task ids in depends_on, impossible deps (a cycle), duplicate ids
def repair(project_dir: str) -> RepairReport: ...                         # regenerates DERIVED data only (ready status, project_state.md, context_summary.md); NEVER marks a task complete

def render_context_summary(project_dir: str) -> str: ...                  # 300-800 tokens target, ~1500 hard cap; raises ContinuityError if it cannot fit even after truncation, rather than silently exceeding the cap
def render_project_state_md(project_dir: str) -> str: ...
```

**Raising posture.** The core module raises `ContinuityError` (and lets a genuine bug raise
its real exception) rather than guessing at recovery. §28 says repair "must never silently
mark unfinished work complete" — the stronger, more general version of that rule is: **the
core never silently converts an error into a plausible-looking success.** Callers that need
never-raises behaviour (the app tool) wrap it themselves; that is the caller's job, not the
core's, exactly as `tool_audit.around()` wraps `execute_tool` rather than `execute_tool`
wrapping itself.

**Atomicity.** Every write is temp-file-then-`os.replace`, matching what a killed write must
guarantee: the real file is either the old content or the new content, never torn. No file
locking in v1 (§2, `.ai/locks/` deferred) — a concurrent writer can lose an update to a
last-write-wins race, but atomicity means it can never corrupt the file into invalid JSON.
That is the stated boundary: safe-if-racy, not race-free.

## 7. App-side tool: `continuity_task`

A built-in tool, registered at the five points this codebase's tools always need:
`ALL_TOOLS`, the `execute_tool` dispatch, the `routes/inference.py` `_addable` re-add (missed
three times in this project's history — checked explicitly in the plan), `_ALWAYS_SAFE_TOOLS`
(continuity_task reads/writes local state only, no network, no code execution — it belongs on
the safe list, unlike `ask_model`), and `_ANTHROPIC_UNPROMPTED_SAFE_TOOLS`.

```
continuity_task(action, ...)
  action = "status"            -> render_context_summary
  action = "add_task"          -> add_task
  action = "set_status"        -> set_task_status
  action = "record_error"      -> record_error
  action = "check_prior_failures" -> prior_failures
  action = "checkpoint"        -> checkpoint
```

`project_dir` is never a tool argument — it is always resolved from the conversation's
sandbox workdir the same way `_edit_file_resolve` and `resolve_image_bytes` do, so a model
cannot point continuity state at an arbitrary path.

**Never-raises at the handler boundary**, same shape as every other tool here: `execute()`
wraps the call in `except BaseException`, returns an error string. A continuity read/write
failure degrades the turn; it must not break it.

**Explicitly out of scope for this piece:** a delegate (from `ask_model`) writing its own
task state back into the primary's project. That is a real future use of this engine — it is
what would make §26's "future specialized agents share state while preventing simultaneous
ownership" concrete — but it needs its own design for the ownership boundary between a
delegate's task view and the primary's, and is deliberately not built here.

## 8. Testing

Pure-function core, testable with no backend, no model, no network — same posture as
`delegation/subagent.py`'s tests.

Required negative controls, each proven live by mutation (this project's standing rule:
break the guard, show the named test fail for the right reason):

- **Atomicity under a killed write**: simulate a process death mid-write (write the temp
  file, do not call `os.replace`) and confirm the original file is untouched and still valid
  JSON. Mutation: skip the temp-file step and write in place; the test must then fail on a
  torn/corrupted file.
- **Illegal status transition refused**: `complete -> in_progress` must raise. Mutation:
  remove the transition table's enforcement; the test must fail by observing the illegal
  transition silently succeeding, not merely by a missing exception type.
- **`complete` requires non-empty acceptance criteria**: a task with `acceptance_criteria: []`
  must refuse the transition to `complete`. Mutation: drop the check; test must fail by
  observing a criterion-free task marked complete.
- **`repair` never marks a task complete**: run `repair` against a fixture with an
  `in_progress` task and corrupted derived data; assert the task's status is unchanged after
  repair. Mutation: have `repair` "fix" the corruption by completing the task; test must fail
  on the status change, not on a missing field.
- **`render_context_summary` respects its cap**: a fixture project state large enough to
  exceed 1500 tokens must render truncated (or raise, per §6) rather than silently exceeding
  the cap. Mutation: remove the cap enforcement; test must fail by measuring an oversized
  render, not by a missing function call.
- **Path confinement**: a `project_dir` argument attempting to escape the sandbox workdir
  (`../../`, absolute path outside it) must resolve inside the workdir or refuse, mirroring
  the existing `resolve_image_bytes` test shape.

No test may load a model, start a backend, or call a real tool — this project's standing
constraint, checked because the core needs none of those to test.

## 9. Explicitly deferred

- `.ai/locks/` (§2) — no concurrent writer exists in this project's actual usage yet.
- Delegate-writes-back-to-primary task state (§7) — needs its own ownership design.
- Any UI surface for `task_queue.json` / `project_state.json` — this piece is the CLI and the
  tool only; a Settings-style panel (if wanted) is a follow-on, the same way the model-roles
  API shipped before its Settings panel did.
- Locking/multi-agent coordination beyond what serial subagent dispatch already provides
  (§26) — revisit once a second concurrent writer is real, not speculative.

## 10. Self-review

- **Placeholder scan:** no TBD/TODO; every schema field and API function has a stated
  behaviour, including the failure behaviour.
- **Internal consistency:** the reconciliation table (§2) and the architecture (§4) agree —
  nothing built here duplicates a store named "drop" in §2.
- **Scope:** deliberately narrower than the master prompt's §17–28 — five artifacts dropped
  in favour of existing stores, `locks/` deferred, delegate write-back deferred. Sized for one
  implementation plan.
- **Ambiguity check:** the one place two readings were possible — whether `"ready"` is stored
  or derived — is resolved explicitly in §5, with the reasoning stated (avoiding a second
  source of truth), not left for an implementer to choose.
