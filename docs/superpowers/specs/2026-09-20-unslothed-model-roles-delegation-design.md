# Unslothed Model Roles and Delegation — Design Spec

## Context

A new sub-project for the fork of [Unsloth](https://github.com/unslothai/unsloth/) that is being
extended into a local AI agent platform. The owner's request:

> Make this dynamically load models based on what I need, so I can set up a model for an agent, one for
> image, one for video, one for coding, one for vision, etc. and then I can load my primary model and if
> it needs to it will call on other models to continue the work required and load back to the main model
> when finished, all while keeping everything in memory or a file it will read from.

Relevant master specification sections: §2/§20 (agent framework), §18 (structured memory — adjacent,
deliberately not this piece).

## What already exists

Investigation reshaped this before design started, as it did for the two discovery pieces.

**Most of the loading machinery is already built.** `utils/openai_auto_switch_settings.py` provides
`openai_api_auto_switch_model` (load the model a request names), `openai_api_auto_download_model`,
`openai_api_auto_unload_idle_seconds` with `openai_api_auto_unload_keep_kv`, the media equivalents
(`media_api_auto_switch_model`, `media_auto_unload_idle_seconds`), and `openai_api_auto_switch_overrides`
— per-model launch settings, read via `get_model_override()` and converted by
`model_override_load_kwargs(override, is_gguf = ...)`.

**Image and video models already load on demand**, through `core/inference/media_auto_switch.py`, and
`core/inference/gpu_arbiter.py` arbitrates a single GPU between `CHAT`, `DIFFUSION` and `VIDEO` lanes,
evicting whichever holds it. On a 6 GB card this is what makes "a model for image, one for video"
already true today.

**Vision-aware switching is partly there.** `_maybe_auto_switch_model(..., require_vision = ...)`
refuses to evict a resident vision model for a text-only target when an image is attached.

**What does not exist** is exactly the owner's three asks: naming *roles* instead of model ids, a model
deciding it needs another model, and work surviving the swap.

**Two findings that make the design possible, both verified rather than assumed:**

1. **Tools execute between model turns, not during generation.** In `studio_tool_loop.py` the tool call
   is a discrete step after the turn's text is complete and its tool calls are parsed. Each round is an
   independent HTTP request carrying the whole message list, so the backend between rounds is
   replaceable. Swapping there costs the primary a re-prefill, not correctness. If generation held a
   long-lived session this feature would have to move to turn boundaries instead.
2. **`run_safetensors_tool_loop` is already dependency-injected** — it takes `single_turn` and
   `execute_tool` as callables plus `max_tool_iterations`, `tool_call_timeout`, `cancel_event`,
   `permission_mode` and `confirm_tool_calls`. A sub-agent can drive it internally rather than streaming
   to a browser.

**Supporting facts:** `stream_tool_execution` already heartbeats "so a long call cannot idle the stream
out", and passes an `output_callback` — a multi-minute delegation neither times out the stream nor has
to be silent. `state/tool_approvals.py::wait_tool_decision` **fails closed**: it returns `"deny"` on
timeout or cancellation and always removes its slot.

## Decisions taken by the owner

| Question | Decision |
|---|---|
| How a hand-off is triggered | **Both** — a tool the model calls, and automatic routing before the turn |
| What the delegate receives / returns | **A brief plus a shared work file**; a short result comes back |
| Scope of this spec | **Roles + delegation (A + B).** Automatic routing (C) is a separate later piece |
| Delegate tool access | **Full tools — a real sub-agent** |

The full-sub-agent choice was made after its cost was stated: a nested agent that can act needs its own
approval handling, cancellation, budget and loop prevention. Those are therefore requirements here, not
options.

## Goal

Let the owner bind a model to a role, and let the loaded model hand a task to another role's model —
brief out, artifact and short result back — then return to the model that was resident before.

## Architecture

### Piece A — Roles

`core/inference/model_roles/`, fork-owned.

| File | Responsibility |
|---|---|
| `__init__.py` | `RoleBinding`, `resolve(role)`, `availability(role)`, `bindings()` |
| `storage.py` | read/write the `model_roles` setting |
| `schemas.py` | the `ask_model` schema + `DELEGATION_TOOLS` / `DELEGATION_TOOL_NAMES` |

A binding, stored under one settings key (`model_roles`) read with the existing `_cached_setting`
pattern:

```python
{
  "coding": {
    "model": "unsloth/Qwen2.5-Coder-14B-GGUF:Q4_K_M",
    "overrides": {"n_ctx": 16384}   # same shape as openai_api_auto_switch_overrides
  }
}
```

`overrides` is passed through the existing `model_override_load_kwargs()` rather than a second
conversion path.

**Role names.** `primary`, `coding`, `vision`, `image`, `video` are defaults offered by the API; any
custom name is accepted. Nothing behavioural is hard-coded to the five.

**Availability reuses the readiness vocabulary** already shipped — `ready` (bound, downloaded,
loadable), `missing` (bound, not downloaded), `unknown` (unbound, or not cheaply checkable). This is
deliberate: a later piece can make `find_capability` list models as providers beside tools instead of
maintaining a second parallel vocabulary.

**Image and video roles are informational in v1.** Media models already auto-switch from the request;
binding those roles records the preference and feeds defaults. Adding a second swap path for media
would create two writers racing the same arbiter lane.

**Configuration surface:** an auth-gated fork-owned router at `/api/model-roles` (GET, PUT), following
`routes/tool_audit.py`. **No new UI in this piece** — that follows once the behaviour has proven itself.

### Piece B — Delegation

`core/inference/delegation/`, fork-owned.

| File | Responsibility |
|---|---|
| `__init__.py` | `execute(name, arguments)` — the `ask_model` handler; orchestration |
| `workspace.py` | the delegation folder, brief, work file, transcript |
| `subagent.py` | the sub-agent loop and its budgets |

`ask_model(role, task)` runs this sequence:

1. **Resolve and check.** Role → binding → availability. `missing` or `unknown` returns a plain error
   naming what is absent. **No swap is attempted on a failed check.**
2. **Create the delegation folder and write `brief.md`.**
3. **Record the resident model** — what is loaded *now*, not what the `primary` role names.
4. **Swap**, through the same loader auto-switch uses, in the arbiter's `CHAT` lane.
5. **Run the sub-agent**, injecting the audited `execute_tool`.
6. **Write `work.md` and `transcript.md`.**
7. **Restore the model from step 3**, in a `finally`.
8. **Return** a short capped summary plus the file paths.

**Step 3 is not a detail.** Restoring "the `primary` role" rather than "what was resident" would
silently change which model the user is talking to whenever they had loaded something by hand. A
delegation must be invisible to the conversation except for its result.

**The loop.** A safetensors delegate drives `run_safetensors_tool_loop` directly. A GGUF delegate
returns structured tool calls rather than markup, so it needs a small fork-owned loop posting to the
loaded backend's OpenAI-compatible endpoint — same injected executor, same budgets. Both paths take the
loader as an injected callable so tests never load a model.

## State

Everything lives under `_get_workdir(session_id)` — this chat's workspace, falling back to the session
sandbox as the other file tools do:

```
delegations/<id>/brief.md        role, goal, constraints, what the primary established, what "done" means
delegations/<id>/work.md         the delegate's artifact: the code, the findings, the description
delegations/<id>/transcript.md   the full sub-conversation
```

**The location is forced, not chosen.** `assist_vision/__init__.py::_write_png` records the rule: tools
return a path, never inline data, and `resolve_image_bytes` refuses any path outside the conversation's
working directory. A delegation folder written elsewhere could not be read back by the primary's own
file tools.

**What returns to the primary** is a capped summary plus the paths. Capped deliberately: a delegate that
returned 40 KB of code into the primary's context would defeat the purpose of writing a file.

This is what satisfies "keeping everything in memory or a file it will read from" — the thread persists
in the database, the artifacts persist on disk, and both survive the swap. A structured memory store is
§18's own piece and is not this one.

## Safety

The full-sub-agent choice makes each of these load-bearing.

**Approvals — inherit, never exceed.** The delegate runs under the primary's permission mode and the
same approval handshake, with the **real session id** so prompts surface in the UI as they do today. It
never receives a bypass the primary did not have. `wait_tool_decision` fails closed, so an unanswered
prompt ends as a denial rather than a hang.

**Loop prevention, two layers.** `ask_model` is removed from the delegate's tool list, *and* a depth
guard rejects a nested delegation while one is running. The first is data someone can edit; the second
is behaviour.

**Budgets, with concrete v1 values** so "a limit" is not left to interpretation:

| Budget | Value | Why |
|---|---|---|
| delegate tool iterations | **12** (the normal loop defaults to 25) | a delegate works on one brief, not a whole conversation |
| delegate wall clock | **600 s** | longer than a large model's load plus real work, shorter than a hung turn |
| delegations per assistant turn | **2** | each costs two swaps; this is the thrash ceiling |

Exceeding a budget returns a refusal string, never an exception.

**Cancellation.** The `cancel_event` already threaded through `execute_tool` is passed into the
sub-loop, so stopping generation stops the delegate.

**Swap-back is guaranteed, and a failed restore is loud.** The `finally` restores on every path. If the
restore itself fails, the returned text says so plainly. This is the one place where "never raises" must
not become "quietly lies": otherwise the next turn runs against a different model than the user
believes, with nothing to indicate it.

**Audit attribution.** A nullable `delegation_id` column on the fork's own `tool_audit` table, set by a
context variable while the delegate runs. Delegate actions stay distinguishable without touching session
identity, which approvals depend on.

That table already exists in installed databases, so this needs an additive migration —
`ALTER TABLE ... ADD COLUMN` guarded by a column check, the pattern `storage/rag_db.py` already uses for
`linked_folders`. A nullable column keeps every existing row valid, and rows written by the primary
simply carry no delegation id.

## Registration

`ask_model` is a built-in tool and needs the project's five registration points: `ALL_TOOLS`, the
`execute_tool` dispatch, the `_addable` re-add in `routes/inference.py`, and — because it is **not**
read-only — it is deliberately **absent** from `_ALWAYS_SAFE_TOOLS` and
`_ANTHROPIC_UNPROMPTED_SAFE_TOOLS`. A tool that loads models, runs a sub-agent and can edit files must
go through the normal risk classification.

Seam budget: about 5 additive lines in `tools.py` (93 → ~98) and 1 in `routes/inference.py` (59 → ~60),
zero deletions, amending fork-inserted lines only.

## Error handling

Every delegation path is guarded so a failure returns text rather than breaking the turn, with the
single exception that a failed restore must be reported in that text. Budgets, unavailable roles, loader
failures and cancellation all produce plain strings the model can act on.

## Testing

**No test loads a model.** The loader is injected; tests supply a fake recording `load(x)` / `restore(y)`.

| Test | Control that proves it is live |
|---|---|
| role resolves to its binding; three states | an unbound role reports `unknown`, not `ready` |
| a `missing` role returns an error and **never calls the loader** | the fake loader records nothing |
| files land under `_get_workdir(session_id)` | a path outside it fails |
| **restore targets what was resident, not the `primary` role** | bind `primary` to a different model; restore still names the resident one |
| restore happens on error and on cancellation | raise inside the sub-loop; restore still recorded |
| a failed restore is reported in the returned text | make the fake restore fail; assert the text says so |
| `ask_model` absent from the delegate's tool list | present it and the depth guard still refuses |
| depth guard refuses a nested delegation | remove it and the nested call proceeds |
| budget exhaustion returns a refusal | raise the cap and it proceeds |
| `cancel_event` stops the sub-loop | |
| audit rows carry `delegation_id` | primary-run calls carry none |
| `ask_model` reaches the model through the **real** `_select_request_tools` | remove it from `_addable` and this fails |

Only named test files are run; **never the full backend suite** (upstream fixtures fabricate GGUF files
up to 40 GB and filled the disk once).

**What automated tests cannot cover:** a real swap, with real VRAM, a real re-prefill and real elapsed
time. That needs one manual run on the owner's machine, and this spec records it as owed rather than
implying the suite proves it.

## Risks

**Swap cost dominates the experience.** Two swaps per delegation, each a model load plus a re-prefill of
the primary's context. On a 6 GB card with a large model this is tens of seconds at best. The
per-turn delegation limit exists because of this, and the manual run is what will tell us whether the
feature is pleasant or merely correct.

**The primary's KV cache is destroyed by the swap.** `openai_api_auto_unload_keep_kv` suggests a stash
mechanism exists; v1 does not depend on it, and any benefit is a bonus rather than a requirement.

**A sub-agent can act.** It edits files and runs commands under the same permissions as the primary.
The approval path, budgets, depth guard and audit attribution are what keep that accountable; none is
optional.

**Two writers of the CHAT lane.** Delegation and the existing auto-switch both load chat models. They
are serialised by the arbiter and the existing auto-switch locks, but this is the first time the fork
initiates a load from inside a tool call.

## Out of scope

- **Automatic routing before the turn (piece C)** — the second trigger the owner chose, deliberately
  deferred so the two paths are not designed before either has run
- Any UI for configuring roles
- A structured memory store (§18)
- Nested delegation beyond depth 1
- Parallel residency of two chat models (the hardware does not allow it)
- Changing how media models load
