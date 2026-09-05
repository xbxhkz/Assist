# Unslothed Draft-Model Chooser — Design Spec

## Context

Fourth sub-project of the initiative to fork [Unsloth](https://github.com/unslothai/unsloth/) and
extend it into a local AI agent platform. Sub-projects 1 (vision tools) and 2 (code intelligence)
are merged and open as [PR #1](https://github.com/xbxhkz/unslothed/pull/1); sub-project 3
(packaging: branding, Windows installer, Docker) is complete.

The owner's request was to "choose the model for speculative decoding, and to toggle it on and off
if it is always on", across two surfaces: per-chat and saved-per-model.

## What the request assumed, and what is actually there

Investigation changed the scope before any design work started. Most of the request already
exists.

| Capability | Status | Where |
|---|---|---|
| On/off toggle | **Exists** | `off` is a canonical mode in `lib/speculative-modes.ts` and `_CANONICAL_SPEC_MODES` |
| Per-chat surface | **Exists** | `chat-settings-sheet.tsx`, persisted under `unsloth_chat_speculative_type` |
| Saved-per-model surface | **Exists** | `saved-model-settings.tsx` + `routes/settings.py:717` |
| Mode choice | **Exists** | `auto`, `mtp`, `dspark`, `dflash`, `ngram`, `mtp+ngram`, `off` |
| Draft depth (`spec_draft_n_max`) | **Exists** | gated to `DRAFT_N_MAX_SPEC_TYPES` |
| **Choosing a specific draft model** | **MISSING** | only via raw `llama_extra_args` |

So there is an off switch. The likely reason speculation feels "always on" is a deliberate
decision in `chat-runtime-store.ts:139-143`:

```js
// Persist only the model-agnostic intents (auto/ngram/off). The model-specific
// drafter modes (mtp/mtp+ngram/dspark/dflash) and spec_draft_n_max stay session-only:
// a persisted choice would silently no-op on a model with no MTP head or no
// DSpark sidecar. Unknown -> auto.
const PERSISTED_SPEC_MODES = new Set(["auto", "ngram", "off"]);
```

A drafter mode chosen per-chat reverts to `auto` next session, and `auto` engages speculation
whenever it finds a sidecar. The owner reviewed this and scoped the sub-project to the
draft-model chooser alone; `PERSISTED_SPEC_MODES` is deliberately left as-is.

## Goal

Let the user choose which model acts as the speculative drafter — a local GGUF, a Hugging Face
repo, or an explicitly named auto-discovered sidecar — at both the per-chat and saved-per-model
surfaces, without meaningfully expanding the upstream merge seam.

## The constraint that governs the approach

The fork's entire upstream seam is **8 lines in `core/inference/tools.py` (3 hunks)** plus **one
hunk in `routes/inference.py`**, validated against 69 commits of upstream drift. Every design
decision below is subordinate to keeping it that small.

### Why not a first-class field

The clean design in a non-fork codebase is a typed `speculative_draft_model` threaded through the
stack. Here it would touch `models/inference.py`, `routes/settings.py`, `routes/inference.py`,
`core/inference/llama_cpp.py`, `utils/openai_auto_switch_settings.py` and both UIs — roughly ten
hunks across five upstream files, converting a maintainable fork into a permanent hard one.
Rejected on that basis, not on merit.

### What makes the cheap approach possible

`llama_extra_args` is already:

- **persisted per-model** (`routes/settings.py:710`, carried through saved overrides and
  auto-switch entries in `utils/openai_auto_switch_settings.py:496-500`)
- **plumbed into both frontend surfaces** (`saved-model-settings.tsx`, and chat via
  `chat-adapter.ts` / `shared-composer.tsx` / `use-chat-model-runtime.ts`)
- **fully honoured by the load path** for a caller-named drafter: VRAM is charged
  (`_extras_bytes`, and `_remote_drafter_repo_bytes` for HF repos), last-wins leaves exactly one
  `--model-draft` resident, and the native path rules at `routes/inference.py:4696` apply to the
  launch path *and* the sibling shards of a split drafter.

The feature therefore reduces to writing the right values into a field the backend already
understands. Both surfaces come free.

## Architecture

One new backend module, `studio/backend/routes/draft_model.py`. Frontend changes are confined to
frontend files.

**Upstream seam cost: one router-registration line.**

### Composition lives in the backend, not the frontend

The tempting shape is for the UI to build `--model-draft <path>` itself. That would mirror the
backend's flag vocabulary — `-md`, `--model-draft`, `-hfd`, `--spec-draft-hf`, each in both `-f v`
and `-f=v` spellings — into TypeScript, while `_extra_args_mtp_draft_path` already parses exactly
that set. Two parsers that must agree and have no test binding them together is the shape of the
bare-`mcp` failure this project has already paid for once.

So the frontend never constructs a flag. It sends a choice and receives a finished
`llama_extra_args` array.

## Endpoints

### `GET /api/draft-model/candidates?model_id=…`

Returns the selectable set:

1. **Auto-discovered sidecars, named explicitly.** Today `auto` silently resolves a colocated
   MTP/DSpark/DFlash sidecar through `drafters/preference.py`'s ranking. Listing them by name makes
   "what is auto actually using?" answerable, which is a real gain independent of the chooser.

   Selecting one **pins** it rather than being a no-op, even when it is the file auto would have
   chosen. The two differ whenever the ranking's inputs change — a newly downloaded higher-precision
   sidecar outranks the current pick, and `dflash_repo_preference_key` re-ranks against the weight
   being loaded — so "the file auto picks today" and "this file" are different requests. Pinning is
   also what makes the choice reproducible across machines, which an implicit ranking is not.
2. **Local GGUFs** from the Studio library.
3. The **Hugging Face repo** free-text option.

### `POST /api/draft-model/select`

Body `{model_id, choice}`; validates, and on success returns the composed `llama_extra_args`: the
caller's existing args with any drafter-naming flags removed and the new one appended, **all other
arguments preserved in order**. Clearing the choice is the same call with an empty choice, which
removes the drafter flags and restores auto-discovery.

## Validation

Nothing in the backend compares a drafter against its target today. Auto-discovery is safe *by
construction* — it only selects colocated, name-matched sidecars — and a free-form chooser removes
that protection. Four checks:

| Check | Failure it catches |
|---|---|
| Existence / reachability | a typo'd path, a repo that does not resolve |
| **Vocabulary match** (`n_vocab`, target vs drafter) | the real compatibility gate; a mismatched drafter otherwise fails inside llama-server |
| Path confinement | a drafter outside the permitted directory, including split-shard symlinks |
| Size | a choice that will 409 on the VRAM budget, reported before the load rather than during it |

`_vocab_size` is already parsed from GGUF headers (`llama_cpp.py:10283`), and the codebase already
performs remote GGUF header range-reads, so the vocabulary check works for both local files and HF
repos. For a remote repo it is best-effort: when the header cannot be read the result is an
explicit "could not verify" state shown to the user, **never a silent pass**.

## Surfaces

Both already carry `llama_extra_args`, so both inherit persistence:

- **Per-chat** — chat settings sheet, alongside the existing speculative-type control.
- **Saved-per-model** — `saved-model-settings.tsx`.

The control is shown only for modes that launch a separate drafter (`auto`, `mtp`, `mtp+ngram`,
`dspark`, `dflash`) and hidden for `ngram` and `off`, which launch none.

## Error handling

Selection-time failures block the choice with a stated reason. The recurring lesson of this
project is that silent degradation is the expensive failure mode — `sqlite-vec` disabling RAG,
`update_flow.py` failing open, a drafter that is "not on disk, so no drafter loads" — so a
rejected drafter says why.

### One fail-open this design does not close, stated plainly

If a chosen drafter later disappears from disk, the existing load path treats it as *"a local
`--model-draft` that is not on disk, so no drafter loads and none is charged"* — silently, with
speculation quietly reverting to none. Validation at selection time cannot prevent that at load
time.

Mitigation: re-validate when a settings surface opens and mark a missing drafter there. The
silent load-time fallback itself remains, because closing it means editing the upstream load path
— the seam trade this design explicitly declines. Recorded as a known limitation rather than
hidden.

## Testing

TDD throughout, with a negative control for every check. This project has produced six negative
controls that turned out inert, so each control must be demonstrated to fail before the fix and
pass after:

- **Vocabulary check** — a mismatched pair must fail *that* check while passing existence,
  confinement and size, so the test binds to the gate it claims to test.
- **Composition** — hand-written arguments survive untouched, in order, while only drafter flags
  are replaced. Controls: an arg that merely *contains* `-md` as a substring must not be stripped;
  both `-f v` and `-f=v` spellings must be recognised.
- **Confinement** — a split drafter whose second shard symlinks outside the permitted directory
  must be rejected, not merely its launch path checked.
- **Clearing** — an empty choice restores auto-discovery, verified by the resulting args
  containing no drafter flag.

## Risks

**The composed flag is visible in the raw-args box.** A user who has hand-written
`llama_extra_args` will see a UI-managed flag appear among their arguments. Mitigated by
surgical replacement and by the chooser reflecting whatever is already there, but the two controls
do share one field and that is inherent to the approach.

**Vocabulary equality is necessary, not sufficient.** Two models can share a vocabulary size and
still be poor speculative partners. The check catches the common catastrophic case, not every bad
pairing.

## Out of scope

- Changing `PERSISTED_SPEC_MODES` (the owner scoped this out)
- Any change to the existing on/off toggle or mode list
- Closing the load-time missing-drafter fail-open
- Automatic drafter *recommendation* — this chooses, it does not rank
