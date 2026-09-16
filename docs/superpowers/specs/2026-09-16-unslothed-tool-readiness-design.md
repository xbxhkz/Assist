# Unslothed Tool Readiness — Design Spec

## Context

Sub-project 6, piece 3 of 3, of the initiative to fork
[Unsloth](https://github.com/unslothai/unsloth/) and extend it into a local AI agent platform.

The master specification asks for three tool-governance capabilities: an audit log (§27), a
permission model (§25/§26) and discovery (§37/§38). The owner chose all three, audit first. The
audit log merged on 2026-09-15.

**Discovery was then reordered ahead of governance, deliberately.** Governance was placed second on
the grounds that the audit log would let risk levels be assigned from observed traffic rather than
intuition. That evidence does not exist yet: the audit log has never run in the owner's installed
build, and `tool_audit` does not exist in `~/.unsloth/studio/studio.db`. Discovery needs no traffic,
so it goes now and governance goes last, by which time there is data to classify from.

## What already exists, and what is actually missing

Investigation reshaped this piece before design started, as it did for the audit log.

**§37's `list_tools()` would be largely redundant.** Every tool's complete schema — name,
description, parameters — is already sent to the model on every turn (`routes/inference.py:3638`
passes `ALL_TOOLS`; `:3635` filters only when a caller supplies `enabled_tools`). MCP tools are
merged in the same way. The model is already looking at the catalogue; a tool that hands it the
catalogue again answers a question it does not have.

**What genuinely does not exist is whether a tool will work.** There is no readiness concept in the
codebase — one private `_ensure_models_available()` in `assist_vision/face_swap.py` and nothing
else. Yet several tools depend on things that may be absent:

| Tool | Depends on |
|---|---|
| `webcam_look` | the yolov8n weight, downloaded on first use |
| `face_swap` | InsightFace models, behind an explicit licence gate |
| the five code tools | a language server on PATH (`csharp-ls`, `typescript-language-server`) |
| `search_knowledge_base` | an indexed knowledge base |
| any MCP tool | its server being connected |

Today the model finds all of this out by calling the tool and failing.

The master spec's own §38 example is readiness-shaped rather than list-shaped — it ends
`Status: Installed` — and §4 automatic tool acquisition cannot work without it: you cannot acquire
what you cannot detect as missing.

**The probes largely exist already, scattered.** `assist_code/servers.py:67` already does
`shutil.which(name)`; `assist_vision/yolo.py` exposes `weights_path()`; `face_swap.py:72` gates the
licence. Probes delegate to these rather than reimplementing them — reimplementation is how the
audit log's redaction layer drifted from the upstream classifier it was supposed to reuse.

## Decisions taken by the owner

| Question | Decision |
|---|---|
| What discovery provides | **Both** readiness and a capability map — split into 3a (readiness, this spec) and 3b (capability map) |
| How thorough a probe may be | **Cheap static checks, cached.** No process spawning, no network calls |
| How the model learns a tool is not ready | **A tool it can call, plus enriched failures** |
| Where the registry lives | Fork-owned, probes delegating to existing helpers (approach A) |

## Goal

Let the model ask whether a tool will actually work before calling it, and tell it what was missing
when a call fails for a missing dependency.

## Architecture

### The registry

`core/inference/tool_readiness/` owns a `tool name → probe()` map. A probe returns:

```python
@dataclass(frozen = True)
class Readiness:
    state:   Literal["ready", "missing", "unknown"]
    detail:  str                # "yolov8n.pt present" / "csharp-ls not on PATH"
    missing: str | None         # the absent thing, named -- what §4 would acquire
    remedy:  str | None         # how to obtain it, when known
```

### Three states, not a boolean

A tool with no probe — every MCP tool, anything added later — reports **`unknown`**, never `ready`.
Defaulting unknown to ready is the optimistic lie that makes this feature worse than not having it:
the model would read "nobody checked" as "verified working". `unknown` is honest and still
actionable.

### Caching

A 60-second TTL keyed by tool name, so a model asking repeatedly within one turn costs a single
filesystem stat. Explicit refresh available.

### Readiness is not always one boolean per tool

The five code tools take a `language` argument, so `code_definition` can be ready for TypeScript and
not for C# at the same instant. `state` answers **"usable for at least one input"**, and `detail`
carries the breakdown (`"typescript ✓, csharp ✗ (csharp-ls not on PATH)"`).

Parameterised probes keyed by tool *and argument* would be more precise and considerably more
machinery. Rejected for v1: nothing consumes that precision yet, and 3b (the capability map) is the
piece that would reveal whether it is needed.

## The tool

`check_tool_readiness`, with an optional `tool` argument — one tool, or all. It returns text,
because `execute_tool` returns text:

```
webcam_look         ready     yolov8n.pt present
face_swap           missing   InsightFace models not downloaded
                              → accept the licence in Settings → Vision
code_definition     ready     typescript ✓, csharp ✗ (csharp-ls not on PATH)
mcp__fs__read_file  unknown   no readiness check defined
```

Every `missing` names the absent thing and the remedy — the record §4 would later act on.

### Registration, and keeping the seam additive

Registration follows the precedent the fork already set: one line `*READINESS_TOOLS,` in `ALL_TOOLS`
(`tools.py:9837`, beside the existing `*ASSIST_VISION_TOOLS`) and a dispatch block inside
`execute_tool` mirroring `tools.py:10066-10073`. About six inserted lines, taking the seam from
**77 to ~83 insertions, still zero deletions**.

### The approval trap, and the additive way out

A read-only readiness query must not trigger an approval prompt, so the tool belongs in
`_ALWAYS_SAFE_TOOLS` (`tools.py:4535`). Unknown tools fail closed, so without this the model would
prompt the user every single time it asked what was available.

But that is a frozenset literal: adding an element edits the line, which is a **deletion** — exactly
what the additive rule exists to prevent. The additive form rebinds after the original, the same
shape as the audit log's `execute_tool` shadow:

```python
# fork: additive, appended after the upstream definition
_ALWAYS_SAFE_TOOLS = _ALWAYS_SAFE_TOOLS | frozenset({"check_tool_readiness"})
```

**Verified empirically before adopting**, not assumed. All three readers (`tools.py:4543`, `:4581`,
`:6571`) look the name up as a module global at call time, and a live check confirmed the effect:

```
before rebind, is_high_risk_tool_call(name, {}) -> True
after  rebind                                    -> False
```

This verification is recorded because assuming precisely this kind of binding behaviour — without
running it — is what produced the inert control in the audit log sub-project.

## Enriched failures

When a call fails **and** that tool's probe reports `missing`, one line is appended. The original
text is never altered:

```
Error: InsightFace analyzer unavailable

[readiness] face_swap is missing: InsightFace models not downloaded
            → accept the licence in Settings → Vision
```

Enrichment attaches in the audit log's existing `around()` wrapper, which already intercepts every
result — so it needs no new interception machinery.

### Deliberate limitation: returned error strings only, not raised exceptions

This codebase returns failures as strings (`return f"Error: MCP server ... not found"`), so that
path covers most of them. Enriching a raised *exception* would mean raising a different one,
changing its identity on a path that currently works — the same class of mistake as the audit log's
`str(result)` evaluated outside its guard. The audit log already records the exception in full.

Failure is detected conservatively: an exception occurred, or the result begins with `Error:`, which
is the form `execute_tool`'s own error returns already use.

## Error handling

Every readiness path is guarded with a **bare `except BaseException`**, matching the convention this
project arrived at: `OverflowError` and unhashable types slip past `except Exception` in practice. A
probe that raises reports `unknown` carrying the error text.

A readiness failure must never break a tool call, and must never break the readiness query itself.

## Testing

Every control is demonstrated to fail before its fix. Nine inert controls have appeared in this
project, one of them written by the controller in the previous piece, so "the test passes" is not
evidence.

| Test | Control that proves it is live |
|---|---|
| a probe reports `missing` when its dependency is absent | with the dependency present it reports `ready` |
| an unregistered tool reports `unknown` | a registered tool does **not** report `unknown` |
| enrichment appends on failure + `missing` | **a successful call is not enriched**; a failure whose probe says `ready` is not enriched |
| a raising probe does not break the tool call | narrow the guard and the test fails |
| `is_high_risk_tool_call("check_tool_readiness", {})` is False | the rebind's behavioural control |
| the seam stays additive | zero deletions against upstream |

The fifth matters most. The rebind depends on a global being read at call time; if that were wrong
the rebind would silently do nothing, every readiness query would prompt for approval, and no other
test would notice. It therefore asserts the **behaviour**, not the rebind's presence.

## Risks

**The seam grows again.** `tools.py` goes from 77 to ~83 insertions. Still zero deletions, but the
fork's footprint in the most contended upstream file keeps rising; each piece should justify its
lines.

**Enrichment changes model-visible output.** Small and additive, but it is a behaviour change on a
working path. Scoped to failures with a concrete missing dependency, and never rewriting the
original text.

**Cheap checks can say `ready` about something broken.** A present-but-corrupt weight file passes.
This narrows the failure surface rather than eliminating it — the accepted cost of probes cheap
enough to run on demand.

**Staleness.** A 60-second cache can report `ready` for a dependency deleted moments ago. Acceptable:
the call then fails as it does today, and the next query corrects itself.

## What deliberately does not change

No tool's behaviour, permissions or approval requirements change. The existing permission modes,
approval handshake and risk classification are untouched — except for the single additive rebind
that marks the new read-only tool safe.

## Out of scope

- The capability map — capability → candidate tools → recommendation (piece 3b)
- Risk levels and permission modes (piece 2, governance)
- Actually acquiring anything that is missing (§4); this spec only makes absence *detectable*
- Live probes that start language servers or ping MCP servers
- Parameterised per-argument readiness
- Any UI; this piece is model-facing, and the audit panel already shows what ran
