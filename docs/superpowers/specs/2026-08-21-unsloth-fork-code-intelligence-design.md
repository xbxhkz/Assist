# Unsloth Fork — Sub-project 2: Code Intelligence (LSP) — Design Spec

## Context

Second sub-project of the initiative to fork [Unsloth](https://github.com/unslothai/unsloth/)
and add capabilities from Assist that Unsloth lacks. Sub-project 1 (vision tools) shipped as
[PR #1](https://github.com/xbxhkz/unslothed/pull/1) — five tools in Studio's agent loop, four
upstream files touched, zero regressions across ~27,000 tests.

The originating request named "the ability to edit rust, c++, c#, javascript". Investigation
reframed it, and the reframing is the reason this spec exists in its current form.

**What was asked for already works.** Studio's `edit_file` is language-agnostic and edits `.rs`,
`.cpp`, `.cs`, and `.js` today. `terminal` ("Execute a terminal command and return
stdout/stderr") already runs `cargo build` or `dotnet build`. Taken literally, the request is
satisfied by code that shipped before this initiative began.

**What is actually missing is intelligence.** The model edits those files blind: no type errors
without running a full build, no go-to-definition, no find-references, no signature or doc
lookup, and build output arrives as raw stdout it must parse by eye.

**Neither app has any of it.** Greps for `pygls`, `pylsp`, `lsprotocol`, `LanguageServer`, and
`textDocument/` return nothing in Studio *or* Assist. Assist's `src/agent_tools/` has
`shell_tools.py` and `subprocess_tools.py` but no language intelligence. This is therefore
**net-new work, not a port** — a materially different risk profile from sub-project 1, which was
roughly 80% transcription of proven, already-tested code.

## Goal

Give Studio's agent loop read-only language intelligence for **C#** and **TypeScript/JavaScript**
via the Language Server Protocol: type-aware diagnostics without a build, go-to-definition,
find-references, hover, and workspace symbol search.

## Language scope, and why it is two and not four

| Language | Toolchain on the dev machine | In v1 |
|---|---|---|
| C# | `dotnet 8.0.424` | yes |
| TypeScript / JavaScript | `node v25.9.0`, `npm 11.12.1` | yes |
| Rust | no `cargo`, no `rustc` | no |
| C++ | no `g++`, no `cl`, no `cmake` | no |

Rust and C++ are deferred because nothing here can exercise them. `rust-analyzer` needs a Rust
toolchain (~1.5 GB) for most of its value, and `clangd` needs a `compile_commands.json` produced
by a build system plus an actual compiler (MSVC, several GB). Shipping two languages that cannot
be tested is precisely how this project has five recorded cases of a fully green suite certifying
non-working code.

They are a second pass once the architecture below is proven, and the design is shaped so adding
them is additive: a new entry in the server registry, not a new mechanism.

## The central design problem

**LSP is stateful and long-lived; Studio's tool boundary is stateless and synchronous.**

`execute_tool(name, arguments, ...) -> str` (`tools.py:9996`) returns a plain string per call. LSP
requires a subprocess per workspace, an `initialize`/`initialized` handshake, capability
negotiation, document-sync notifications, and a session that persists across calls.

Starting a server per call is not viable: `typescript-language-server` takes seconds to index a
mid-sized project and `rust-analyzer`/`clangd` take tens of seconds. Every tool call would pay it.

The design therefore commits to a **persistent server pool** keyed by `(language,
workspace_root)`, with idle timeout, a bounded server count, and LRU eviction. That is a real
commitment — subprocesses outliving a request inside a long-running server — and it is the main
thing v1 exists to prove.

## Architecture

Implementations live in a package we own: `studio/backend/core/inference/assist_code/`. Upstream
files are touched in exactly the same two places as sub-project 1, plus the request-filter hunk it
already established.

**Model the pool on `mcp_client.py`, not on first principles.** Studio already runs long-lived
stdio JSON-RPC subprocesses for MCP, and has already paid for the hard parts:

- `_StdioSession` (`mcp_client.py:390`) — persistent stdio session with `connect` / `run` /
  `close`, each owning a thread running its own asyncio loop (`_run_loop`, `:414`)
- `_StdioKeyLock` (`:540`) — per-key locking, which is exactly what a pool keyed by workspace
  root needs
- `_transport_dead` (`:318`) and `_session_responsive` (`:344`) — health checking
- `_SessionWedged` (`:373`) and `_SessionClosed` (`:377`) — the two failure modes a long-lived
  stdio session actually has
- `_abort_future` (`:381`) — cancellation

The MCP *protocol* is not LSP and is not reusable, but this *lifecycle architecture* is directly
transferable, and a naive implementation would rediscover wedged-session handling painfully.

**Process cleanup must not log.** `llama_cpp.py:20925` documents this from experience: by the time
`atexit` runs, the streams log handlers write to may already be closed, and the failure surfaces
as unrelated tracebacks printed after the test summary. Two mechanisms are required together —
`logging.raiseExceptions = False` for stdlib loggers other libraries install, and a bare `except`
for this module's structlog `PrintLogger`, which consults neither. The LSP pool has N subprocesses
with the identical exposure and inherits both.

**Threading is already settled.** Sub-project 1 established by inspection that `execute_tool` runs
on a dedicated daemon worker thread inside `stream_tool_execution` (`tool_stream_exec.py:161-176`),
with the polling loop pushed through `asyncio.to_thread` (`studio_tool_loop.py:1266`). Blocking on
an LSP response therefore stalls no event loop. The pool is shared across those threads and needs
its own locking.

## The subtlety that must be designed for, not discovered

**Diagnostics are pushed, not pulled.** Classic LSP has no "give me this file's errors" request:
the server emits `textDocument/publishDiagnostics` unsolicited after `didOpen`/`didChange`. So
`get_diagnostics` must open the document and then *wait* for a notification that may never arrive,
under a timeout.

LSP 3.17 added pull diagnostics (`textDocument/diagnostic`), but server support is uneven. The
client negotiates during `initialize` and uses pull when the server advertises it, falling back to
push-with-wait otherwise.

This forces the client's shape: a reader thread per session pumping stdout into **two** sinks — a
response map keyed by request id, and a notification queue. A design that only models
request/response cannot express diagnostics at all.

## Addressing: symbols, not coordinates

LSP speaks `line`/`character`. An agent thinks in names and frequently has not read the file.

Every tool accepts **either** a `symbol` name — resolved via workspace symbol search — **or**
explicit `line` and `column`. Symbol-first is the usable path; coordinates are the escape hatch
when a name is ambiguous.

## Tool surface

Five read-only tools:

| Tool | Answers |
|---|---|
| `code_diagnostics` | What is wrong in this file, without running a build |
| `code_definition` | Where is this symbol defined |
| `code_references` | Everywhere this symbol is used |
| `code_hover` | Type, signature, and docs at this position |
| `code_symbols` | Find a symbol by name across the workspace |

**Read-only is deliberate.** The model still makes every change through the existing `edit_file`,
so there is one write path and one set of guards. LSP-powered `rename_symbol` is genuinely more
correct than text editing for cross-file renames and is a plausible v2, but it is a multi-file
mutation on a second write path that bypasses `edit_file`'s guards, and a bad rename is hard to
undo.

**Completions are permanently out of scope.** They are designed for a human typing character by
character; an agent that emits whole edits gets little from them relative to their complexity.

## Server acquisition

Both verified to exist and be installable from this machine, not assumed:

| Language | Server | Verified | Install |
|---|---|---|---|
| TypeScript / JavaScript | `typescript-language-server` (+ `typescript`) | npm, v6.0.0 | `npm install` |
| C# | `csharp-ls` | dotnet tool, v0.26.0 | `dotnet tool install` |

`csharp-ls` over OmniSharp: lighter, actively maintained, and a plain stdio LSP server with no
editor-specific protocol extensions to work around.

Installed on first use, and **disclosed with a log line naming the server and where it comes from
before the install starts**, matching the convention sub-project 1 settled on. A failure names the
server, the command that failed, and the manual install command. No language server is vendored.

## Data flow

Tool call → resolve and confine the path → determine language from file extension → acquire or
create the pooled session for `(language, workspace_root)` → ensure the document is open and
in sync → issue the LSP request or await the diagnostic notification → format the reply as text
naming files by workspace-relative path with 1-based line numbers.

**Confinement is unconditional.** The workspace root and every path must resolve inside the
session workdir, reusing the mechanism sub-project 1 established. Sub-project 1's Task 1 shipped
a version where confinement was gated on `session_id` being present, which made it default-open;
the ruling there was that confinement is unconditional and the tests change. That ruling carries
forward. A language server indexes an entire directory tree, so an unconfined root is a
materially worse leak than a single unconfined file read.

## Error handling

**A tool never raises into the agent loop.** Every failure returns actionable text: server not
installed (naming the install command), server failed to start, server wedged or crashed
(restarted once, then reported), request timed out, language unsupported (naming those that are
supported), path outside the workspace, file not found, no results found.

**No results is not an error.** `code_references` finding nothing, or `code_diagnostics` finding a
clean file, are valid answers and must read as such — the same distinction sub-project 1 drew
between zero detections and no face detected.

## Registration

Two touchpoints in `tools.py` — one `ALL_TOOLS` splice, one delegating branch in `execute_tool` —
matching sub-project 1 exactly.

**And the request filter, which is not optional.** Registering in `ALL_TOOLS` is necessary but
insufficient: Studio's frontend sends an explicit `enabled_tools` allowlist built from a
pill-driven literal naming only upstream tools, and `_select_request_tools`
(`routes/inference.py:3634`) filters `ALL_TOOLS` down to it. Sub-project 1 shipped tools that were
silently unreachable from every Studio chat for exactly this reason while all its registration
tests stayed green, because those tests call `execute_tool` directly — the one path that bypasses
the filter.

This sub-project **extends sub-project 1's existing re-add hunk** rather than adding a second one,
keeping the upstream seam at the same two files and avoiding a merge conflict between the two
branches. It therefore branches from `assist-vision-tools`.

The re-add must stay gated on the filter having admitted something (`if tools_on and tools:`). An
unconditional re-add makes the catalogue never-empty and defeats upstream's guard that an empty
selection skips the tool loop; that regression broke four upstream tests in sub-project 1 and was
caught only after a reviewer retracted its own approval.

## Testing

**A real fake server, not mocks.** Tests run against a small Python subprocess that genuinely
speaks Content-Length-framed JSON-RPC and implements a minimal LSP surface, so framing, the
handshake, capability negotiation, request/response correlation, and push notifications are all
actually exercised. A mock that returns dicts tests none of that, and the framing is where this
kind of client breaks.

**Plus true end-to-end tests** against real `typescript-language-server` on a fixture TypeScript
project, since node is present. These are the only tests that prove the thing works.

Required coverage:

- Diagnostics on a file with a real type error, asserting the message and line
- Definition and references across two files
- The push path: a server that only publishes diagnostics after `didOpen`
- The pull path: a server advertising `textDocument/diagnostic`
- A wedged server: one that accepts a request and never replies, asserting the timeout is
  reported rather than hanging
- A crashed server: restarted once, then reported
- Pool behaviour: a second call reuses the session; eviction closes the process
- Confinement: a workspace root outside the session workdir is refused
- Registration reachability through the **real** `_select_request_tools` with a Studio-shaped
  allowlist, and the empty-selection case staying empty

**Negative controls on the critical paths** — reintroduce the bug, confirm the test fails,
restore. Stated explicitly because a suite that cannot fail has certified non-working code on this
project five times.

## Out of scope

- Rust and C++ (deferred until their toolchains exist; additive by design)
- `rename_symbol` and any other mutating LSP operation
- Completions, permanently
- Structured build/run with parsed compiler output — a plausible sibling sub-project, but
  independent of LSP and not required by it
- Any new UI; these are agent tools on Studio's existing chat surface
- The remaining initiative sub-projects: multi-persona Crew, workflow engine + node editor, and
  optionally Assist's tool-RAG retrieval and privilege gating
