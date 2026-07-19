# Industrial Vision Diagnostics — Design

**Goal:** A `diagnose_equipment` builtin tool that routes an industrial image
(schematic, HMI/fault display, drive, component, thermal) to the shipped vision
model with a maintenance-expert prompt and returns a safety-first structured
diagnosis — a distinctive maintenance assistant built on foundations the app
already has.

**Scope:** v1 of the "Industrial Assistant" (the vision-diagnostics slice). One
new tool, four task modes, reusing the existing vision inference path. Non-goals:
live equipment data (OPC UA / Modbus), predictive maintenance, an equipment-manual
RAG, and a dedicated UI panel — each a separate follow-on sub-project.

---

## Background — what this builds on

- **Vision already works.** The app serves a vision-capable model (Qwen2.5-VL) and
  can send an image (base64 / image content) to it — this is how `capture_screen`
  and attached-image chat already "see." The inference path lives in the
  chat/model layer (`src/llm_core.py` / `src/chat_handler.py` /
  `src/agent_tools/desktop_tools.py`). This tool **reuses that path**, adding a
  maintenance-expert prompt and a task selector — no new model, no new capture code.
- **Builtin tools** are the app's capability pattern: a `SomeTool().execute(content,
  ctx)` async handler registered at ~7 surfaces (`TOOL_HANDLERS`/`TOOL_TAGS`,
  `FUNCTION_TOOL_SCHEMAS`, `tool_security`, `BUILTIN_TOOL_DESCRIPTIONS`,
  `agent_loop.TOOL_SECTIONS`/`_DOMAIN_TOOL_MAP`, a parity test).
- **No feasibility gate needed** — unlike OPC UA (which needs a device to test) or
  the rejected EXL2/ControlNet spikes, the vision mechanism is shipped and proven.
  Only the *domain answer quality* is unproven, and that is a manual evaluation.

## The `diagnose_equipment` tool

Handler `DiagnoseEquipmentTool().execute(content, ctx)`, delegating to a module-level
`diagnose_equipment(content, ctx, *, vision_call=None)`. `content` is a JSON string of
the args; `ctx` carries `owner`. The `vision_call` is **injectable** (default = the
app's real vision inference) so tests never hit a real model.

Flow: the agent shows/points at an industrial image → calls `diagnose_equipment`
with an image path + task mode → the tool validates the image, composes the task's
expert prompt (+ optional context), calls the vision model, and returns the model's
structured diagnosis text under `{"output": …}`.

## Inputs

- **`image`** (required) — a file path to the image (an attachment the app stored, a
  `capture_screen` screenshot, or a file). Validated before any model call: exists,
  is a readable image, under a size cap. Failure → `{"error": …}` (never raises).
- **`task`** (optional, default `auto`) — one of `schematic | fault_code | vfd |
  component | auto`. Selects the expert prompt; `auto` asks the model to classify the
  image type first, then diagnose.
- **`context`** (optional) — free text (symptom, equipment model/tag, what was already
  checked) folded into the prompt so the diagnosis is grounded in the situation.

## Task modes (expert prompts)

Same mechanism, different expert framing:
- **schematic** — controls/electrical expert: trace circuits, identify symbols/
  components, locate the section relevant to `context`, note wire numbers/terminals.
- **fault_code** — read the HMI/panel/drive display text, decode the fault/alarm,
  explain meaning + causes.
- **vfd** — drive specialist: map to common fault families (overcurrent/overvoltage/
  undervoltage/overheat/ground-fault/overload), give the parameter/measurement check
  sequence.
- **component** — identify the part (type, function, legible manufacturer/part #);
  for a thermal image, interpret hotspots and temperature deltas and flag the abnormal
  component.
- **auto** — classify the image type, then apply the matching expert framing.

## Structured output

One consistent, scannable structure the prompt requests across all modes:
- **observed** — what is visible (display text/reading, circuit, part, thermal pattern);
- **assessment** — decoded fault / schematic interpretation / component ID / hotspot finding;
- **likely_causes** — ranked, most-probable first;
- **recommended_steps** — ordered check/troubleshoot sequence;
- **safety** — hazard-specific warnings;
- **confidence + caveats** — how sure, and what to verify against the real equipment/manual.

The tool returns the model's response **text as-is** under `{"output": …}`. The prompt
drives the structure (labeled sections); the tool does **not** brittle-parse strict JSON
(vision models are unreliable at it).

## Safety (a first-class requirement, not boilerplate)

This is energized, rotating, stored-energy equipment where bad advice injures people.
- Every result carries a **safety** section and a standing disclaimer: *decision-support,
  not a substitute for a qualified person — verify de-energized, follow LOTO and site
  procedures.*
- The prompts are explicitly constrained to **never** advise bypassing interlocks/safety
  devices, defeating guards, or working energized; checks are framed "with the equipment
  safely isolated…" where applicable.
- When a live hazard is involved, the output **leads with safety**.

## Model routing & error handling

- **Vision-capable model only** — reuse the same resolution the existing `capture_screen`
  vision path uses. No vision model served/configured → `{"error": "no vision model
  available — serve a vision model such as Qwen2.5-VL"}`.
- **Never raises into the agent** — a missing/unreadable/non-image path, an oversized
  image, no vision model, or a failed vision call each return `{"error": …}`.
- Image validation (exists / is image / size cap) runs before any model call.

## Gating

`diagnose_equipment` is a read/analyze capability (it sees an image the agent could
already see via `capture_screen`; it runs no shell and no arbitrary tools). Its gating
**matches the existing vision-tools family** (`capture_screen` / `webcam_look`) exactly —
the plan will read those and mirror them rather than invent a new policy.

## Testing

- The **`vision_call` is injectable/mockable** — no real model or inference runs. Tests
  cover: the correct expert prompt is composed per mode; `context` is folded in; **the
  mandatory safety constraint text appears in every mode's prompt**; image validation
  (missing / non-image / oversized → error, no raise); no-vision-model → error; the mocked
  model output is returned under `{"output"}`; and the handler never raises.
- **Registration parity** — the tool is wired at every builtin-tool surface, asserted by a
  parity test (like `run_workflow`).
- **Answer quality is a manual check** — the domain expert evaluates real diagnoses against
  real schematics/faults; the automated tests prove the plumbing + the safety framing.

## Non-goals (this sub-project)

- Live equipment data (OPC UA / Modbus), predictive maintenance / alarm trends.
- An equipment-manual RAG grounding (a separate "knowledge base" sub-project).
- A dedicated upload-and-diagnose UI panel (the agent-tool path is v1; the image comes
  from an attachment / `capture_screen`).
- Strict-JSON structured parsing; multi-image comparison; report generation.
