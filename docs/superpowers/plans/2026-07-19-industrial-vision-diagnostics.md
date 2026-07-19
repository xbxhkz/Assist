# Industrial Vision Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `diagnose_equipment` builtin tool that routes an industrial image to the shipped vision model with a maintenance-expert, safety-first prompt (four task modes) and returns a structured diagnosis.

**Architecture:** Reuse the existing vision inference (`analyze_image_with_vl_result`) — extended once to accept a custom prompt — behind a new thin tool handler (`src/agent_tools/industrial_tools.py`). No new model, no new capture code. The tool composes a per-mode expert prompt (with mandatory safety framing), validates the image, calls the injectable vision function, and returns the model's text. Then register it at every builtin-tool surface, matching the `capture_screen`/`webcam_look` gating.

**Tech Stack:** Python. Reuses `src/document_processor.py` (vision inference) and the agent-tool registration surfaces.

## Global Constraints

- One tool, `diagnose_equipment`. Handler `DiagnoseEquipmentTool().execute(content: str, ctx: dict) -> dict`, delegating to module-level `diagnose_equipment(content, ctx, *, vision_call=None)`. `content` is a JSON string of args `{"image", "task"?, "context"?}`; `ctx` carries `owner`.
- **Task modes:** `schematic | fault_code | vfd | component | auto` (default `auto`). Each selects a curated maintenance-expert prompt.
- **Safety is mandatory in EVERY mode's prompt** — a standing "decision-support, not a substitute for a qualified person; verify de-energized, follow LOTO/site procedures" disclaimer, and an explicit constraint to NEVER advise bypassing interlocks/guards or working energized. Every prompt requests a structured result (observed / assessment / likely_causes / recommended_steps / safety / confidence).
- **Never raises into the agent** — a missing/unreadable/non-image path, an oversized image, no vision model, or a failed vision call each return `{"error": …}`.
- **`vision_call` is injectable** (default = the app's real vision inference); tests mock it — **no real model or inference runs in any test**.
- **Reuse, don't reinvent:** the default vision path is `analyze_image_with_vl_result(image_path, owner, *, prompt=<expert prompt>)` — the existing function (model resolution + base64 + vision fallback chain), extended once with an optional `prompt`.
- **Gating matches the vision-tools family** (`capture_screen`/`webcam_look`): in `NON_ADMIN_BLOCKED_TOOLS` (admin-only) AND `PLAN_MODE_READONLY_TOOLS` (read-only, plan-safe); domain `_DOMAIN_TOOL_MAP["desktop"]`.
- **Answer quality is a manual check** (the domain expert evaluates real diagnoses); the automated tests prove the plumbing + the safety framing.
- pytest `--import-mode=importlib`. Stage specific files (never `git add -A`). Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Commit directly to `dev`.

---

### Task 1: Extend the vision inference to accept a custom prompt

**Files:**
- Modify: `src/document_processor.py` (`analyze_image_with_vl_result` — add an optional `prompt` param; ~lines 341, 366)
- Test: `tests/test_vl_custom_prompt.py`

**Interfaces:**
- Produces: `analyze_image_with_vl_result(image_path, owner=None, *, prompt=None) -> {"text": str, "model": str}` — when `prompt` is given it becomes the user text sent to the vision model; when `None` the current default `"Describe this image in detail"` is preserved. Backward-compatible (every existing caller keeps working).

- [ ] **Step 1: Write the failing test**

Create `tests/test_vl_custom_prompt.py`:

```python
import base64

import src.document_processor as dp


def _tiny_png(tmp_path):
    # 1x1 PNG — enough for open()/base64 in the function under test
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    p = tmp_path / "img.png"
    p.write_bytes(png)
    return str(p)


def _stub_vl(monkeypatch, capture):
    # Make the function reach message construction + our fake llm_call without a network call.
    monkeypatch.setattr(dp, "_load_vl_settings", lambda: {"vision_enabled": True, "vision_model": "vl"})
    monkeypatch.setattr(dp, "_resolve_vl_model", lambda m, owner=None: ("http://x", "vl-model", {}))
    monkeypatch.setattr(dp, "resolve_vision_fallback_candidates", lambda owner=None: [], raising=False)

    def fake_llm_call(url, model, messages, headers=None, timeout=None):
        capture["messages"] = messages
        return "VL-REPLY"
    monkeypatch.setattr(dp, "llm_call", fake_llm_call)


def test_custom_prompt_is_sent_to_the_model(tmp_path, monkeypatch):
    cap = {}
    _stub_vl(monkeypatch, cap)
    out = dp.analyze_image_with_vl_result(_tiny_png(tmp_path), owner="admin",
                                          prompt="You are a controls expert. Diagnose this.")
    assert out == {"text": "VL-REPLY", "model": "vl-model"}
    text_part = cap["messages"][0]["content"][0]["text"]
    assert text_part == "You are a controls expert. Diagnose this."


def test_default_prompt_preserved_when_absent(tmp_path, monkeypatch):
    cap = {}
    _stub_vl(monkeypatch, cap)
    dp.analyze_image_with_vl_result(_tiny_png(tmp_path), owner="admin")
    assert cap["messages"][0]["content"][0]["text"] == "Describe this image in detail"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vl_custom_prompt.py --import-mode=importlib -q`
Expected: FAIL (`analyze_image_with_vl_result() got an unexpected keyword argument 'prompt'`).

- [ ] **Step 3: Write the implementation**

In `src/document_processor.py`, change the `analyze_image_with_vl_result` signature (line ~341) and the hardcoded text (line ~366):

```python
def analyze_image_with_vl_result(image_path: str, owner: str | None = None, *, prompt: str | None = None) -> dict:
    """Analyze an image and return both text and the model that produced it.

    `prompt` (optional) overrides the default instruction sent to the vision
    model — used by callers that need a task-specific expert prompt."""
```

and in the `vl_messages` construction, replace the hardcoded instruction:

```python
                    {"type": "text", "text": prompt or "Describe this image in detail"},
```

Do not change anything else in the function (model resolution, fallback chain, error handling all stay).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_vl_custom_prompt.py --import-mode=importlib -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/document_processor.py tests/test_vl_custom_prompt.py
git commit -m "feat(vision): analyze_image_with_vl_result accepts an optional prompt

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: The `diagnose_equipment` tool handler

**Files:**
- Create: `src/agent_tools/industrial_tools.py`
- Test: `tests/test_diagnose_equipment.py`

**Interfaces:**
- Consumes (Task 1): `analyze_image_with_vl_result(image_path, owner, *, prompt=...)`.
- Produces: `class DiagnoseEquipmentTool` with `async def execute(self, content, ctx) -> dict`; module-level `async def diagnose_equipment(content, ctx, *, vision_call=None) -> dict`; `TASK_MODES` (dict of mode → expert-prompt fragment); `SAFETY_CLAUSE`, `STRUCTURE_CLAUSE` (str constants).

- [ ] **Step 1: Write the failing test**

Create `tests/test_diagnose_equipment.py`:

```python
import asyncio
import base64
import json

import src.agent_tools.industrial_tools as it


def _run(coro):
    return asyncio.run(coro)


def _img(tmp_path, name="p.png"):
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    p = tmp_path / name
    p.write_bytes(png)
    return str(p)


def _exec(content, ctx=None, vision_call=None):
    return _run(it.diagnose_equipment(content, ctx or {}, vision_call=vision_call))


def _capture_call():
    seen = {}
    def vc(image_path, owner=None, *, prompt=None):
        seen["image_path"] = image_path
        seen["owner"] = owner
        seen["prompt"] = prompt
        return {"text": "DIAGNOSIS", "model": "vl-model"}
    return vc, seen


def test_each_mode_selects_its_expert_prompt_and_returns_output(tmp_path):
    for mode in ("schematic", "fault_code", "vfd", "component"):
        vc, seen = _capture_call()
        out = _exec(json.dumps({"image": _img(tmp_path), "task": mode}), {"owner": "admin"}, vision_call=vc)
        assert out == {"output": "DIAGNOSIS"}
        # the EXACT mode fragment is present (distinguishes modes), plus shared safety + structure
        assert it.TASK_MODES[mode] in seen["prompt"]
        assert "not a substitute for a qualified person" in seen["prompt"]
        assert "never" in seen["prompt"].lower()          # safety constraint
        assert "likely_causes" in seen["prompt"]           # structured-output request


def test_safety_clause_in_every_mode_prompt(tmp_path):
    for mode in list(it.TASK_MODES) + ["auto"]:
        vc, seen = _capture_call()
        _exec(json.dumps({"image": _img(tmp_path), "task": mode}), {"owner": "admin"}, vision_call=vc)
        assert it.SAFETY_CLAUSE in seen["prompt"]


def test_context_is_folded_into_the_prompt(tmp_path):
    vc, seen = _capture_call()
    _exec(json.dumps({"image": _img(tmp_path), "task": "vfd", "context": "trips on start, overcurrent"}),
          {"owner": "admin"}, vision_call=vc)
    assert "trips on start, overcurrent" in seen["prompt"]


def test_default_task_is_auto(tmp_path):
    vc, seen = _capture_call()
    _exec(json.dumps({"image": _img(tmp_path)}), {"owner": "admin"}, vision_call=vc)
    assert it.TASK_MODES["auto"] in seen["prompt"]
    # an unknown task also falls back to auto
    vc2, seen2 = _capture_call()
    _exec(json.dumps({"image": _img(tmp_path), "task": "bogus"}), {"owner": "admin"}, vision_call=vc2)
    assert it.TASK_MODES["auto"] in seen2["prompt"]


def test_missing_image_arg_is_error(tmp_path):
    out = _exec(json.dumps({"task": "vfd"}), {"owner": "admin"}, vision_call=_capture_call()[0])
    assert "error" in out and "image" in out["error"].lower()


def test_nonexistent_or_non_image_path_is_error(tmp_path):
    out = _exec(json.dumps({"image": str(tmp_path / "nope.png")}), {"owner": "admin"}, vision_call=_capture_call()[0])
    assert "error" in out
    txt = tmp_path / "notes.txt"; txt.write_text("hi")
    out2 = _exec(json.dumps({"image": str(txt)}), {"owner": "admin"}, vision_call=_capture_call()[0])
    assert "error" in out2


def test_oversized_image_is_error(tmp_path, monkeypatch):
    big = tmp_path / "big.png"
    big.write_bytes(b"x" * 10)                 # small file, but shrink the cap so it trips
    monkeypatch.setattr(it, "MAX_IMAGE_BYTES", 5)
    out = _exec(json.dumps({"image": str(big)}), {"owner": "admin"}, vision_call=_capture_call()[0])
    assert "error" in out


def test_no_vision_model_available_is_error(tmp_path):
    def vc(image_path, owner=None, *, prompt=None):
        return {"text": "[No vision model configured — set one in Settings → Vision]", "model": ""}
    out = _exec(json.dumps({"image": _img(tmp_path)}), {"owner": "admin"}, vision_call=vc)
    assert "error" in out


def test_never_raises_on_vision_exception(tmp_path):
    def vc(image_path, owner=None, *, prompt=None):
        raise RuntimeError("boom")
    out = _exec(json.dumps({"image": _img(tmp_path)}), {"owner": "admin"}, vision_call=vc)
    assert "error" in out and "boom" in out["error"]


def test_bad_json_content_is_error(tmp_path):
    out = _exec("not json", {"owner": "admin"}, vision_call=_capture_call()[0])
    assert "error" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_diagnose_equipment.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.agent_tools.industrial_tools'`).

- [ ] **Step 3: Write the implementation**

Create `src/agent_tools/industrial_tools.py`:

```python
"""The `diagnose_equipment` builtin tool: route an industrial image (schematic,
fault-code/HMI, VFD, component/thermal) to the shipped vision model with a
maintenance-expert, safety-first prompt, and return a structured diagnosis.

A read/analyze vision tool (matches capture_screen/webcam_look gating). The
vision inference is injectable so tests never hit a real model. The handler
NEVER raises into the agent — every failure returns {"error": ...}."""
import json
import os

MAX_IMAGE_BYTES = 12 * 1024 * 1024          # vision models choke on huge inputs
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

# Mandatory in every mode's prompt — industrial equipment is a life-safety context.
SAFETY_CLAUSE = (
    "SAFETY: This is decision-support, not a substitute for a qualified person. "
    "The technician must verify the equipment is de-energized and follow LOTO and "
    "site procedures before any hands-on work. NEVER advise bypassing interlocks, "
    "defeating guards/safety devices, or working on energized or unguarded equipment; "
    "frame checks as performed with the equipment safely isolated where applicable. "
    "Lead with the safety warning whenever a live electrical, stored-energy, or "
    "rotating hazard is involved.")

# Requested (not strictly parsed) result shape — the model returns labeled prose.
STRUCTURE_CLAUSE = (
    "Answer with these labeled sections: observed (what is visible), assessment "
    "(the decoded fault / interpretation / identification), likely_causes (ranked, "
    "most probable first), recommended_steps (ordered checks), safety (hazard-"
    "specific warnings), confidence (and what to verify against the real equipment "
    "or manual).")

TASK_MODES = {
    "schematic": ("Act as an industrial controls and electrical expert. Interpret this "
                  "electrical schematic or wiring diagram: trace the relevant circuit, "
                  "identify symbols and components, and locate the section relevant to the "
                  "reported problem, noting wire numbers and terminals where legible."),
    "fault_code": ("Act as an industrial maintenance expert. Read the text on this HMI, "
                   "panel, or drive display, decode the fault or alarm code for the "
                   "equipment, and explain what it means and its common causes."),
    "vfd": ("Act as a variable-frequency-drive specialist. From this drive display or "
            "fault photo, map the fault to its family (overcurrent, overvoltage, "
            "undervoltage, overheat, ground fault, overload) and give the parameter and "
            "measurement check sequence to isolate the cause."),
    "component": ("Act as an industrial maintenance expert. Identify the component or part "
                  "in this photo (type, function, and any legible manufacturer or part "
                  "number). If this is a thermal image, interpret the hotspots and "
                  "temperature deltas and flag the abnormal component."),
    "auto": ("Act as an industrial maintenance expert. First determine what this image "
             "shows (a schematic, an HMI or drive fault display, a component, or a thermal "
             "image), then diagnose it accordingly."),
}


def _compose_prompt(task, context):
    mode = TASK_MODES.get(task, TASK_MODES["auto"])
    parts = [mode, SAFETY_CLAUSE, STRUCTURE_CLAUSE]
    if context:
        parts.append(f"Technician-provided context: {context}")
    return "\n\n".join(parts)


def _validate_image(path):
    """Return an error string, or None if the image is usable."""
    if not path:
        return "diagnose_equipment: an 'image' path is required"
    if not os.path.isfile(path):
        return f"diagnose_equipment: image not found: {path}"
    if os.path.splitext(path)[1].lower() not in _IMAGE_EXTS:
        return f"diagnose_equipment: not a recognized image file: {path}"
    try:
        if os.path.getsize(path) > MAX_IMAGE_BYTES:
            return f"diagnose_equipment: image too large (> {MAX_IMAGE_BYTES} bytes)"
    except OSError as e:
        return f"diagnose_equipment: cannot read image: {e}"
    return None


async def diagnose_equipment(content, ctx, *, vision_call=None):
    ctx = ctx or {}
    if vision_call is None:
        from src.document_processor import analyze_image_with_vl_result as vision_call
    try:
        args = json.loads(content) if content and content.strip() else {}
        if not isinstance(args, dict):
            return {"error": "diagnose_equipment: arguments must be a JSON object"}
    except (ValueError, TypeError):
        return {"error": "diagnose_equipment: arguments must be valid JSON"}

    image = args.get("image")
    err = _validate_image(image)
    if err:
        return {"error": err}
    task = (args.get("task") or "auto").strip().lower()
    if task not in TASK_MODES:
        task = "auto"
    prompt = _compose_prompt(task, (args.get("context") or "").strip())

    try:
        result = vision_call(image, ctx.get("owner"), prompt=prompt)
    except Exception as e:  # never raise into the agent loop
        return {"error": f"diagnose_equipment: vision call failed: {e}"}

    text = (result or {}).get("text", "")
    model = (result or {}).get("model", "")
    # analyze_image_with_vl_result signals unavailability with an empty model + a
    # bracketed marker in the text (e.g. "[No vision model configured …]").
    if not model and text.strip().startswith("["):
        return {"error": text.strip()}
    if not text.strip():
        return {"error": "diagnose_equipment: vision model returned no result"}
    return {"output": text}


class DiagnoseEquipmentTool:
    async def execute(self, content, ctx):
        return await diagnose_equipment(content, ctx)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_diagnose_equipment.py --import-mode=importlib -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add src/agent_tools/industrial_tools.py tests/test_diagnose_equipment.py
git commit -m "feat(industrial): diagnose_equipment tool — 4 modes, safety-first, never-raises

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Register the tool at every surface

**Files:**
- Modify: `src/agent_tools/__init__.py` (import + `TOOL_HANDLERS` + `TOOL_TAGS`)
- Modify: `src/tool_schemas.py` (`FUNCTION_TOOL_SCHEMAS`)
- Modify: `src/tool_security.py` (`NON_ADMIN_BLOCKED_TOOLS` + `PLAN_MODE_READONLY_TOOLS`)
- Modify: `src/tool_index.py` (`BUILTIN_TOOL_DESCRIPTIONS`)
- Modify: `src/agent_loop.py` (`TOOL_SECTIONS` + `_DOMAIN_TOOL_MAP["desktop"]`)
- Test: `tests/test_diagnose_equipment_registration.py`

**Interfaces:**
- Consumes (Task 2): `DiagnoseEquipmentTool` from `src.agent_tools.industrial_tools`.
- Produces: `diagnose_equipment` present at every registration surface; gated exactly like `capture_screen` (admin-blocked AND plan-mode-readonly).

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_diagnose_equipment_registration.py`:

```python
def test_registered_in_handlers_and_tags():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    from src.agent_tools.industrial_tools import DiagnoseEquipmentTool
    assert "diagnose_equipment" in TOOL_HANDLERS
    assert "diagnose_equipment" in TOOL_TAGS
    assert TOOL_HANDLERS["diagnose_equipment"].__self__.__class__ is DiagnoseEquipmentTool


def test_gating_matches_capture_screen():
    import src.tool_security as ts
    # a read-only vision tool: admin-blocked AND plan-mode readonly (like capture_screen)
    assert "diagnose_equipment" in ts.NON_ADMIN_BLOCKED_TOOLS
    assert "diagnose_equipment" in ts.PLAN_MODE_READONLY_TOOLS
    assert "capture_screen" in ts.NON_ADMIN_BLOCKED_TOOLS and "capture_screen" in ts.PLAN_MODE_READONLY_TOOLS


def test_has_schema():
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    names = {s["function"]["name"] for s in FUNCTION_TOOL_SCHEMAS if s.get("type") == "function"}
    assert "diagnose_equipment" in names


def test_in_index_and_agent_loop_desktop_domain():
    import src.tool_index as ti
    import src.agent_loop as al
    assert "diagnose_equipment" in ti.BUILTIN_TOOL_DESCRIPTIONS
    assert "diagnose_equipment" in al.TOOL_SECTIONS
    assert "diagnose_equipment" in al._DOMAIN_TOOL_MAP["desktop"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_diagnose_equipment_registration.py --import-mode=importlib -q`
Expected: FAIL (not registered anywhere yet).

- [ ] **Step 3: Register at each surface**

Read each file's current structure first, then add `diagnose_equipment` following the `capture_screen`/`webcam_look` sibling pattern:

1. **`src/agent_tools/__init__.py`:** add `from .industrial_tools import DiagnoseEquipmentTool` with the other `from .` imports; add `"diagnose_equipment": DiagnoseEquipmentTool().execute,` to `TOOL_HANDLERS`; add `"diagnose_equipment"` to `TOOL_TAGS`.

2. **`src/tool_schemas.py`:** add to the `FUNCTION_TOOL_SCHEMAS` list:
```python
    {
        "type": "function",
        "function": {
            "name": "diagnose_equipment",
            "description": "Diagnose an industrial image with a maintenance-expert vision model: read a fault code / HMI / drive display, interpret an electrical schematic or wiring diagram, troubleshoot a VFD/drive fault, or identify a component / interpret a thermal image. Returns a safety-first structured diagnosis. Provide an image file path (an attachment, a capture_screen screenshot, or a file).",
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Path to the image file to diagnose."},
                    "task": {"type": "string", "enum": ["schematic", "fault_code", "vfd", "component", "auto"],
                              "description": "Diagnostic mode (default auto: infer the image type)."},
                    "context": {"type": "string", "description": "Optional: the symptom, equipment model/tag, or what was already checked."}
                },
                "required": ["image"]
            }
        }
    },
```

3. **`src/tool_security.py`:** add `"diagnose_equipment"` to **`NON_ADMIN_BLOCKED_TOOLS`** (near `"capture_screen"`, ~line 56) AND to **`PLAN_MODE_READONLY_TOOLS`** (near `"capture_screen"`, ~line 140). Both, matching the vision-tools family (admin-only, read-only).

4. **`src/tool_index.py`:** add to `BUILTIN_TOOL_DESCRIPTIONS`: `"diagnose_equipment": "Diagnose an industrial image (schematic, fault code/HMI, VFD fault, component/thermal) with a maintenance-expert vision model.",`.

5. **`src/agent_loop.py`:**
   - `TOOL_SECTIONS` — add an entry modeling the `capture_screen` section (~line 418):
     ```python
     "diagnose_equipment": "- ```diagnose_equipment``` — Diagnose an industrial image with a maintenance-expert vision model. Args (JSON): {\"image\": \"<path>\", \"task\": \"schematic|fault_code|vfd|component|auto\", \"context\": \"<optional symptom>\"}. Use for fault codes, schematics, VFD faults, component ID, thermal images.",
     ```
   - `_DOMAIN_TOOL_MAP["desktop"]` (~line 306) — add `"diagnose_equipment"` to the `"desktop"` set alongside `"capture_screen"`/`"webcam_look"`.

- [ ] **Step 4: Run the parity test + import smoke**

Run: `python -m pytest tests/test_diagnose_equipment_registration.py --import-mode=importlib -q` (Expected: PASS, 4 passed) and `python -c "import app"` (no error).

- [ ] **Step 5: Run the Task-2 handler suite (no regression)**

Run: `python -m pytest tests/test_diagnose_equipment.py --import-mode=importlib -q`
Expected: PASS (10 passed).

- [ ] **Step 6: Commit**

```bash
git add src/agent_tools/__init__.py src/tool_schemas.py src/tool_security.py src/tool_index.py src/agent_loop.py tests/test_diagnose_equipment_registration.py
git commit -m "feat(industrial): register diagnose_equipment (vision-tools gating)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **All three tasks are TDD.** No test may run a real model/vision — Task 1 mocks `document_processor.llm_call`; Task 2 injects a fake `vision_call`; Task 3 asserts registration + `import app`.
- **Safety framing is a spec requirement, not decoration** — `SAFETY_CLAUSE` must appear in *every* mode's composed prompt; the Task-2 test enforces this. Do not weaken it.
- **Gating is a security decision** — `diagnose_equipment` matches `capture_screen`: admin-only (`NON_ADMIN_BLOCKED_TOOLS`) AND plan-mode-readonly (`PLAN_MODE_READONLY_TOOLS`). It is read-only (sees an image, no side effects), so unlike `run_workflow` it *is* plan-mode-readonly.
- **Verify the real module-level names** before editing (`FUNCTION_TOOL_SCHEMAS`, `BUILTIN_TOOL_DESCRIPTIONS`, `NON_ADMIN_BLOCKED_TOOLS`, `PLAN_MODE_READONLY_TOOLS`, `TOOL_SECTIONS`, `_DOMAIN_TOOL_MAP`) — all verified present at spec time; if one moved, fix the registration AND the assertion to the real name, never weaken an assertion.
- **The end-to-end "show the agent a real fault photo" path needs a served vision model** — that's the manual answer-quality check, owed by the domain expert, not automated here.
- Scope: one tool, four modes. Do NOT build OPC UA/live data, a manual RAG, predictive maintenance, or a UI panel (v1 non-goals).
