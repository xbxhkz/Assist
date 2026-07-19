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
