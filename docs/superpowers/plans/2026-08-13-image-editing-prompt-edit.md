# AI Image Editing Sub-project 2: Natural-Language Editing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user upload an image in chat, describe an edit in natural language ("add a red hat", "make the sky sunset-colored"), and get back an edited image, shown inline and saved to the Gallery — via img2img on the bundled `sd-server.exe` binary, the same process already used for `generate_image`.

**Architecture:** A new module, `src/image_edit.py`, POSTs to the already-running `sd-server.exe`'s `/sdapi/v1/img2img` endpoint (live-verified by this sub-project's own feasibility spike to genuinely condition on the input image, unlike ControlNet's silently-ignored `control_image`). A new builtin agent tool, `edit_image_prompt`, resolves a chat attachment, auto-serves the default image model, calls the new module, and returns the result — registered with every lesson sub-project 1's whole-branch review surfaced applied from day one, not discovered again after the fact.

**Tech Stack:** `httpx` (already a dependency, used by `ai_interaction.py`'s own image-generation call), no new runtime dependencies.

## Global Constraints

- No new runtime dependencies.
- `src/image_edit.py`'s public function is `edit_image(image_bytes: bytes, prompt: str, base_url: str, *, headers=None, poster=None) -> bytes` — raises on failure (mirrors `src/bg_removal.py`'s convention: the core module is an honest, raising library function; the never-raises discipline is applied by its callers).
- Fixed generation parameters for v1, per the approved design's "no tunable knobs" decision: `denoising_strength=0.6`, `steps=20` (the feasibility spike's validated configuration), `width=512, height=512` (matches this app's own established OOM-avoidance precedent for local diffusion on small GPUs — `src/ai_interaction.py`'s `_local_diffusion_size_hint` docstring: "FLUX.1 12B OOMs at 1024x1024 on a 6GB card but generates fine at 512x512" — and the spike's own tested, working configuration). None of these are exposed as tool parameters.
- The new builtin tool is named `edit_image_prompt` — deliberately distinct from the existing, broken, out-of-scope `edit_image` tool (`src/tools/image.py`, confirmed in sub-project 1's research to proxy to nonexistent `/api/gallery/{action}` routes). `edit_image_prompt` is new, standalone code; `edit_image` is untouched.
- Tests must never require a real, running `sd-server` — every test injects a fake `poster` (module-level tests) or a fake `editor` (tool-level tests), mirroring `src/bg_removal.py`/`remove_background_tool`'s established injectable-dependency pattern exactly.
- **Registration applies every lesson sub-project 1's whole-branch review found, from the start** — not just the original 6 points that sub-project were later found incomplete. Confirmed against the CURRENT, already-corrected live code (not the plan text that predated the fix), at these exact locations:
  - `src/tool_execution.py:870-881` — the `elif tool == "remove_background":` dispatch branch that threads `owner=`/`session_id=` (added in sub-project 1's final-review fix wave; the tool would otherwise fall into the generic `dynamic_handlers` catch-all, which never threads `owner`, making it fail-closed for every real user in any auth-enabled install while tests stay green).
  - `routes/chat_routes.py:843-844` — `if not _privs.get("can_generate_images", True): disabled_tools.update({"generate_image", "remove_background"})`.
  - `src/tool_security.py:186-187` — `_PLAN_MODE_KNOWN_MUTATORS` includes `"generate_image", "edit_image", ..., "remove_background"`.
  - `src/tool_security.py`'s `NON_ADMIN_BLOCKED_TOOLS` does NOT include `generate_image` (confirmed by direct read) — `edit_image_prompt` mirrors this (not added), matching `generate_image`/`remove_background`'s precedent exactly.
- `src/agent_tools/image_tools.py` already has `_default_gallery_saver(image_bytes, owner)` (writes `prompt="Background removed", model="remove_background"`) and `remove_background_tool`, both from sub-project 1. This plan extends that file rather than creating a new one, matching this codebase's `<domain>_tools.py` convention of grouping multiple tools per domain (mirrors `desktop_tools.py`, `industrial_tools.py`).

---

### Task 1: `src/image_edit.py` — the img2img calling module

**Files:**
- Create: `src/image_edit.py`
- Test: `tests/test_image_edit.py`

**Interfaces:**
- Produces: `edit_image(image_bytes: bytes, prompt: str, base_url: str, *, headers=None, poster=None) -> bytes` — PNG bytes in, PNG bytes out. `poster` is an injectable callable `(base_url, payload, headers) -> dict` (the parsed JSON response), matching `src/bg_removal.py`'s `session=None` injectable-dependency pattern. When omitted, performs a real HTTP POST via `httpx`. Raises on any failure — callers apply the never-raises discipline at their own boundary.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_image_edit.py`:

```python
"""image_edit.edit_image POSTs to sd-server's /sdapi/v1/img2img (live-verified
by this sub-project's feasibility spike to genuinely condition on init_images,
unlike ControlNet's silently-ignored control_image) via an injectable poster,
mirroring src/bg_removal.py's session=None pattern so tests never need a real
running sd-server. See
docs/superpowers/specs/2026-08-12-image-editing-prompt-edit-design.md.
"""
import base64

import pytest

from src import image_edit


def _fake_poster(response=None, capture=None):
    response = response if response is not None else {"images": ["ZmFrZS1wbmc="]}

    def poster(base_url, payload, headers):
        if capture is not None:
            capture["base_url"] = base_url
            capture["payload"] = payload
            capture["headers"] = headers
        return response
    return poster


def test_edit_image_returns_decoded_bytes_from_response():
    result = image_edit.edit_image(
        b"input-png-bytes", "add a red hat", "http://127.0.0.1:8300",
        poster=_fake_poster(response={"images": [base64.b64encode(b"edited-png").decode()]}),
    )
    assert result == b"edited-png"


def test_edit_image_strips_data_uri_prefix_from_response():
    b64 = base64.b64encode(b"edited-png").decode()
    result = image_edit.edit_image(
        b"input-png-bytes", "add a red hat", "http://127.0.0.1:8300",
        poster=_fake_poster(response={"images": [f"data:image/png;base64,{b64}"]}),
    )
    assert result == b"edited-png"


def test_edit_image_sends_expected_payload_shape():
    capture = {}
    image_edit.edit_image(
        b"input-png-bytes", "add a red hat", "http://127.0.0.1:8300",
        poster=_fake_poster(capture=capture),
    )
    payload = capture["payload"]
    assert payload["prompt"] == "add a red hat"
    assert payload["denoising_strength"] == 0.6
    assert payload["steps"] == 20
    assert payload["width"] == 512
    assert payload["height"] == 512
    assert payload["init_images"][0].startswith("data:image/png;base64,")
    decoded = base64.b64decode(payload["init_images"][0].split(",", 1)[1])
    assert decoded == b"input-png-bytes"


def test_edit_image_passes_base_url_and_headers_through():
    capture = {}
    image_edit.edit_image(
        b"input-png-bytes", "add a red hat", "http://127.0.0.1:8300",
        headers={"Authorization": "Bearer x"},
        poster=_fake_poster(capture=capture),
    )
    assert capture["base_url"] == "http://127.0.0.1:8300"
    assert capture["headers"] == {"Authorization": "Bearer x"}


def test_edit_image_raises_when_no_images_in_response():
    with pytest.raises(RuntimeError):
        image_edit.edit_image(
            b"input-png-bytes", "add a red hat", "http://127.0.0.1:8300",
            poster=_fake_poster(response={"images": []}),
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_image_edit.py -v --import-mode=importlib`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.image_edit'`.

- [ ] **Step 3: Create `src/image_edit.py`**

```python
"""Natural-language image editing via img2img on the bundled sd-server
binary. This sub-project's own feasibility spike live-verified that
init_images/denoising_strength genuinely condition sd-server's output --
the opposite result from a prior ControlNet sub-project, where control_image
was accepted by the API but silently ignored. See
docs/superpowers/specs/2026-08-12-image-editing-prompt-edit-design.md.
"""
import base64

import httpx

# Fixed generation parameters for v1 (not exposed as tool parameters, per the
# approved design's "no tunable knobs" decision):
_WIDTH = 512
_HEIGHT = 512  # Matches this app's own established OOM-avoidance precedent
# for local diffusion on small GPUs (src/ai_interaction.py's
# _local_diffusion_size_hint: FLUX OOMs at 1024x1024 on a 6GB card, works
# reliably at 512x512) and the feasibility spike's own tested configuration.
_DENOISING_STRENGTH = 0.6  # The spike's validated balance point: a real,
# visible edit that still resembles the original image.
_STEPS = 20


def _post(base_url: str, payload: dict, headers: dict) -> dict:
    with httpx.Client(timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)) as client:
        resp = client.post(base_url + "/sdapi/v1/img2img", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


def edit_image(image_bytes: bytes, prompt: str, base_url: str, *, headers=None, poster=None) -> bytes:
    """Edit image_bytes per prompt via img2img on the sd-server instance at
    base_url. Returns PNG bytes. Raises on failure -- the never-raises
    discipline is applied by callers, matching src/bg_removal.py's convention.
    """
    payload = {
        "init_images": ["data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")],
        "prompt": prompt,
        "denoising_strength": _DENOISING_STRENGTH,
        "steps": _STEPS,
        "width": _WIDTH,
        "height": _HEIGHT,
    }
    post = poster or _post
    data = post(base_url, payload, headers or {})

    images = data.get("images") or []
    if not images:
        raise RuntimeError("sd-server returned no image from img2img")
    b64 = images[0]
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_image_edit.py -v --import-mode=importlib`
Expected: PASS (all 5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/image_edit.py tests/test_image_edit.py
git commit -m "feat(image-edit): img2img calling module for natural-language image editing

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `edit_image_prompt` builtin agent tool + full registration

**Files:**
- Modify: `src/agent_tools/image_tools.py` (parameterize `_default_gallery_saver`; add `edit_image_prompt_tool` + `EditImagePromptTool`)
- Modify: `src/agent_tools/__init__.py` (import + `TOOL_HANDLERS` + `TOOL_TAGS`)
- Modify: `src/tool_schemas.py` (`FUNCTION_TOOL_SCHEMAS`)
- Modify: `src/agent_loop.py` (`TOOL_SECTIONS` + `_DOMAIN_TOOL_MAP["desktop"]`)
- Modify: `src/tool_index.py` (`BUILTIN_TOOL_DESCRIPTIONS`)
- Modify: `src/tool_execution.py` (owner-threading dispatch branch)
- Modify: `routes/chat_routes.py` (`can_generate_images` privilege gate)
- Modify: `src/tool_security.py` (`_PLAN_MODE_KNOWN_MUTATORS`)
- Test: `tests/test_edit_image_prompt_tool.py`
- Test: `tests/test_edit_image_prompt_registration.py`

**Interfaces:**
- Consumes: `src.image_edit.edit_image(image_bytes, prompt, base_url, *, headers=None, poster=None)` (Task 1); `src.upload_handler.UploadHandler.resolve_upload(upload_id, owner=None)` (existing, same as `remove_background_tool`); `src.ai_interaction._apply_image_autoserve(model_spec, explicit, owner, *, ensure=None)` and `src.ai_interaction._resolve_model(spec, owner=None)` (existing, private-by-convention but already imported this way from 8+ other modules across this codebase — `mcp_servers/image_gen_server.py`, `src/agent_tools/session_tools.py`, `src/agent_tools/model_interaction_tools.py`, `src/document_processor.py`, `routes/operator_routes.py`, `routes/preset_routes.py`, `routes/skills_routes.py`, `src/localmodels/manager.py`, `src/teacher_escalation.py` — this is an established, accepted cross-module reuse pattern in this codebase, not a new precedent).
- Produces: `edit_image_prompt_tool(content, ctx, *, editor=None, upload_resolver=None, gallery_saver=None)` in `src/agent_tools/image_tools.py`, and `EditImagePromptTool` (a thin `execute(self, content, ctx)` wrapper class), registered as builtin tool `"edit_image_prompt"`.

**On `_default_gallery_saver`**: currently (post sub-project 1) `_default_gallery_saver(image_bytes, owner)` hardcodes `prompt="Background removed", model="remove_background"`. This task parameterizes it to `_default_gallery_saver(image_bytes, owner, *, prompt="Background removed", model="remove_background")` — the defaults preserve `remove_background_tool`'s existing call (`saver(result_bytes, owner)`, 2 positional args, unaffected by this change) exactly, while `edit_image_prompt_tool` calls it with edit-specific values via a small per-call closure (see Step 3) rather than duplicating the file-write + DB-insert logic in a second near-identical function.

**On auto-serving**: `edit_image_prompt_tool` always auto-serves the configured default image model (v1 has no "pick a specific model" parameter, matching the approved design). It calls `_apply_image_autoserve("", False, owner)` — `model_spec=""` and `explicit=False` force the auto-serve branch (`if explicit: return model_spec, None` is skipped) — then `_resolve_model(model_spec, owner=owner)` to get `(url, model_id, headers)`, then derives `base_url` the identical way `do_generate_image` does (`src/ai_interaction.py:1073`): `url.replace("/chat/completions", "").replace("/v1/messages", "").rstrip("/")`. If `_apply_image_autoserve` returns a falsy `model_spec` (the rare case where a local model is serving but its advertised id couldn't be probed), this task returns a clear error rather than replicating `do_generate_image`'s further DB-endpoint-discovery fallback (`src/ai_interaction.py:1023-1056`) — an accepted v1 simplification, not a silent gap, since it only affects an already-rare edge case and fails with an actionable message rather than misbehaving.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_edit_image_prompt_tool.py`:

```python
"""edit_image_prompt_tool resolves a chat attachment, auto-serves the default
image model, runs it through src.image_edit, and returns an inline
data: URI or short served URL via the established image_url convention
(same as remove_background_tool). Never raises into the agent loop. See
docs/superpowers/specs/2026-08-12-image-editing-prompt-edit-design.md.
"""
import asyncio
import base64
import json

from src.agent_tools.image_tools import EditImagePromptTool, edit_image_prompt_tool


def _fake_upload_resolver(found=True, path="/tmp/fake.png"):
    def resolver(upload_id, owner=None):
        if not found:
            return None
        return {"id": upload_id, "path": path, "name": "fake.png", "mime": "image/png"}
    return resolver


def _fake_editor(output=b"fake-edited-png"):
    def editor(image_bytes, prompt, base_url, headers):
        return output
    return editor


def _fake_gallery_saver(image_id="gallery-img-1", filename="abc123def456.png"):
    calls = []

    def saver(image_bytes, owner):
        calls.append((image_bytes, owner))
        return {"id": image_id, "filename": filename}

    saver.calls = calls
    return saver


def test_missing_attachment_id_returns_error():
    content = json.dumps({"prompt": "add a hat"})
    result = asyncio.run(edit_image_prompt_tool(content, {"owner": "alice"}))
    assert "error" in result


def test_missing_prompt_returns_error():
    content = json.dumps({"attachment_id": "up-1"})
    result = asyncio.run(edit_image_prompt_tool(content, {"owner": "alice"}))
    assert "error" in result


def test_invalid_json_returns_error():
    result = asyncio.run(edit_image_prompt_tool("not json", {"owner": "alice"}))
    assert "error" in result


def test_unresolvable_attachment_returns_error():
    content = json.dumps({"attachment_id": "missing-id", "prompt": "add a hat"})
    result = asyncio.run(edit_image_prompt_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=False),
        editor=_fake_editor(),
    ))
    assert "error" in result


def test_successful_edit_returns_short_url_and_saves_to_gallery(tmp_path):
    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1", "prompt": "add a red hat"})
    gallery_saver = _fake_gallery_saver(image_id="gallery-img-1", filename="abc123def456.png")

    result = asyncio.run(edit_image_prompt_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        editor=_fake_editor(output=b"edited-bytes"),
        gallery_saver=gallery_saver,
    ))

    assert result["image_url"] == "/api/generated-image/abc123def456.png"
    assert "base64" not in result["image_url"]
    assert result["gallery_image_id"] == "gallery-img-1"
    assert gallery_saver.calls == [(b"edited-bytes", "alice")]


def test_model_failure_returns_error_not_raise(tmp_path):
    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1", "prompt": "add a red hat"})

    def failing_editor(image_bytes, prompt, base_url, headers):
        raise RuntimeError("boom")

    result = asyncio.run(edit_image_prompt_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        editor=failing_editor,
        gallery_saver=_fake_gallery_saver(),
    ))

    assert "error" in result


def test_gallery_save_failure_falls_back_to_inline_data_uri(tmp_path):
    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1", "prompt": "add a red hat"})

    def failing_saver(image_bytes, owner):
        raise RuntimeError("db unavailable")

    result = asyncio.run(edit_image_prompt_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        editor=_fake_editor(output=b"edited-bytes"),
        gallery_saver=failing_saver,
    ))

    assert result["image_url"].startswith("data:image/png;base64,")
    decoded = base64.b64decode(result["image_url"].split(",", 1)[1])
    assert decoded == b"edited-bytes"
    assert "gallery_image_id" not in result


def test_gallery_saver_receives_prompt_and_model_via_default_saver(tmp_path, monkeypatch):
    """The real (non-injected) saver path must write the user's actual edit
    prompt and the serving model id into the Gallery row -- not
    remove_background's hardcoded 'Background removed'/'remove_background'
    defaults, which _default_gallery_saver still uses for ITS OWN caller."""
    import src.agent_tools.image_tools as image_tools

    captured = {}

    def fake_default_saver(image_bytes, owner, *, prompt="Background removed", model="remove_background"):
        captured["prompt"] = prompt
        captured["model"] = model
        return {"id": "gid", "filename": "f.png"}

    monkeypatch.setattr(image_tools, "_default_gallery_saver", fake_default_saver)

    real_file = tmp_path / "upload.png"
    real_file.write_bytes(b"original-bytes")
    content = json.dumps({"attachment_id": "up-1", "prompt": "add a red hat"})

    def fake_editor(image_bytes, prompt, base_url, headers):
        return b"edited-bytes"

    asyncio.run(edit_image_prompt_tool(
        content, {"owner": "alice"},
        upload_resolver=_fake_upload_resolver(found=True, path=str(real_file)),
        editor=fake_editor,
    ))

    assert captured["prompt"] == "add a red hat"


def test_tool_class_delegates_to_module_function():
    tool = EditImagePromptTool()
    content = json.dumps({"prompt": "add a hat"})
    result = asyncio.run(tool.execute(content, {"owner": "alice"}))
    assert "error" in result
```

Create `tests/test_edit_image_prompt_registration.py`:

```python
"""edit_image_prompt must be registered everywhere a builtin tool needs to be,
applying every lesson sub-project 1's whole-branch review found (the
dispatcher owner-threading branch, the can_generate_images privilege gate,
and the plan-mode backstop) from the start, not discovered again after the
fact. Mirrors tests/test_remove_background_registration.py's structure. See
docs/superpowers/specs/2026-08-12-image-editing-prompt-edit-design.md."""
import asyncio
from pathlib import Path

_CHAT_ROUTES = Path(__file__).resolve().parent.parent / "routes" / "chat_routes.py"


def test_edit_image_prompt_registered_everywhere():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    from src.agent_loop import TOOL_SECTIONS, _DOMAIN_TOOL_MAP
    from src.tool_index import BUILTIN_TOOL_DESCRIPTIONS

    assert "edit_image_prompt" in TOOL_HANDLERS
    assert "edit_image_prompt" in TOOL_TAGS
    names = [(s.get("function") or {}).get("name") for s in FUNCTION_TOOL_SCHEMAS]
    assert "edit_image_prompt" in names
    assert "edit_image_prompt" in TOOL_SECTIONS
    assert "edit_image_prompt" in _DOMAIN_TOOL_MAP["desktop"]
    assert "edit_image_prompt" in BUILTIN_TOOL_DESCRIPTIONS


def test_edit_image_prompt_matches_generate_image_admin_gating():
    from src.tool_security import NON_ADMIN_BLOCKED_TOOLS
    generate_image_blocked = "generate_image" in NON_ADMIN_BLOCKED_TOOLS
    edit_image_prompt_blocked = "edit_image_prompt" in NON_ADMIN_BLOCKED_TOOLS
    assert edit_image_prompt_blocked == generate_image_blocked, (
        "edit_image_prompt's NON_ADMIN_BLOCKED_TOOLS membership must mirror "
        "generate_image's, not default to admin-only"
    )


def test_edit_image_prompt_blocked_when_can_generate_images_disabled():
    source = _CHAT_ROUTES.read_text(encoding="utf-8")
    assert 'if not _privs.get("can_generate_images", True):' in source
    idx = source.index('if not _privs.get("can_generate_images", True):')
    following = source[idx: idx + 300]
    assert "edit_image_prompt" in following, (
        "edit_image_prompt must be added to disabled_tools in the same "
        "can_generate_images privilege branch as generate_image/remove_background"
    )


def test_dispatcher_threads_owner_and_session_into_tool_ctx(monkeypatch):
    """The REAL dispatcher (execute_tool_block, not a mock of it) must thread
    owner= into edit_image_prompt's ctx -- otherwise the tool falls into the
    generic dynamic_handlers catch-all, which never threads owner, and
    resolve_upload denies every real (owned) attachment. This is the exact
    gap sub-project 1's whole-branch review found for remove_background,
    applied here from the start."""
    import src.agent_tools as agent_tools
    from src.agent_tools import ToolBlock
    from src.tool_execution import execute_tool_block

    seen = {}

    async def spy(content, ctx):
        seen["content"] = content
        seen["ctx"] = ctx
        return {"output": "Image edited.", "exit_code": 0}

    monkeypatch.setitem(agent_tools.TOOL_HANDLERS, "edit_image_prompt", spy)

    desc, result = asyncio.run(execute_tool_block(
        ToolBlock("edit_image_prompt", '{"attachment_id": "up-1", "prompt": "add a hat"}'),
        session_id="sess-1",
        owner="alice",
    ))

    assert result.get("exit_code") == 0
    assert seen["ctx"].get("owner") == "alice", (
        "edit_image_prompt's ctx lost the owner -- resolve_upload will deny "
        "every owned attachment"
    )
    assert seen["ctx"].get("session_id") == "sess-1"


def test_edit_image_prompt_in_plan_mode_known_mutators():
    """edit_image_prompt writes a PNG to disk and inserts a Gallery DB row --
    the same class of mutator as generate_image/edit_image/remove_background,
    all members of _PLAN_MODE_KNOWN_MUTATORS, the defense-in-depth backstop
    that keeps known mutators blocked in plan mode even if the schema-derived
    denylist logic fails to import."""
    import src.tool_security as ts
    assert "edit_image_prompt" in ts._PLAN_MODE_KNOWN_MUTATORS
    disabled = ts.plan_mode_disabled_tools()
    assert "edit_image_prompt" in disabled
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_edit_image_prompt_tool.py tests/test_edit_image_prompt_registration.py -v --import-mode=importlib`
Expected: FAIL — `ImportError: cannot import name 'edit_image_prompt_tool' from 'src.agent_tools.image_tools'`.

- [ ] **Step 3: Modify `src/agent_tools/image_tools.py`**

Find this exact block:

```python
def _default_gallery_saver(image_bytes, owner):
    """Persist a new Gallery image, mirroring POST /api/gallery/upload's own
    GalleryImage field set exactly (routes/gallery/gallery_routes.py:230-248),
    minus the EXIF-derived fields that don't apply to a synthetically
    generated PNG.

    Returns {"id": <gallery row id>, "filename": <name under
    GENERATED_IMAGES_DIR>}. The filename matters as much as the id: it is what
    app.py's GET /api/generated-image/{filename} serves (with per-row owner
    enforcement), which lets the tool hand back a SHORT url instead of an
    inline multi-MB data: URI."""
    import hashlib
    import uuid
    from pathlib import Path

    from core.database import GalleryImage, SessionLocal
    from src.constants import GENERATED_IMAGES_DIR

    db = SessionLocal()
    try:
        img_dir = Path(GENERATED_IMAGES_DIR)
        img_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex[:12]}.png"
        (img_dir / filename).write_bytes(image_bytes)

        img_id = str(uuid.uuid4())
        db.add(GalleryImage(
            id=img_id,
            filename=filename,
            prompt="Background removed",
            model="remove_background",
            owner=owner,
            file_hash=hashlib.sha256(image_bytes).hexdigest(),
            file_size=len(image_bytes),
        ))
        db.commit()
        return {"id": img_id, "filename": filename}
    finally:
        db.close()
```

Replace it with (only the signature and the two `GalleryImage` field values change — everything else is identical):

```python
def _default_gallery_saver(image_bytes, owner, *, prompt="Background removed", model="remove_background"):
    """Persist a new Gallery image, mirroring POST /api/gallery/upload's own
    GalleryImage field set exactly (routes/gallery/gallery_routes.py:230-248),
    minus the EXIF-derived fields that don't apply to a synthetically
    generated PNG.

    Returns {"id": <gallery row id>, "filename": <name under
    GENERATED_IMAGES_DIR>}. The filename matters as much as the id: it is what
    app.py's GET /api/generated-image/{filename} serves (with per-row owner
    enforcement), which lets the tool hand back a SHORT url instead of an
    inline multi-MB data: URI. `prompt`/`model` default to
    remove_background's own values so its existing 2-positional-arg call
    (saver(result_bytes, owner)) is unaffected; edit_image_prompt passes its
    own values instead of duplicating this file-write + DB-insert logic."""
    import hashlib
    import uuid
    from pathlib import Path

    from core.database import GalleryImage, SessionLocal
    from src.constants import GENERATED_IMAGES_DIR

    db = SessionLocal()
    try:
        img_dir = Path(GENERATED_IMAGES_DIR)
        img_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex[:12]}.png"
        (img_dir / filename).write_bytes(image_bytes)

        img_id = str(uuid.uuid4())
        db.add(GalleryImage(
            id=img_id,
            filename=filename,
            prompt=prompt,
            model=model,
            owner=owner,
            file_hash=hashlib.sha256(image_bytes).hexdigest(),
            file_size=len(image_bytes),
        ))
        db.commit()
        return {"id": img_id, "filename": filename}
    finally:
        db.close()
```

Then, at the end of the file (after `RemoveBackgroundTool`'s class definition), append:

```python
async def edit_image_prompt_tool(content, ctx, *, editor=None, upload_resolver=None, gallery_saver=None):
    ctx = ctx or {}
    owner = ctx.get("owner")

    try:
        args = json.loads(content) if content and content.strip() else {}
        if not isinstance(args, dict):
            return {"error": "edit_image_prompt: arguments must be a JSON object"}
    except (ValueError, TypeError):
        return {"error": "edit_image_prompt: arguments must be valid JSON"}

    attachment_id = args.get("attachment_id")
    if not isinstance(attachment_id, str) or not attachment_id.strip():
        return {"error": "edit_image_prompt: an 'attachment_id' is required"}

    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "edit_image_prompt: a 'prompt' describing the edit is required"}

    if upload_resolver is None:
        from src.constants import DATA_DIR, UPLOAD_DIR
        from src.upload_handler import UploadHandler
        upload_resolver = UploadHandler(DATA_DIR, UPLOAD_DIR).resolve_upload

    try:
        info = upload_resolver(attachment_id, owner=owner)
    except Exception as e:
        return {"error": f"edit_image_prompt: could not resolve attachment: {e}"}

    if not info or not info.get("path"):
        return {"error": f"edit_image_prompt: attachment '{attachment_id}' not found"}

    try:
        # Off the event loop, matching remove_background_tool's established
        # pattern -- reading the source file and img2img inference can both
        # take real time.
        image_bytes = await asyncio.to_thread(_read_bytes, info["path"])
    except OSError as e:
        return {"error": f"edit_image_prompt: could not read attachment: {e}"}

    if editor is None:
        from src.ai_interaction import _apply_image_autoserve, _resolve_model
        from src.image_edit import edit_image as editor

        model_spec, autoserve_err = await _apply_image_autoserve("", False, owner)
        if autoserve_err:
            return {"error": f"edit_image_prompt: {autoserve_err}"}
        if not model_spec:
            return {"error": "edit_image_prompt: no local image model is configured or "
                              "being served; configure a default image model in "
                              "Admin -> Image Generation"}
        try:
            url, model_id, headers = await asyncio.to_thread(_resolve_model, model_spec, owner=owner)
        except ValueError as e:
            return {"error": f"edit_image_prompt: no endpoint found for image model "
                              f"'{model_spec}': {e}"}
        base_url = url.replace("/chat/completions", "").replace("/v1/messages", "").rstrip("/")
    else:
        # Injected editor (tests): skip real auto-serve/resolve entirely.
        base_url, model_id, headers = "", "edit_image_prompt", {}

    try:
        # headers must be passed as a keyword here: src.image_edit.edit_image's
        # real signature makes it keyword-only (after `*`), and asyncio.to_thread
        # forwards **kwargs as well as *args -- a positional call would raise
        # TypeError against the real function (test-injected editors accept
        # either form, since their `headers` parameter isn't keyword-only).
        result_bytes = await asyncio.to_thread(editor, image_bytes, prompt, base_url, headers=headers)
    except Exception as e:
        return {"error": f"edit_image_prompt: model failed: {e}"}

    result = {"output": "Image edited."}

    def _saver(image_bytes, owner):
        return _default_gallery_saver(image_bytes, owner, prompt=prompt, model=model_id)

    saver = gallery_saver or _saver
    try:
        saved = saver(result_bytes, owner)
    except Exception:
        logger.warning("edit_image_prompt: failed to save result to Gallery", exc_info=True)
        saved = None

    filename = None
    if isinstance(saved, dict):
        if saved.get("id"):
            result["gallery_image_id"] = saved["id"]
        _fn = saved.get("filename")
        filename = _fn if isinstance(_fn, str) and _fn.strip() else None
    elif saved:
        result["gallery_image_id"] = saved

    if filename:
        image_url = f"/api/generated-image/{filename}"
    else:
        image_url = "data:image/png;base64," + base64.b64encode(result_bytes).decode("ascii")
    result["image_url"] = image_url

    return result


class EditImagePromptTool:
    async def execute(self, content, ctx):
        return await edit_image_prompt_tool(content, ctx)
```

- [ ] **Step 4: Register in `src/agent_tools/__init__.py`**

Find this line:

```python
from .image_tools import RemoveBackgroundTool
```

Replace it with:

```python
from .image_tools import RemoveBackgroundTool, EditImagePromptTool
```

Find this line:

```python
    "remove_background": RemoveBackgroundTool().execute,
```

Immediately after it, insert:

```python
    "edit_image_prompt": EditImagePromptTool().execute,
```

Find the `TOOL_TAGS` set's `"remove_background",` entry and add `"edit_image_prompt",` immediately after it, matching the set's existing formatting.

- [ ] **Step 5: Add the schema in `src/tool_schemas.py`**

Find the `remove_background` entry in `FUNCTION_TOOL_SCHEMAS` (its closing `},` is immediately followed by the `ui_control` entry) and insert this new entry immediately after it:

```python
    {
        "type": "function",
        "function": {
            "name": "edit_image_prompt",
            "description": "Edit an image the user uploaded in this chat by describing the change in natural language (e.g. 'add a red hat', 'make the sky sunset-colored'). Returns the edited image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "attachment_id": {"type": "string", "description": "The id of the uploaded image attachment"},
                    "prompt": {"type": "string", "description": "A natural-language description of the edit to make"},
                },
                "required": ["attachment_id", "prompt"],
            },
        },
    },
```

- [ ] **Step 6: Add to `TOOL_SECTIONS` and `_DOMAIN_TOOL_MAP` in `src/agent_loop.py`**

Find the `TOOL_SECTIONS` dict's `"remove_background"` entry and insert this new entry immediately after it (after its closing `""",`):

```python
    "edit_image_prompt": """```edit_image_prompt
{"attachment_id": "<id from an uploaded image>", "prompt": "<describe the edit, e.g. 'add a red hat'>"}
```
Edit an image the user uploaded in chat by describing the change in natural language. Returns the edited image shown inline in your response.""",
```

Find this exact block:

```python
    "desktop": {"launch_app", "find_files", "list_windows", "control_window", "capture_screen",
                "webcam_look", "diagnose_equipment", "remove_background",
                "ingest_equipment_manual", "search_equipment_manual",
                "list_ui_elements", "click_element", "set_element_text", "mouse", "keyboard"},
```

Replace it with:

```python
    "desktop": {"launch_app", "find_files", "list_windows", "control_window", "capture_screen",
                "webcam_look", "diagnose_equipment", "remove_background", "edit_image_prompt",
                "ingest_equipment_manual", "search_equipment_manual",
                "list_ui_elements", "click_element", "set_element_text", "mouse", "keyboard"},
```

- [ ] **Step 7: Add to `BUILTIN_TOOL_DESCRIPTIONS` in `src/tool_index.py`**

Find the `"remove_background"` entry and insert this new entry immediately after it:

```python
    "edit_image_prompt": "Edit an uploaded chat image by describing the change in natural language, returning the edited image.",
```

- [ ] **Step 8: Add the owner-threading dispatch branch in `src/tool_execution.py`**

Find this exact block:

```python
    elif tool == "remove_background":
        # Registry-dispatched (agent_tools.image_tools); owner threaded because
        # the tool resolves the caller's OWN chat attachment —
        # upload_handler.resolve_upload() denies any upload record that has an
        # owner when it is called with owner=None, so without this the tool
        # returns "attachment not found" for every real user in an auth-enabled
        # install (it only appears to work in single-user/no-auth mode).
        # session_id follows the same convention as the other ctx-dependent
        # registry tools, so the Gallery save can be scoped later if needed.
        desc = f"remove_background: {content.split(chr(10))[0][:80]}"
        result = await _direct_fallback(tool, content, session_id=session_id, owner=owner) \
            or {"error": "remove_background: execution failed", "exit_code": 1}
```

Immediately after it, insert:

```python
    elif tool == "edit_image_prompt":
        # Registry-dispatched (agent_tools.image_tools); owner threaded for the
        # exact same reason as remove_background just above — the tool resolves
        # the caller's OWN chat attachment via upload_handler.resolve_upload(),
        # which denies any owned upload record when called with owner=None.
        desc = f"edit_image_prompt: {content.split(chr(10))[0][:80]}"
        result = await _direct_fallback(tool, content, session_id=session_id, owner=owner) \
            or {"error": "edit_image_prompt: execution failed", "exit_code": 1}
```

- [ ] **Step 9: Add the privilege gate in `routes/chat_routes.py`**

Find this exact line:

```python
                disabled_tools.update({"generate_image", "remove_background"})
```

Replace it with:

```python
                disabled_tools.update({"generate_image", "remove_background", "edit_image_prompt"})
```

- [ ] **Step 10: Add to `_PLAN_MODE_KNOWN_MUTATORS` in `src/tool_security.py`**

Find this exact line:

```python
    "generate_image", "edit_image", "trigger_research", "manage_research",
    "remove_background",
```

Replace it with:

```python
    "generate_image", "edit_image", "trigger_research", "manage_research",
    "remove_background", "edit_image_prompt",
```

Do NOT add `edit_image_prompt` to `NON_ADMIN_BLOCKED_TOOLS` — `generate_image` is confirmed absent from that set, and `edit_image_prompt` mirrors it, matching `remove_background`'s precedent (enforced by `test_edit_image_prompt_matches_generate_image_admin_gating`, Step 1).

- [ ] **Step 11: Run tests to verify they pass**

Run: `pytest tests/test_edit_image_prompt_tool.py tests/test_edit_image_prompt_registration.py -v --import-mode=importlib`
Expected: PASS (9 + 5 = 14 tests).

Run: `pytest tests/test_remove_background_tool.py tests/test_remove_background_registration.py -v --import-mode=importlib` to confirm no regression to sub-project 1's tool from the `_default_gallery_saver` signature change (its existing 2-positional-arg call must still work via the new keyword defaults).
Expected: PASS (12 + 5 = 17 tests, matching sub-project 1's final state).

Run: `pytest tests/test_desktop_registration.py -v --import-mode=importlib` to confirm no regression to the pre-existing desktop-tool parity test.

Run: `python -c "import app"` to confirm the app still imports cleanly with the new tool wired in.

- [ ] **Step 12: Commit**

```bash
git add src/agent_tools/image_tools.py src/agent_tools/__init__.py src/tool_schemas.py src/agent_loop.py src/tool_index.py src/tool_execution.py routes/chat_routes.py src/tool_security.py tests/test_edit_image_prompt_tool.py tests/test_edit_image_prompt_registration.py
git commit -m "feat(image-edit): edit_image_prompt builtin agent tool

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
