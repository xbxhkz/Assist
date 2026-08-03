# Multi-Persona System — "Crew" (Sub-project 1) — Design

**Goal:** Generalize the existing singleton "Personal Assistant" feature into a real multi-persona system:
create multiple named AI personas (personality/system-prompt, avatar, model/endpoint, tool access), and start a
new chat bound to any of them. Sub-project 1 of a 3-part initiative (voice-per-persona and per-persona memory
isolation follow as sub-projects 2 and 3) picked from the user's personal feature list ("AI Personality
Builder").

## Existing infrastructure this reuses (found during exploration, not built here)

- **`CrewMember`** (`core/database.py:546`) — already has exactly the shape this needs: `name`, `avatar`,
  `user_name` (what it calls the user), `personality` (system prompt), `model`, `endpoint_url`, `greeting`,
  `enabled_tools` (JSON array or `"all"`), `session_id`, `is_active`, `sort_order`, `is_default_assistant`,
  `owner`. No new columns needed for this sub-project.
- **`routes/assistant_routes.py`** — the ONLY existing code path that uses `CrewMember`, and only for the
  singleton (`is_default_assistant=True`) row. Not admin-gated — the Assistant is per-user/owner-scoped. Its
  `_crew_to_dict()` helper (line 46) already converts a `CrewMember` row to the exact JSON shape a general list
  endpoint needs.
- **`build_context_preface(..., preset_system_prompt=, character_name=, ...)`** (`src/chat_processor.py:198`) —
  the existing text-preset system (`PresetManager`) already injects a custom system prompt + display name
  through these two parameters (see the call site in `routes/chat_helpers.py` around lines 340 and 716-723). A
  persona's `personality`/`name` slot into these same two parameters — no new injection mechanism needed.
- **`build_effective_tool_policy(disabled_tools=..., last_user_message=...)`** (`src/tool_policy.py:174`) —
  composes a *denylist* for the current turn. `CrewMember.enabled_tools` is an *allowlist*
  (`"JSON array or 'all'"`), so wiring it in means computing `known_tool_names() - set(enabled_tools)` and
  unioning the result into the `disabled_tools` passed to this function — no new enforcement mechanism needed.
- **`create_session()`** (`routes/session_routes.py:320`) — already resolves an `endpoint_id` Form field into
  `model`/`endpoint_url` for a new session (see the `endpoint_id` branch). A `crew_member_id` Form field follows
  the identical resolve-then-default pattern.

## Scope decisions (baked in)

- **Reuse `CrewMember` as-is.** No new columns this sub-project — voice (sub-project 2) and memory-scope
  (sub-project 3) additions happen later, in their own sub-projects.
- **Owner-scoped, not admin-gated.** Matches the existing Assistant feature's gating exactly — every
  authenticated user can create and use their own personas, not just admins. This is a deliberate departure
  from most other panels built this session (Training GUI, Image Dataset, etc.), which ARE admin-gated — those
  are power-user/infrastructure tools; a custom assistant persona is a per-user personalization feature, same
  category as the Assistant singleton it generalizes.
- **One persona per SESSION, bound at creation — no mid-conversation switching.** Matches
  `Session.crew_member_id`'s existing FK design exactly (confirmed with the user). "Switch personas" means
  "start a new chat with a different persona," not "change the active persona of an ongoing conversation."
- **The existing Assistant (`is_default_assistant=True`) appears in the same general list**, not as a separate
  concept. Its check-in/timezone extras stay in `assistant_routes.py`, entirely untouched by this sub-project.
  The general CRUD endpoint this sub-project adds manages only the fields both share (name, avatar, personality,
  model, endpoint_url, greeting, enabled_tools, is_active, sort_order) — timezone/check-ins are Assistant-only
  and stay where they are.
- **Deleting the default Assistant is blocked** through the new general endpoint — the Assistant feature
  assumes exactly one `is_default_assistant=True` row always exists per owner; a generic delete would break
  that invariant. Deleting any other (non-default) persona is unrestricted.
- **Precedence when a persona-bound session's turn would otherwise also resolve a separately-selected text
  preset** (the existing `PresetManager` "Code Analyze"/"Brainstorm"/etc. picker): **the persona wins.** A
  persona-bound session's `personality`/`name` fully replace what the preset selector would have contributed for
  that turn — the persona already defines the assistant's voice for this conversation, and letting two
  system-prompt sources compete for the same slot would be confusing. If the linked `CrewMember` has since been
  deleted, fall back silently to normal preset-only behavior (never raise into the chat request over a
  dangling reference).
- **A malformed/missing `enabled_tools` fails OPEN, not closed** (see Error Handling) — this is a self-service
  customization within one user's own tool access, not a privilege boundary between users/roles, so a parsing
  hiccup should never silently lock the user out of their own tools mid-conversation.

## Architecture

**Backend — `routes/crew_routes.py` (new):**
- `GET /api/crew` — list all crew members for the current owner (including the default Assistant), each via a
  shared `_crew_to_dict`-shaped dict.
- `POST /api/crew` — create a new persona (`name` required; `avatar`/`personality`/`model`/`endpoint_url`/
  `greeting`/`enabled_tools` optional), owned by the current user.
- `PATCH /api/crew/{id}` — update the shared fields (owner-scoped; 404 if not found or not owned).
- `DELETE /api/crew/{id}` — delete (owner-scoped; 400 if the target is the default Assistant).
- `GET /api/crew/tool-names` — returns `{"tools": [...]}`, the sorted output of `tool_policy.py`'s existing
  `known_tool_names()`. **New, not previously exposed** — checked during exploration and confirmed no endpoint
  currently surfaces this list to the frontend; this is what the create/edit form's tool-enable checklist reads
  from.

**Refactor-as-you-go:** promote `_crew_to_dict()` out of `routes/assistant_routes.py` into a small shared module
(`src/crew_helpers.py`) so both route files import the same conversion instead of duplicating it.

**Session creation (`routes/session_routes.py`, modify `create_session`):** add an optional
`crew_member_id: str = Form(None)`. When present: resolve the row (owner-scoped; 400 if not found or not
owned), default `model`/`endpoint_url` from it when the request didn't explicitly supply those, and store
`crew_member_id` on the newly created `Session` row (the column already exists).

**Chat-turn wiring (`routes/chat_helpers.py`, at the existing `PresetInfo`/`build_context_preface` call site):**
when the session has a `crew_member_id`, load that `CrewMember`; if found and its `personality` is non-empty,
pass its `personality`/`name` as `preset_system_prompt`/`character_name` instead of whatever the separately-
selected preset would have contributed (see precedence rule above). A deleted/missing linked persona falls back
to normal preset-only behavior.

**Tool-turn wiring (wherever `build_effective_tool_policy(disabled_tools=...)` is called for a chat turn):**
when the session has a `crew_member_id` and that persona's `enabled_tools` is present, non-empty, and not the
literal string `"all"`, union `known_tool_names() - set(enabled_tools)` into the `disabled_tools` passed
through. Any other case (missing, `"all"`, empty list, unparseable JSON, persona not found) adds no extra
restriction — see the fail-open rationale above.

**Frontend:**
- New `#crew-modal` + `static/js/crew.js` (ES module, mirrors this session's established panel-controller
  shape: module-scope `$`/`esc`/`api` helpers, `Modals.register`). One modal with two views toggled in place
  (mirrors `imageDataset.js`'s list↔working-set toggle pattern): a card **grid** view (avatar, name, a
  truncated personality preview, a primary "New Chat" button, and Edit/Delete) and a **create/edit form** view
  (name, avatar picker/URL, personality textarea, model/endpoint fields reusing the existing model-picker
  component, a tool-enable checklist populated from the new `GET /api/crew/tool-names`, greeting).
- `#rail-crew` / `#tool-crew-btn` — **NOT admin-gated** (visible to every authenticated user), unlike every
  other panel shipped this session.
- "New Chat" on a card calls the existing session-creation flow with `crew_member_id` set, then opens that
  session exactly like any other new chat.
- The existing Assistant entry point (`static/js/assistant.js`) is untouched — Crew is for browsing/creating
  every persona (the Assistant included, read-only as far as its check-in settings go from this panel), not a
  replacement for the Assistant's own dedicated settings modal.

## Error handling

- Owner-scoping everywhere via the existing `owner_filter`/`get_current_user` helpers — never a client-supplied
  owner field, matching the pattern already used by `ModelEndpoint`, sessions, and every other owned row in this
  app.
- `PATCH`/`DELETE` on a persona id that doesn't exist or isn't owned by the caller returns 404, not a silent
  no-op.
- Malformed `enabled_tools` JSON on a `CrewMember` row is treated as "all" (fail-open, see rationale above) —
  mirrors the existing `_crew_to_dict()`'s own `try/except` around `json.loads(c.enabled_tools)`.
- A session whose linked persona has since been deleted degrades silently to normal (non-persona) chat
  behavior — never raises into an in-progress conversation.

## Testing

- `tests/test_crew_routes.py` — CRUD, owner-scoping (a second owner's personas are invisible and unowned-id
  operations 404), malformed-`enabled_tools` handling, delete-blocked-on-default-assistant, and that the
  refactored `_crew_to_dict` still produces the exact shape `assistant_routes.py`'s existing tests expect (no
  regression on the shipped Assistant feature).
- A pure-function test for the allowlist→denylist conversion (`known_tool_names() - enabled_tools`), covering
  `"all"`, empty, missing, and malformed-JSON inputs.
- `tests/test_crew_ui.py` — HTML-id-presence, JS-wiring-presence, `node --check` syntax, mirroring every other
  frontend sub-project this session's established test shape.
- Manual GUI verification (persona creation, "New Chat as persona" actually changing the assistant's behavior
  in a live conversation) is owed by the user, matching every other frontend piece shipped this session.

## Non-goals (this sub-project)

- Voice per-persona (sub-project 2).
- Per-persona memory isolation (sub-project 3).
- Mid-conversation persona switching (explicitly rejected — one persona per session, bound at creation).
- Admin-only gating (explicitly NOT admin-gated).
- Any change to the Assistant's existing check-in/timezone functionality.
- A tool-permission model beyond the existing allowlist (`enabled_tools`) — no new "sub-permissions" concept.
