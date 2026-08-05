# Multi-Persona System sub-project 2: Voice Per Persona — Design Spec

## Context

Sub-project 1 (the general Crew multi-persona system) shipped on `dev`
(`5a047739..4737ec43`): personas have their own name, personality, model/endpoint override, and
tool allowlist, bound at session-creation time. The Multi-Persona initiative was decomposed into
3 ordered sub-projects at design time — core system (done), voice-per-persona (this spec),
memory isolation (deferred, unbuilt). This spec covers sub-project 2 only.

Separately, "Voice Conversation" (a different, earlier sub-project — hands-free mode with
STT+TTS) already shipped and reuses the app's existing TTS service. Today TTS voice is a single
global admin setting; nothing in the TTS path knows about sessions or personas at all. This spec
makes a session's bound persona (if any) override the voice used to speak its replies.

## Current State (as of this spec)

- `TTSService` (`services/tts/tts_service.py`) reads `tts_provider`/`tts_model`/`tts_voice`/
  `tts_speed` fresh from global settings (`data/settings.json`) on every `synthesize()` call. No
  session or persona awareness anywhere in the TTS stack.
- Three providers, three different voice namespaces, but only one is ever active app-wide at a
  time (admin-configured): `local` (Kokoro named voices, e.g. `af_heart`), `endpoint:<id>`
  (OpenAI-compatible `/audio/speech`, a fixed 9-name set: Alloy/Ash/Coral/Echo/Fable/Nova/Onyx/
  Sage/Shimmer), `browser` (client-side Web Speech API — synthesis never touches the server at
  all; the voice name is only used locally by `tts-ai.js`'s `_findBrowserVoice()`).
- The existing global TTS settings UI (`static/js/settings.js` `initTtsSettings` +
  `static/index.html` `#set-ttsVoiceSelect`/`#set-ttsVoiceInput`) already solves "which voices
  are valid for the current provider" by switching between a fixed dropdown (endpoint provider)
  and a free-text input (local/browser providers).
- `POST /api/tts/synthesize` (`routes/tts_routes.py`) takes only `text`+`format` — no session
  context. `GET /api/tts/stats` likewise has no session awareness.
- A session's `crew_member_id` (persisted at creation, `routes/session_routes.py`) is never
  returned by any read endpoint — the frontend currently has no way to learn which persona an
  already-materialized session is bound to.
- The Personal Assistant (`is_default_assistant=True` `CrewMember` row) has its own settings
  panel (`static/js/assistant.js`) in addition to the general Crew panel — both render/save the
  same underlying `CrewMember` fields (personality, model/endpoint, tools) today.

## Goal

A session bound to a persona with a voice override speaks all of its replies — manual play
button, auto-play, and hands-free Voice Conversation mode alike — in that persona's voice,
across all three TTS providers. A session with no persona binding, or a persona with no voice
override, keeps using the current global voice exactly as today. No behavior change for any
existing user who hasn't set a persona voice.

## Architecture

Resolve the effective voice **server-side**, reusing the exact pattern `chat_helpers.py`'s
`extract_preset` already established for persona personality: given a `session_id`, look up the
bound persona via the existing `resolve_crew_binding` helper (owner-scoped, fail-open — no
binding, dangling binding, or any lookup error silently falls through to the global voice) and
substitute its voice if set.

This single resolution point is reused by both TTS code paths:

- **Server-synthesized audio** (`local`/`endpoint:` providers): `TTSService.synthesize()` takes
  an optional `session_id`; when given, it resolves the effective voice before generating audio.
- **Client-synthesized audio** (`browser` provider — the Web Speech API never calls the server
  to synthesize at all): the voice name itself has to reach the client. `GET /api/tts/stats`
  already returns the voice for `tts-ai.js`'s `checkAvailability()` to cache as `browserVoice`;
  extending that endpoint to accept the same optional `session_id` and resolve the same way
  means the frontend needs no new endpoint, and `_findBrowserVoice()` needs no changes at all.

Considered and rejected: resolving the voice client-side (fetch the persona's voice whenever a
session is persona-bound, pass it as an explicit override into every TTS call). Rejected because
it duplicates persona-resolution logic in JS instead of reusing the tested Python helper every
other persona-override feature already shares, and scatters "remember to fetch persona data"
across every call site — the exact class of gap the sub-project 1 review chain spent three fix
waves closing (a fix or feature landing in one of two panels/paths but not the other).

Voice Conversation (hands-free mode) has no TTS calls of its own — it already funnels through
the same shared auto-play/manual-play code paths every other chat surface uses. Threading
`session_id` into those shared call sites once means hands-free mode benefits automatically,
with no hands-free-specific code.

## Data Model

New nullable column, same idempotent-migration pattern as `CrewMember.endpoint_id`:

```python
tts_voice = Column(String, nullable=True)   # provider-specific voice name; None = use the global default
```

No validation against a fixed list at write time — unlike `endpoint_id` (which references a
real `ModelEndpoint` row and would 400 on garbage), a voice name is just an opaque string handed
to whichever provider happens to be configured, exactly like the existing global `tts_voice`
setting already is. A persona's voice becoming stale after an admin switches TTS provider is the
same failure mode the global setting already has today (a wrong/unrecognized voice name is the
provider's problem to reject or ignore, not this app's), so it is explicitly not hardened beyond
what the global setting already tolerates.

`crew_to_dict()` includes `tts_voice` in its output — the one shared dict both `crew.js` and
`assistant.js` read.

## API Changes

- `POST /api/tts/synthesize` — `TTSRequest` gains an optional `session_id: str | None = None`.
  When present, threaded into `TTSService.synthesize(text, session_id=...)`.
- `GET /api/tts/stats` — gains an optional `session_id` query param, threaded the same way into
  `TTSService.get_stats(session_id=...)`. Response shape unchanged (still just `stats.voice`
  reflecting whichever voice — persona or global — is effective).
- `routes/crew_routes.py` — `create_crew`/`update_crew` accept a `tts_voice` field, coerced via
  the same `_s()`-style "non-`None`, non-`str` → `str(x)`" pattern already applied to
  `avatar`/`personality`/`model`/`greeting` in both handlers. "No server-side format validation"
  (the schema note above) means no check against a fixed voice-name list or a registered
  resource — it does **not** mean skipping the type coercion every other free-text field on this
  model already gets. This distinction matters: sub-project 1's review chain spent three fix
  waves closing exactly this class of gap (a value that passes a coarse check but still isn't
  the right *type* by the time it reaches a fragile operation), so `tts_voice` must not
  reintroduce it.

## Frontend Changes

- `static/js/tts-ai.js` — `checkAvailability()` and `synthesize()` both need the current
  session id to pass through to the backend. Exact call-site wiring (where the auto-play queue
  and manual play button currently invoke these methods, and how the current session id reaches
  them) is an implementation-plan-level detail, not pinned down here.
- `static/js/crew.js` + `static/js/assistant.js` — both gain the same dual-mode voice picker
  the global TTS setting already uses (dropdown for `endpoint:` provider, free-text otherwise),
  added to the persona/Assistant edit forms alongside the existing model/endpoint picker.
- `static/index.html` — new form fields for both panels, mirroring the existing
  `#set-ttsVoiceSelect`/`#set-ttsVoiceInput` pair's structure.

## Error Handling

Every failure mode here already has an established precedent from sub-project 1 and is reused,
not reinvented:

- No `session_id` passed to a TTS call → today's exact behavior, unchanged (global voice).
- `session_id` passed but the session has no persona binding → falls through to global voice.
- Persona bound but its `tts_voice` is empty/unset → falls through to global voice.
- `resolve_crew_binding` lookup fails for any reason (dangling reference, DB error) → fail-open
  to global voice, never an error surfaced to the TTS caller.
- A persona's `tts_voice` is set but invalid/unrecognized for whatever provider is currently
  configured → passed through as-is; the provider's own error handling applies, same as today's
  global setting.

## Testing

- `TTSService.synthesize`/`get_stats` with `session_id`: unbound session, dangling persona
  reference, persona with empty `tts_voice`, persona with a real `tts_voice` (asserting it wins
  over the global setting) — mirrors the existing personality/tool-policy fail-open test battery
  exactly (real DB rows via the ORM, not mocks).
- Route tests for the new optional `session_id` param on both `/api/tts/synthesize` and
  `/api/tts/stats`.
- Frontend: source-presence tests (`node --check` + regex extraction, matching this codebase's
  established `crew.js`/`assistant.js` test style) for the two new voice-picker fields and for
  the session-id threading into `tts-ai.js`'s calls.
- Explicitly out of scope for automated testing: actually hearing a persona's reply spoken in a
  different voice. Manual verification, same as every prior audio/vision/image quality check in
  this project has been.

## Out of Scope

- Per-persona STT (speech-to-text) behavior — STT is always the human user's own voice; "voice"
  here means TTS output only.
- Validating a persona's voice name against the currently configured provider's actual
  supported-voice list.
- A voice-preview/test button in the persona edit form.
- Sub-project 3 (memory isolation) — remains a separate, unbuilt, deferred sub-project.
