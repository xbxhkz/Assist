# Hands-Free Voice Conversation Design

**Goal:** A "voice mode" toggle that turns the existing speech-to-text and
text-to-speech features into a hands-free conversation loop: open the mic,
detect when you stop talking, auto-submit the transcript, let the AI reply be
spoken aloud, then re-open the mic — repeat until you stop.

**Scope:** The conversation loop only. This is sub-project #1 of a larger
"voice" initiative. **Wake word** ("Hey Assist" always-on trigger) and
**barge-in** (talking over the AI to interrupt) are explicitly out of scope and
become their own later sub-projects. Because the mic is closed whenever the AI
is speaking, v1 needs no acoustic echo cancellation.

---

## Background — what already exists

Voice mode is orchestration over mature, shipped pieces; it builds almost
nothing new on the audio side.

- **STT** — `services/stt/stt_service.py` + `routes/stt_routes.py`: multi-provider
  (faster-whisper local / OpenAI-compatible endpoint / browser Web Speech),
  exposed at `POST /api/stt/transcribe` and `GET /api/stt/stats`.
- **STT capture UI** — `static/js/voiceRecorder.js`: **push-to-talk** today
  (tap to record, tap to stop → transcribe → drop text into `#message`; user
  still sends manually). It manages its own `getUserMedia`/`MediaRecorder`.
- **TTS** — `services/tts/tts_service.py` + `routes/tts_routes.py`: Kokoro-82M
  (local GPU) / endpoint / browser, at `POST /api/tts/synthesize` and
  `GET /api/tts/stats`.
- **TTS playback UI** — `static/js/tts-ai.js` (`window.aiTTSManager`): has an
  `autoPlay` flag and **streaming sentence-by-sentence** synthesis
  (`streamingStart/streamingUpdate/streamingEnd`, a play queue, `isPlaying` /
  `_processing`). When `autoPlay` is on, AI replies are already spoken as they
  stream.
- **Chat send + stream hooks** — `static/js/chat.js` already drives streaming
  TTS: `streamingStart()` at [chat.js:1289](../../static/js/chat.js#L1289) when
  `aiTTSManager.autoPlay && available`, `streamingUpdate()` during the stream,
  and `streamingEnd(accumulated)` at stream completion (~
  [chat.js:2939](../../static/js/chat.js#L2939)).

**The only gap:** nothing ties capture → transcribe → send → spoken reply →
re-listen into a loop. That is this feature.

## Architecture (client-side orchestrator, no server changes)

Two new frontend files plus two tiny event hooks in existing files. **No Python
/ backend changes** — all existing endpoints are reused as-is.

```
composer "voice mode" toggle
        │ on
        ▼
voiceConversation.js  (DOM + audio wiring; owns the loop)
  getUserMedia → MediaRecorder + Web Audio AnalyserNode (RMS)
        │  uses ↓ pure logic
  voiceConversationCore.js  (nextState reducer + turn detector) ← unit-tested
        │
  turn ends → POST /api/stt/transcribe → text
        │
  auto-submit via the composer's EXISTING send handler
        │  (normal pipeline: current model/session/tools/streaming)
        ▼
  AI streams → existing streaming TTS speaks it (aiTTSManager, autoPlay on)
        │
  listens for CustomEvents: 'assist:reply-complete' + 'assist:tts-idle'
        ▼
  re-open mic (LISTENING)         ── loop ──
```

### New unit 1 — `static/js/voiceConversationCore.js` (pure, node-testable)

No DOM / `window` / `document` access at import time, so it imports cleanly
under `node --input-type=module` (the repo's JS-test pattern). Exports:

- `nextState(state, event) -> state` — pure state-machine reducer.
  States: `idle`, `listening`, `transcribing`, `thinking`, `speaking`.
  Events: `TOGGLE_ON`, `SPEECH_END`, `TEXT`, `EMPTY`, `REPLY_SPOKEN`
  (reply done, TTS pending), `REPLY_SILENT` (reply done, no TTS), `TTS_IDLE`,
  `SILENCE_TIMEOUT`, `STOP`, `ERROR`. Transition table:
  - `idle` + `TOGGLE_ON` → `listening`
  - `listening` + `SPEECH_END` → `transcribing`
  - `listening` + `SILENCE_TIMEOUT` → `idle`
  - `transcribing` + `EMPTY` → `listening`
  - `transcribing` + `TEXT` → `thinking`
  - `thinking` + `REPLY_SPOKEN` → `speaking`
  - `thinking` + `REPLY_SILENT` → `listening`
  - `speaking` + `TTS_IDLE` → `listening`
  - any state + `STOP` → `idle`
  - `transcribing`/`thinking`/`speaking` + `ERROR` → `listening` (skip the turn)
  - `listening` + `ERROR` (mic failure) → `idle`
  - any other (state, event) → unchanged (defensive).

- `createTurnDetector({ threshold = 0.015, hangoverMs = 1200, minSpeechMs = 300 })`
  → object with `observe(rms, tMs) -> 'idle' | 'speech' | 'ended'`.
  Energy VAD logic: `rms >= threshold` marks speech; once at least
  `minSpeechMs` of speech has occurred (so a single click/blip can't end a turn)
  and `rms` then stays `< threshold` continuously for `hangoverMs`, it returns
  `'ended'` (fire once). Deterministic given an (rms, time) sequence — the unit
  under test.

### New unit 2 — `static/js/voiceConversation.js` (DOM + audio wiring)

Owns all browser side effects; imports the core. Responsibilities:
- **Toggle + status pill** — a voice-mode button in the composer toolbar
  (beside the existing mic/send); a status pill it builds via
  `createElement`/`textContent` showing the state label
  (`Listening… / Thinking… / Speaking…`), a live mic-level meter (driven by the
  same RMS), and a Stop button. CSP-safe (no `innerHTML`-with-data, no inline
  handlers).
- **Capture** — its own `getUserMedia({audio})` → `MediaRecorder` for the clip
  **plus** an `AudioContext` `AnalyserNode` tapping the same stream for RMS. It
  does not reuse `voiceRecorder.startRecording` (that auto-inserts and doesn't
  expose the analyser tap); it MAY import a shared `transcribeOnServer` helper
  if `voiceRecorder.js` is refactored to export one, else it POSTs
  `/api/stt/transcribe` itself.
- **Drive the loop** — feed RMS to the turn detector each analyser frame; on
  `'ended'`, stop the recorder, POST the clip, map result to `TEXT`/`EMPTY`,
  and on `TEXT` set `#message` and trigger the composer's existing send handler
  (no parallel send path).
- **Know when the AI is done** — listen for two `CustomEvent`s (see hooks
  below) rather than polling: `assist:reply-complete` (LLM turn finished) and
  `assist:tts-idle` (TTS queue drained). If spoken replies are active, the turn
  ends on `assist:tts-idle`; if not, on `assist:reply-complete`.
- **Lifecycle** — enable `aiTTSManager.autoPlay` on entry (remember prior
  value, restore on exit); a 25s no-speech watchdog fires `SILENCE_TIMEOUT`;
  Stop / Esc / a manual typed send all fire `STOP`, which stops the recorder,
  releases the mic track, closes the `AudioContext`, and cancels pending TTS
  (`aiTTSManager.stop()`).

### Integration hooks (minimal, in existing files)

- `static/js/chat.js` — at the existing stream-completion point (where
  `streamingEnd` is called) dispatch
  `window.dispatchEvent(new CustomEvent('assist:reply-complete'))`. One line;
  no restructuring of the send function.
- `static/js/tts-ai.js` — when the play queue drains (end of `_processQueue`)
  and in `stop()`, dispatch `window.dispatchEvent(new CustomEvent('assist:tts-idle'))`.
  Guarded so it fires once per idle transition.

## Turn detection (energy + silence timeout)

Chosen approach: Web Audio `AnalyserNode` RMS with a silence hangover — no new
dependencies, fully offline, works with the local Whisper path. Defaults
(tunable constants): `threshold ≈ 0.015` (normalized RMS), `hangoverMs = 1200`,
`minSpeechMs = 300`. A `SILENCE_TIMEOUT` of ~25s of continuous non-speech exits
voice mode so the mic never stays hot indefinitely. Known limitation: a
persistently noisy room can hold RMS above threshold; the thresholds are
constants tuned in live-verify and can later be surfaced as settings (out of
scope here).

## Provider rules & graceful degradation

- **Entry requires STT** available: on toggle-on, check `GET /api/stt/stats`;
  if `provider == "disabled"` / unavailable, show a toast ("Enable
  Speech-to-Text in settings") and stay `idle`. You can't converse if it can't
  hear you.
- **Spoken replies auto-enabled:** entering voice mode sets
  `aiTTSManager.autoPlay = true` for the session (restored on exit). If no TTS
  provider is configured (`aiTTSManager.available` is false), the loop still
  runs but replies are **text-only**, with a one-time notice; the turn then
  ends on `assist:reply-complete` instead of `assist:tts-idle`.
- **Auto-submit is immediate** — no confirm/cancel window (it would defeat
  hands-free). The transcript is visible in the sent user bubble like any
  message.
- **Not admin-gated.** Voice mode only orchestrates existing user-level STT/TTS
  and chat; it adds no server capability or new security surface. Mic access is
  browser-permissioned.

## Error handling

- Mic denied / insecure context → toast, revert to `idle`. (localhost is a
  secure context, so this is a non-issue inside the app; the check mirrors
  `voiceRecorder.js`.)
- STT failure or empty transcript on a turn → `EMPTY`/`ERROR` → return to
  `listening` (skip the turn; never crash the loop).
- TTS failure → reply already shown as text; treat as `REPLY_SILENT` and
  continue.
- Toggling off mid-turn → `STOP`: stop recorder, release the mic track, close
  the `AudioContext`, cancel pending TTS.

## Testing

- **Unit (node-driven pytest, the repo pattern —
  `tests/test_emoji_shortcodes_js.py` etc.):** new
  `tests/test_voice_conversation_core_js.py` imports
  `voiceConversationCore.js` via `node --input-type=module`, `skipif` no node.
  - `nextState`: every transition in the table above, plus that unknown
    `(state,event)` pairs are unchanged and `STOP` from every state → `idle`.
  - `createTurnDetector`: a below→above→below RMS sequence returns `'ended'`
    exactly once after `hangoverMs`; a sub-`minSpeechMs` blip never ends a turn;
    continuous speech never ends.
- **Live-verify (in the running app):** toggle voice mode → speak → see it
  transcribe, auto-send, speak the reply, then re-listen; confirm the mic
  closes while the AI talks (no echo/self-trigger); Stop and Esc both exit and
  release the mic; the 25s silence watchdog exits; a noisy-room sanity check of
  the threshold.

## Non-goals

- Wake word / always-on listening (separate sub-project).
- Barge-in / interrupting the AI by speaking (separate sub-project; would add
  echo-cancellation complexity).
- A full-screen "orb" voice UI (v1 uses the inline status pill).
- Model-based VAD (Silero/ONNX), server-side conversation sessions, or
  WebSocket audio streaming.
- New STT/TTS providers or models — reuse whatever is configured.
- Exposing VAD thresholds as user settings (constants for v1).
