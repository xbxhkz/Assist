# Hands-Free Voice Conversation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "voice mode" toggle that turns the existing STT + streaming-TTS into a hands-free conversation loop: mic opens, detects when you stop talking, auto-submits the transcript, the AI reply is spoken, then the mic re-opens — until you stop.

**Architecture:** A client-side orchestrator (`voiceConversation.js`) drives a state machine + energy-VAD (Web Audio `AnalyserNode` RMS), reusing the existing `POST /api/stt/transcribe`, the normal chat send (`.send-btn` click), and `window.aiTTSManager` streaming TTS. All pure logic (state reducer + turn detector) lives in `voiceConversationCore.js` and is unit-tested via `node`-from-pytest. Two one-line `CustomEvent` hooks in `chat.js`/`tts-ai.js` tell the loop when a turn's reply and speech are done. No backend/Python changes.

**Tech Stack:** Vanilla ES-module JS (browser), Web Audio API, MediaRecorder; pytest driving `node --input-type=module` for the pure core (repo pattern: `tests/test_emoji_shortcodes_js.py`).

## Global Constraints

- **No backend/Python changes.** Reuse `POST /api/stt/transcribe`, `GET /api/stt/stats`, and `window.aiTTSManager`; never build a parallel send path (trigger the existing `.send-btn`).
- **CSP-safe UI only:** `createElement` + `addEventListener` + `textContent` (+ canvas/DOM); NO `innerHTML`-with-data, NO inline `on*=` handlers. Module scripts load from `'self'` (like `voiceRecorder.js`). Inline `style=` is allowed (`style-src 'self' 'unsafe-inline'`).
- **Mic is not captured/analyzed while the AI is thinking or speaking** (loop is sequential) — so no echo handling in v1.
- **Entering voice mode requires STT available** (`GET /api/stt/stats` provider != `"disabled"`), else a toast and stay idle.
- **Voice mode sets `aiTTSManager.autoPlay = true` on entry and restores the prior value on exit.**
- **Not admin-gated** (orchestrates existing user-level features only).
- VAD starting constants (tunable, tuned in live-verify): `threshold 0.015`, `hangoverMs 1200`, `minSpeechMs 300`; no-speech auto-exit `25000` ms.
- Pure core file (`voiceConversationCore.js`) must have **no DOM/`window`/`document` access at import time** so it imports under `node`.
- Every pytest run uses `--import-mode=importlib`. JS tests `skipif` when `node` is not on PATH.
- Stage specific files (never `git add -A`). Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Pure core — state reducer + energy-VAD turn detector

**Files:**
- Create: `static/js/voiceConversationCore.js`
- Test: `tests/test_voice_conversation_core_js.py`

**Interfaces:**
- Produces (consumed by Task 2):
  - `export const STATES` — `{ IDLE:'idle', LISTENING:'listening', TRANSCRIBING:'transcribing', THINKING:'thinking', SPEAKING:'speaking' }`.
  - `export function nextState(state: string, event: string) -> string` — pure reducer.
  - `export function createTurnDetector({threshold?, hangoverMs?, minSpeechMs?}) -> { observe(rms: number, tMs: number) -> 'idle'|'speech'|'ended', reset() }`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_voice_conversation_core_js.py`:

```python
"""Pin the pure voice-conversation core (state reducer + energy VAD).

Driven through `node --input-type=module` so we exercise the real JS without a
Vitest/Jest setup (same approach as test_emoji_shortcodes_js.py). Skips when
`node` is not installed rather than failing.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_CORE = _REPO / "static" / "js" / "voiceConversationCore.js"
_HAS_NODE = shutil.which("node") is not None


def _run(js: str) -> str:
    proc = subprocess.run(
        ["node", "--input-type=module"],
        input=js, capture_output=True, text=True, cwd=str(_REPO), timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@pytest.mark.skipif(not _HAS_NODE, reason="node binary not on PATH")
def test_reducer_happy_path_transitions():
    js = f"""
    import {{ nextState }} from '{_CORE.as_posix()}';
    const seq = [
      nextState('idle','TOGGLE_ON'),
      nextState('listening','SPEECH_END'),
      nextState('transcribing','TEXT'),
      nextState('thinking','REPLY_SPOKEN'),
      nextState('speaking','TTS_IDLE'),
      nextState('transcribing','EMPTY'),
      nextState('thinking','REPLY_SILENT'),
      nextState('listening','SILENCE_TIMEOUT'),
    ];
    console.log(JSON.stringify(seq));
    """
    assert json.loads(_run(js)) == [
        'listening', 'transcribing', 'thinking', 'speaking',
        'listening', 'listening', 'listening', 'idle',
    ]


@pytest.mark.skipif(not _HAS_NODE, reason="node binary not on PATH")
def test_reducer_stop_from_any_state_and_noops():
    js = f"""
    import {{ nextState }} from '{_CORE.as_posix()}';
    const states = ['idle','listening','transcribing','thinking','speaking'];
    const stops = states.map(s => nextState(s,'STOP'));
    const out = {{
      stops,
      // wrong-state / unknown events are no-ops
      wrongState: nextState('idle','SPEECH_END'),
      unknown: nextState('listening','WAT'),
      // ERROR: listening->idle, mid-turn->listening
      errListening: nextState('listening','ERROR'),
      errThinking: nextState('thinking','ERROR'),
    }};
    console.log(JSON.stringify(out));
    """
    out = json.loads(_run(js))
    assert out["stops"] == ['idle', 'idle', 'idle', 'idle', 'idle']
    assert out["wrongState"] == 'idle'
    assert out["unknown"] == 'listening'
    assert out["errListening"] == 'idle'
    assert out["errThinking"] == 'listening'


@pytest.mark.skipif(not _HAS_NODE, reason="node binary not on PATH")
def test_turn_detector_speech_then_silence_ends_once():
    # 400ms speech (>minSpeechMs), then 1200ms silence (>=hangoverMs) at 50ms steps.
    js = f"""
    import {{ createTurnDetector }} from '{_CORE.as_posix()}';
    const d = createTurnDetector({{ threshold: 0.02, hangoverMs: 1000, minSpeechMs: 300 }});
    let t = 0; const out = [];
    for (let i = 0; i < 8; i++)  {{ out.push(d.observe(0.10, t)); t += 50; }}  // speech
    for (let i = 0; i < 24; i++) {{ out.push(d.observe(0.00, t)); t += 50; }}  // silence
    console.log(JSON.stringify(out));
    """
    out = json.loads(_run(js))
    assert out.count('ended') == 1
    # first 'ended' only after >=1000ms of trailing silence (>=20 silence steps in)
    assert out.index('ended') >= 8 + 20


@pytest.mark.skipif(not _HAS_NODE, reason="node binary not on PATH")
def test_turn_detector_short_blip_never_ends():
    # 100ms blip (<minSpeechMs) then long silence must NOT end a turn.
    js = f"""
    import {{ createTurnDetector }} from '{_CORE.as_posix()}';
    const d = createTurnDetector({{ threshold: 0.02, hangoverMs: 1000, minSpeechMs: 300 }});
    let t = 0; const out = [];
    for (let i = 0; i < 2; i++)  {{ out.push(d.observe(0.10, t)); t += 50; }}
    for (let i = 0; i < 40; i++) {{ out.push(d.observe(0.00, t)); t += 50; }}
    console.log(JSON.stringify(out));
    """
    assert 'ended' not in json.loads(_run(js))


@pytest.mark.skipif(not _HAS_NODE, reason="node binary not on PATH")
def test_turn_detector_continuous_speech_never_ends():
    js = f"""
    import {{ createTurnDetector }} from '{_CORE.as_posix()}';
    const d = createTurnDetector({{ threshold: 0.02, hangoverMs: 1000, minSpeechMs: 300 }});
    let t = 0; const out = [];
    for (let i = 0; i < 60; i++) {{ out.push(d.observe(0.10, t)); t += 50; }}
    console.log(JSON.stringify(out));
    """
    assert 'ended' not in json.loads(_run(js))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_voice_conversation_core_js.py --import-mode=importlib -q`
Expected: FAIL (module `voiceConversationCore.js` does not exist → node import error / non-zero exit).

- [ ] **Step 3: Write the implementation**

Create `static/js/voiceConversationCore.js`:

```javascript
// static/js/voiceConversationCore.js
// Pure, browser-free logic for hands-free voice mode: the state-machine reducer
// and the energy-VAD turn detector. No DOM/window access at import time, so it
// imports cleanly under `node --input-type=module` for unit tests.

export const STATES = Object.freeze({
  IDLE: 'idle',
  LISTENING: 'listening',
  TRANSCRIBING: 'transcribing',
  THINKING: 'thinking',
  SPEAKING: 'speaking',
});

// Pure reducer. Any (state, event) pair not explicitly handled returns the same
// state, so stray or duplicate events are safe no-ops.
export function nextState(state, event) {
  switch (event) {
    case 'STOP':
      return STATES.IDLE; // stop from any state
    case 'TOGGLE_ON':
      return state === STATES.IDLE ? STATES.LISTENING : state;
    case 'SPEECH_END':
      return state === STATES.LISTENING ? STATES.TRANSCRIBING : state;
    case 'SILENCE_TIMEOUT':
      return state === STATES.LISTENING ? STATES.IDLE : state;
    case 'EMPTY':
      return state === STATES.TRANSCRIBING ? STATES.LISTENING : state;
    case 'TEXT':
      return state === STATES.TRANSCRIBING ? STATES.THINKING : state;
    case 'REPLY_SPOKEN':
      return state === STATES.THINKING ? STATES.SPEAKING : state;
    case 'REPLY_SILENT':
      return state === STATES.THINKING ? STATES.LISTENING : state;
    case 'TTS_IDLE':
      return state === STATES.SPEAKING ? STATES.LISTENING : state;
    case 'ERROR':
      // mic failure while listening drops to idle; a transient error mid-turn
      // skips the turn and returns to listening.
      if (state === STATES.LISTENING) return STATES.IDLE;
      if (state === STATES.TRANSCRIBING || state === STATES.THINKING || state === STATES.SPEAKING) {
        return STATES.LISTENING;
      }
      return state;
    default:
      return state;
  }
}

// Energy VAD. Feed it (rms, tMs) samples; returns 'idle' | 'speech' | 'ended'.
// Returns 'ended' exactly once (then 'idle' until reset()), after at least
// minSpeechMs of speech has accumulated and rms has then stayed below threshold
// continuously for hangoverMs. The caller stops observing on the first 'ended'.
export function createTurnDetector(opts) {
  opts = opts || {};
  const threshold = opts.threshold != null ? opts.threshold : 0.015;
  const hangoverMs = opts.hangoverMs != null ? opts.hangoverMs : 1200;
  const minSpeechMs = opts.minSpeechMs != null ? opts.minSpeechMs : 300;

  let speechMs = 0;         // cumulative time rms >= threshold
  let silenceStart = null;  // tMs when the current below-threshold run began
  let lastT = null;
  let ended = false;

  return {
    observe(rms, tMs) {
      if (ended) return 'idle'; // already fired once; caller should have stopped
      const dt = lastT == null ? 0 : Math.max(0, tMs - lastT);
      lastT = tMs;

      if (rms >= threshold) {
        speechMs += dt;
        silenceStart = null;
        return 'speech';
      }
      // below threshold
      if (silenceStart == null) silenceStart = tMs;
      if (speechMs >= minSpeechMs && (tMs - silenceStart) >= hangoverMs) {
        ended = true;
        return 'ended';
      }
      // during trailing silence (post-speech, pre-hangover) report the turn as
      // still ongoing; before any real speech, report idle.
      return speechMs >= minSpeechMs ? 'speech' : 'idle';
    },
    reset() {
      speechMs = 0;
      silenceStart = null;
      lastT = null;
      ended = false;
    },
  };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_voice_conversation_core_js.py --import-mode=importlib -q`
Expected: PASS (5 passed), or SKIPPED if `node` is not installed.

- [ ] **Step 5: Commit**

```bash
git add static/js/voiceConversationCore.js tests/test_voice_conversation_core_js.py
git commit -m "feat(voice): pure conversation core (state reducer + energy VAD)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Orchestrator + composer toggle + status pill + event hooks

**Files:**
- Create: `static/js/voiceConversation.js`
- Modify: `static/index.html` (voice-mode button in `.chat-input-right`; module `<script>` beside `voiceRecorder.js`)
- Modify: `static/js/chat.js` (dispatch `assist:reply-complete` in the send `finally`, ~line 3189)
- Modify: `static/js/tts-ai.js` (dispatch `assist:tts-idle` at `_processQueue` drain + in `stop()`)
- Test: none automated (DOM/audio) — the pure logic is covered by Task 1; behavior is live-verified in Task 3.

**Interfaces:**
- Consumes (from Task 1): `STATES`, `nextState`, `createTurnDetector` from `./voiceConversationCore.js`.
- Consumes (existing): `GET /api/stt/stats`, `POST /api/stt/transcribe`, `window.aiTTSManager` (`autoPlay`, `isPlaying`, `_processing`, `stop()`), `#message`, `.send-btn`.
- Produces: listens for `window` events `assist:reply-complete` and `assist:tts-idle` (dispatched by the hooks below).

- [ ] **Step 1: Add the voice-mode toggle button**

In `static/index.html`, inside `<div class="chat-input-right">` (around line 1226), immediately **before** the `<button type="submit" ... class="send-btn ...">` (line 1232), add:

```html
          <!-- Hands-free voice conversation toggle -->
          <button type="button" class="input-icon-btn" id="voice-mode-btn" title="Voice conversation" aria-pressed="false">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
          </button>
```

- [ ] **Step 2: Add the module script tag**

In `static/index.html`, next to the existing `<script type="module" src="/static/js/voiceRecorder.js"></script>` (line 2788), add on the following line:

```html
<script type="module" src="/static/js/voiceConversation.js"></script>
```

- [ ] **Step 3: Add the `assist:reply-complete` hook in chat.js**

In `static/js/chat.js`, in the send `finally` block, immediately after the `aria-busy` line at [chat.js:3189](../../static/js/chat.js#L3189):

```javascript
      if (_chatLogDone) _chatLogDone.setAttribute('aria-busy', 'false');
```

add:

```javascript
      // Voice mode: signal the hands-free loop that this turn's response is done
      // (fires on success, error, and background paths). No-op if nobody listens.
      try { window.dispatchEvent(new CustomEvent('assist:reply-complete')); } catch (_) {}
```

- [ ] **Step 4: Add the `assist:tts-idle` hooks in tts-ai.js**

In `static/js/tts-ai.js`, at the end of `_processQueue()` where it sets `this._processing = false;` (~line 277), change:

```javascript
        this._processing = false;
    }
```

to:

```javascript
        this._processing = false;
        // Voice mode: the play queue has fully drained.
        try { window.dispatchEvent(new CustomEvent('assist:tts-idle')); } catch (_) {}
    }
```

And in `stop()`, as the **last** statement of the method (after the `if (this.currentAudio) { ... }` block, ~line 247), add:

```javascript
        // Voice mode: TTS was force-stopped — treat as idle.
        try { window.dispatchEvent(new CustomEvent('assist:tts-idle')); } catch (_) {}
```

(The orchestrator only acts on `assist:tts-idle` when `replyComplete` is true and it is in the `speaking` state, so extra/early dispatches are harmless no-ops.)

- [ ] **Step 5: Create the orchestrator**

Create `static/js/voiceConversation.js`:

```javascript
// static/js/voiceConversation.js
// Hands-free voice conversation loop. Orchestrates the existing STT
// (/api/stt/transcribe), the normal chat send (.send-btn), and streaming TTS
// (window.aiTTSManager) into a continuous turn loop. DOM/audio wiring only —
// all pure logic lives in voiceConversationCore.js.
import { STATES, nextState, createTurnDetector } from './voiceConversationCore.js';

(function () {
  const SILENCE_TIMEOUT_MS = 25000; // no speech at all → exit voice mode
  const VAD = { threshold: 0.015, hangoverMs: 1200, minSpeechMs: 300 };
  const LABELS = {
    listening: 'Listening…', transcribing: 'Thinking…',
    thinking: 'Thinking…', speaking: 'Speaking…',
  };

  let state = STATES.IDLE;
  let stream = null;        // MediaStream (persists across turns)
  let audioCtx = null;
  let analyser = null;
  let recorder = null;      // MediaRecorder (recreated per turn)
  let chunks = [];
  let detector = null;
  let rafId = null;
  let watchdog = null;
  let sawSpeech = false;    // did the current listening turn hear any speech?
  let replyComplete = false;
  let prevAutoPlay = null;  // to restore aiTTSManager.autoPlay on exit
  let pill = null, pillLabel = null, pillMeter = null;

  function mgr() { return window.aiTTSManager; }
  function toast(msg, ms) {
    try { if (window.uiModule && window.uiModule.showToast) window.uiModule.showToast(msg, ms); else console.log('[voice]', msg); } catch (_) {}
  }

  // ── State machine ──
  function dispatch(event) {
    const from = state;
    state = nextState(state, event);
    if (state !== from) onEnter(state, from);
    else updatePill();
  }

  function onEnter(s, from) {
    updatePill();
    if (s === STATES.LISTENING) beginListening();
    else if (s === STATES.IDLE && from !== STATES.IDLE) teardown();
  }

  // ── UI: status pill ──
  function ensurePill() {
    if (pill) return;
    pill = document.createElement('div');
    pill.id = 'voice-pill';
    pill.style.cssText = 'display:none;align-items:center;gap:8px;padding:4px 10px;margin:4px 8px;border-radius:14px;background:rgba(127,127,127,0.14);font-size:12px;';
    pillLabel = document.createElement('span');
    pillLabel.textContent = 'Listening…';
    const meterWrap = document.createElement('span');
    meterWrap.style.cssText = 'flex:1;height:4px;max-width:80px;background:rgba(127,127,127,0.25);border-radius:2px;overflow:hidden;';
    pillMeter = document.createElement('span');
    pillMeter.style.cssText = 'display:block;height:100%;width:0%;background:#50fa7b;';
    meterWrap.appendChild(pillMeter);
    const stopBtn = document.createElement('button');
    stopBtn.type = 'button';
    stopBtn.textContent = 'Stop';
    stopBtn.style.cssText = 'background:none;border:1px solid currentColor;border-radius:10px;padding:1px 8px;cursor:pointer;font-size:11px;color:inherit;';
    stopBtn.addEventListener('click', stop);
    pill.appendChild(pillLabel); pill.appendChild(meterWrap); pill.appendChild(stopBtn);
    // Insert just above the composer input. The anchor class may differ; fall
    // back to inserting before the #message element's parent row.
    const input = document.getElementById('message');
    const row = input ? input.parentElement : null;
    if (row && row.parentNode) row.parentNode.insertBefore(pill, row);
    else document.body.appendChild(pill);
  }

  function updatePill() {
    if (!pill) return;
    const active = state !== STATES.IDLE;
    pill.style.display = active ? 'flex' : 'none';
    if (pillLabel) pillLabel.textContent = LABELS[state] || '';
    if (!active && pillMeter) pillMeter.style.width = '0%';
    const btn = document.getElementById('voice-mode-btn');
    if (btn) { btn.classList.toggle('active', active); btn.setAttribute('aria-pressed', String(active)); }
  }

  // ── Start / capture ──
  async function start() {
    if (state !== STATES.IDLE) return;
    try {
      const r = await fetch('/api/stt/stats', { credentials: 'same-origin' });
      const s = r.ok ? await r.json() : {};
      if (!s.provider || s.provider === 'disabled') {
        toast('Enable Speech-to-Text in settings to use voice mode');
        return;
      }
    } catch (_) { toast('Speech-to-Text unavailable'); return; }

    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast('Microphone unavailable (needs HTTPS or localhost)');
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (_) { toast('Microphone access denied'); return; }

    const m = mgr();
    if (m) { prevAutoPlay = m.autoPlay; m.autoPlay = true; }

    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const srcNode = audioCtx.createMediaStreamSource(stream);
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      srcNode.connect(analyser);
    } catch (_) { toast('Audio init failed'); teardown(); return; }

    window.addEventListener('assist:reply-complete', onReplyComplete);
    window.addEventListener('assist:tts-idle', onTtsIdle);
    document.addEventListener('keydown', onKeydown);

    ensurePill();
    dispatch('TOGGLE_ON'); // → LISTENING → beginListening()
  }

  function beginListening() {
    replyComplete = false;
    sawSpeech = false;
    chunks = [];
    detector = createTurnDetector(VAD);
    try {
      recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    } catch (_) {
      try { recorder = new MediaRecorder(stream); } catch (e) { dispatch('ERROR'); return; }
    }
    recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    recorder.onstop = onRecorderStop;
    recorder.start();

    clearTimeout(watchdog);
    watchdog = setTimeout(() => { if (!sawSpeech && state === STATES.LISTENING) dispatch('SILENCE_TIMEOUT'); }, SILENCE_TIMEOUT_MS);

    const buf = new Float32Array(analyser.fftSize);
    const tick = () => {
      if (state !== STATES.LISTENING) return;
      analyser.getFloatTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      const rms = Math.sqrt(sum / buf.length);
      if (pillMeter) pillMeter.style.width = Math.min(100, Math.round(rms * 400)) + '%';
      const res = detector.observe(rms, performance.now());
      if (res === 'speech') sawSpeech = true;
      if (res === 'ended') { endTurn(); return; }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
  }

  function endTurn() {
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
    clearTimeout(watchdog);
    dispatch('SPEECH_END'); // → TRANSCRIBING
    try {
      if (recorder && recorder.state === 'recording') recorder.stop(); // → onRecorderStop
      else dispatch('ERROR');
    } catch (_) { dispatch('ERROR'); }
  }

  async function onRecorderStop() {
    if (state !== STATES.TRANSCRIBING) return;
    const blob = new Blob(chunks, { type: 'audio/webm' });
    let text = '';
    try {
      const fd = new FormData();
      fd.append('file', blob, 'audio.webm');
      const r = await fetch('/api/stt/transcribe', { method: 'POST', credentials: 'same-origin', body: fd });
      if (r.ok) { const d = await r.json(); text = (d.text || '').trim(); }
    } catch (_) { /* fall through to EMPTY */ }

    if (!text) { dispatch('EMPTY'); return; } // → LISTENING → beginListening()

    const input = document.getElementById('message');
    if (input) {
      input.value = text;
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }
    dispatch('TEXT'); // → THINKING
    const sb = document.querySelector('.send-btn');
    if (sb) sb.click(); // reuse the existing send path
    else dispatch('ERROR');
  }

  // ── Turn completion signals ──
  function onReplyComplete() {
    if (state !== STATES.THINKING) return;
    replyComplete = true;
    const m = mgr();
    const speaking = !!(m && (m.isPlaying || m._processing));
    dispatch(speaking ? 'REPLY_SPOKEN' : 'REPLY_SILENT');
    // REPLY_SILENT → LISTENING → beginListening(); REPLY_SPOKEN → SPEAKING (await tts-idle)
  }

  function onTtsIdle() {
    if (replyComplete && state === STATES.SPEAKING) dispatch('TTS_IDLE'); // → LISTENING
  }

  function onKeydown(e) { if (e.key === 'Escape' && state !== STATES.IDLE) stop(); }

  // ── Stop / teardown ──
  function stop() { dispatch('STOP'); } // → IDLE → teardown()

  function teardown() {
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
    clearTimeout(watchdog);
    try { if (recorder && recorder.state === 'recording') recorder.stop(); } catch (_) {}
    recorder = null;
    if (stream) { try { stream.getTracks().forEach((t) => t.stop()); } catch (_) {} stream = null; }
    if (audioCtx) { try { audioCtx.close(); } catch (_) {} audioCtx = null; }
    analyser = null; detector = null; chunks = [];
    const m = mgr();
    if (m) { try { m.stop(); } catch (_) {} if (prevAutoPlay !== null) m.autoPlay = prevAutoPlay; }
    prevAutoPlay = null; replyComplete = false; sawSpeech = false;
    window.removeEventListener('assist:reply-complete', onReplyComplete);
    window.removeEventListener('assist:tts-idle', onTtsIdle);
    document.removeEventListener('keydown', onKeydown);
    updatePill();
  }

  // ── Init ──
  function init() {
    const btn = document.getElementById('voice-mode-btn');
    if (!btn) return;
    btn.addEventListener('click', () => { if (state === STATES.IDLE) start(); else stop(); });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
```

- [ ] **Step 6: Self-review (no automated tests for this task)**

Re-read `voiceConversation.js` and the three edits, checking:
- **CSP-safe:** grep your new file for `innerHTML` and inline `on*=` — there must be none; all dynamic text uses `textContent`; the meter/pill colors are literal.
- **Reuse, not reinvent:** the send is triggered only via `.send-btn.click()`; no direct fetch to a chat endpoint. STT uses `POST /api/stt/transcribe`.
- **Lifecycle correctness:** `teardown()` stops the recorder, stops all stream tracks, closes the `AudioContext`, cancels the RAF + watchdog, calls `aiTTSManager.stop()`, restores `autoPlay`, and removes all three listeners.
- **Race safety:** `onTtsIdle` acts only when `replyComplete && state==='speaking'` (so mid-stream queue drains are ignored); `onReplyComplete` acts only in `thinking`.
- **Hooks fire once per turn:** `assist:reply-complete` is in the send `finally`; `assist:tts-idle` at queue drain + `stop()`.
Fix anything you find.

- [ ] **Step 7: Commit**

```bash
git add static/js/voiceConversation.js static/index.html static/js/chat.js static/js/tts-ai.js
git commit -m "feat(voice): hands-free conversation loop (orchestrator, pill, hooks)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Package + live-verify

**Files:**
- Modify: `installer/Output/Assist-Setup.exe` (rebuilt; force-added like prior build commits)

**Interfaces:**
- Consumes: Tasks 1-2.

- [ ] **Step 1: Full affected-suite run**

Run: `python -m pytest tests/test_voice_conversation_core_js.py --import-mode=importlib -q`
Expected: PASS (5 passed) — or SKIPPED if `node` is not on PATH (note which in the report).

- [ ] **Step 2: Build the installer**

Run: `powershell -ExecutionPolicy Bypass -File ./build-installer.ps1 -Fast`
Expected: ends with `Installer built: ...\installer\Output\Assist-Setup.exe`.

- [ ] **Step 3: Live-verify in the running app (manual)**

Reinstall, then with an STT provider configured (e.g. local Whisper) and a TTS provider configured:
- Click the **voice-mode** button in the composer → the status pill appears showing **Listening…** with a live mic-level meter.
- Speak a short question, then pause ~1.5s → pill goes **Thinking…**, the transcript is auto-sent as a user message, the AI replies, and the reply is **spoken** (pill **Speaking…**).
- When the reply finishes speaking, the pill returns to **Listening…** automatically (the loop continues).
- Confirm the mic is not capturing while the AI speaks (talking during the reply does not start a new turn until it returns to Listening).
- Press **Stop** (and separately **Esc**) → voice mode exits, the pill disappears, and the mic indicator/track is released.
- Stay silent ~25s in Listening → voice mode auto-exits.
- With STT disabled in settings, click the button → a toast tells you to enable Speech-to-Text and nothing else happens.

- [ ] **Step 4: Commit the installer**

```bash
git add -f installer/Output/Assist-Setup.exe
git commit -m "build: Assist-Setup.exe with hands-free voice conversation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **No backend/Python changes** in Tasks 1-2 — only `static/js/*` and `static/index.html`. The one Python file added is the *test* in Task 1.
- Task 1 is the only task with automated tests; its pure functions are the whole reason the orchestrator (Task 2) can be thin untested wiring.
- The mic `MediaStream` stays open for the whole session; only the `MediaRecorder` and the RMS RAF loop cycle per turn, and both are inactive during `thinking`/`speaking` — that is what "mic closed while the AI speaks" means in practice (no track re-acquisition per turn).
- Known v1 limitation (acceptable, note only): a very short reply whose spoken text never reaches the streaming-TTS minimum sentence length is not spoken; the loop simply returns to Listening. Do not add special handling.
