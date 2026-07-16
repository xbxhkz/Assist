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
