// static/js/voiceConversation.js
// Hands-free voice conversation loop. Orchestrates the existing STT
// (/api/stt/transcribe), the normal chat send (.send-btn), and streaming TTS
// (window.aiTTSManager) into a continuous turn loop. DOM/audio wiring only —
// all pure logic lives in voiceConversationCore.js.
import { STATES, nextState, createTurnDetector } from './voiceConversationCore.js';

(function () {
  const SILENCE_TIMEOUT_MS = 25000; // no speech at all → exit voice mode
  const TRANSCRIBE_TIMEOUT_MS = 20000; // abort a stalled STT request → EMPTY → re-listen
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
  let starting = false;     // guards re-entry during the async start() setup

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
  function setBtnPreparing(on) {
    const btn = document.getElementById('voice-mode-btn');
    if (btn) btn.style.opacity = on ? '0.55' : '';
  }
  // Undo the "preparing" UI on a soft failure before the loop actually starts.
  function endPreparing() {
    starting = false;
    setBtnPreparing(false);
    if (pill) pill.style.display = 'none';
  }

  async function start() {
    if (state !== STATES.IDLE || starting) return;
    starting = true;

    // Guard: block voice mode entry in Compare or Group modes
    const compareActive = !!(window.compareModule && window.compareModule.isActive && window.compareModule.isActive());
    const groupActive = !!(window.groupModule && window.groupModule.isActive && window.groupModule.isActive());
    if (compareActive || groupActive) {
      endPreparing();
      toast('Voice mode is not available in Compare or Group chat');
      return;
    }

    // Show a "Preparing…" indicator up front: on first use the /api/stt/stats
    // call blocks while the local Whisper model downloads/loads (~30-60s), so
    // the button would otherwise look unresponsive with no feedback.
    ensurePill();
    if (pill) pill.style.display = 'flex';
    if (pillMeter) pillMeter.style.width = '0%';
    if (pillLabel) pillLabel.textContent = 'Preparing voice…';
    setBtnPreparing(true);

    try {
      const r = await fetch('/api/stt/stats', { credentials: 'same-origin' });
      const s = r.ok ? await r.json() : {};
      if (!s.provider || s.provider === 'disabled') {
        endPreparing();
        toast('Enable Speech-to-Text in settings to use voice mode');
        return;
      }
    } catch (_) { endPreparing(); toast('Speech-to-Text unavailable'); return; }

    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      endPreparing();
      toast('Microphone unavailable (needs HTTPS or localhost)');
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (_) { endPreparing(); toast('Microphone access denied'); return; }

    const m = mgr();
    if (m) { prevAutoPlay = m.autoPlay; m.autoPlay = true; }

    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const srcNode = audioCtx.createMediaStreamSource(stream);
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      srcNode.connect(analyser);
    } catch (_) { setBtnPreparing(false); starting = false; toast('Audio init failed'); teardown(); return; }

    window.addEventListener('assist:reply-complete', onReplyComplete);
    window.addEventListener('assist:tts-idle', onTtsIdle);
    document.addEventListener('keydown', onKeydown);

    setBtnPreparing(false);
    starting = false;
    dispatch('TOGGLE_ON'); // → LISTENING → beginListening() (pill relabels to Listening…)
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
      // Bound the request so a stalled STT can't wedge the loop in 'transcribing':
      // on timeout the fetch aborts → caught here → EMPTY → back to listening.
      const ctrl = new AbortController();
      const to = setTimeout(() => ctrl.abort(), TRANSCRIBE_TIMEOUT_MS);
      try {
        const r = await fetch('/api/stt/transcribe', { method: 'POST', credentials: 'same-origin', body: fd, signal: ctrl.signal });
        if (r.ok) { const d = await r.json(); text = (d.text || '').trim(); }
      } finally {
        clearTimeout(to);
      }
    } catch (_) { /* fall through to EMPTY (incl. AbortError on timeout) */ }

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
    prevAutoPlay = null; replyComplete = false; sawSpeech = false; starting = false;
    setBtnPreparing(false);
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
