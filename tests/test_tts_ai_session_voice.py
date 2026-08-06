"""tts-ai.js becomes session-aware without changing any of its callers'
call sites (play/enqueue/streaming*) -- synthesize() and the browser-voice
path both resolve the current session internally via sessions.js's
getCurrentSessionId(), matching how modelPicker.js already consumes it."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_tts_ai_imports_get_current_session_id():
    src = (ROOT / "static" / "js" / "tts-ai.js").read_text(encoding="utf-8")
    assert "getCurrentSessionId" in src
    assert re.search(r"import\s*\{[^}]*getCurrentSessionId[^}]*\}\s*from\s*['\"]\./sessions\.js['\"]", src)


def test_synthesize_sends_session_id_in_request_body():
    src = (ROOT / "static" / "js" / "tts-ai.js").read_text(encoding="utf-8")
    m = re.search(r"async synthesize\([^)]*\)\s*\{.*?\n    \}", src, re.S)
    assert m is not None, "synthesize method not found"
    assert "session_id" in m.group(0)


def test_resolve_browser_voice_uses_session_scoped_stats():
    src = (ROOT / "static" / "js" / "tts-ai.js").read_text(encoding="utf-8")
    assert "_resolveBrowserVoice" in src
    m = re.search(r"_resolveBrowserVoice\([^)]*\)\s*\{.*?\n    \}", src, re.S)
    assert m is not None, "_resolveBrowserVoice method not found"
    assert "session_id" in m.group(0) or "getCurrentSessionId" in m.group(0)


def test_playbrowser_is_async_and_awaits_voice_resolution():
    """Must actually AWAIT _resolveBrowserVoice(), not just reference its
    name somewhere in the body -- `const voiceName =
    this._resolveBrowserVoice();` (a dropped await) would still pass a
    substring-only check, but would leave voiceName a pending Promise, and
    _findBrowserVoice(voiceName) would throw calling .toLowerCase() on a
    Promise object."""
    src = (ROOT / "static" / "js" / "tts-ai.js").read_text(encoding="utf-8")
    m = re.search(r"async _playBrowser\([^)]*\)\s*\{.*?\n    \}", src, re.S)
    assert m is not None, "_playBrowser must be declared async"
    assert "await this._resolveBrowserVoice()" in m.group(0)


def test_get_cache_key_includes_session_id():
    """Behavioral, not identifier-presence: the extraction regex captures
    getCacheKey's parameter list too, so `sessionId in body` would still
    pass even if the function silently dropped the parameter and never used
    it in the hash (reintroducing the pre-Task-4 cache-collision bug under a
    new signature). Extract the real method and run it in node, proving two
    different sessionId values actually produce two different cache keys
    for the same text."""
    src = (ROOT / "static" / "js" / "tts-ai.js").read_text(encoding="utf-8")
    m = re.search(r"getCacheKey\([^)]*\)\s*\{.*?\n    \}", src, re.S)
    assert m is not None, "getCacheKey method not found"
    body = m.group(0)

    script = f"""
    class T {{
        {body}
    }}
    const t = new T();
    const a = t.getCacheKey('hello world', 'session-a');
    const b = t.getCacheKey('hello world', 'session-b');
    if (a === b) throw new Error('same text, different sessionId produced the same cache key: ' + a);
    console.log('OK:' + a + ':' + b);
    """
    p = subprocess.run(["node", "-e", script], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.startswith("OK:")


def test_stop_invalidates_inflight_playbrowser_before_speak():
    """Important review finding: _playBrowser awaits _resolveBrowserVoice()
    (a real fetch on a session's first browser-TTS utterance), and nothing
    re-checked play state after that await resolved before this fix -- a
    Stop click during the await window did nothing (nothing had reached
    speechSynthesis.speak() yet, so speechSynthesis.cancel() was a no-op),
    so audio started playing anyway once the fetch resolved with a button
    UI that already claimed playback had stopped.

    This extracts the ACTUAL _playBrowser and stop() method bodies from
    tts-ai.js (not a reimplementation) and runs them in node with
    speechSynthesis mocked: it starts _playBrowser, holds its voice
    resolution pending, calls stop() while that await is still in flight,
    THEN resolves the pending voice promise -- and asserts
    speechSynthesis.speak() was never called. Run against the pre-fix
    source (no _playGeneration guard) this same harness fails, proving the
    test discriminates the bug rather than passing unconditionally."""
    src = (ROOT / "static" / "js" / "tts-ai.js").read_text(encoding="utf-8")

    m_play = re.search(r"async _playBrowser\([^)]*\)\s*\{.*?\n    \}", src, re.S)
    assert m_play is not None, "_playBrowser method not found"
    play_body = m_play.group(0)

    m_stop = re.search(r"stop\(\)\s*\{.*?\n    \}", src, re.S)
    assert m_stop is not None, "stop method not found"
    stop_body = m_stop.group(0)

    script = f"""
    let speakCalls = 0;
    global.window = {{
        speechSynthesis: {{
            cancel: () => {{}},
            // Resolve the inner Promise the real _playBrowser code creates
            // so playPromise below actually settles either way.
            speak: (utterance) => {{ speakCalls++; if (utterance.onend) utterance.onend(); }},
            getVoices: () => [],
        }},
        dispatchEvent: () => {{}},
    }};
    global.SpeechSynthesisUtterance = function(text) {{ this.text = text; }};
    global.CustomEvent = function(name) {{ this.name = name; }};

    class TestManager {{
        constructor() {{
            this._playGeneration = 0;
            this.playbackSpeed = 1;
            this.isPlaying = false;
            this.useBrowserTTS = true;
            this._queue = [];
            this._processing = false;
            this._streamActive = false;
            this._streamDebounceTimer = null;
            this._streamSentencesSent = 0;
            this.currentAudio = null;
        }}
        _resolveBrowserVoice() {{ return this._pendingVoicePromise; }}
        _findBrowserVoice(voiceName) {{ return null; }}
        {play_body}
        {stop_body}
    }}

    const mgr = new TestManager();
    let resolveVoice;
    mgr._pendingVoicePromise = new Promise((res) => {{ resolveVoice = res; }});

    const playPromise = mgr._playBrowser('hello world');
    // Simulate the user clicking Stop while _resolveBrowserVoice()'s fetch
    // is still in flight -- before the fix, this was a silent no-op.
    mgr.stop();
    // Now let the delayed voice resolution complete.
    resolveVoice('some-voice');

    playPromise.then(() => {{
        if (speakCalls !== 0) {{
            console.error('FAIL speak called ' + speakCalls + ' times after stop()');
            process.exitCode = 1;
            return;
        }}
        console.log('OK');
    }}).catch((err) => {{
        console.error('FAIL ' + err);
        process.exitCode = 1;
    }});
    """
    p = subprocess.run(["node", "-e", script], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "OK" in p.stdout
