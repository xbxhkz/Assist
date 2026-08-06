"""tts-ai.js becomes session-aware without changing any of its callers'
call sites (play/enqueue/streaming*) -- synthesize() and the browser-voice
path both resolve the current session internally via sessions.js's
getCurrentSessionId(), matching how modelPicker.js already consumes it."""
import re
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
    src = (ROOT / "static" / "js" / "tts-ai.js").read_text(encoding="utf-8")
    m = re.search(r"async _playBrowser\([^)]*\)\s*\{.*?\n    \}", src, re.S)
    assert m is not None, "_playBrowser must be declared async"
    assert "_resolveBrowserVoice" in m.group(0)


def test_get_cache_key_includes_session_id():
    src = (ROOT / "static" / "js" / "tts-ai.js").read_text(encoding="utf-8")
    m = re.search(r"getCacheKey\([^)]*\)\s*\{.*?\n    \}", src, re.S)
    assert m is not None, "getCacheKey method not found"
    assert "sessionId" in m.group(0)
