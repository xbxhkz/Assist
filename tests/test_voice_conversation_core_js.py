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
    import {{ nextState }} from '{_CORE.as_uri()}';
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
    import {{ nextState }} from '{_CORE.as_uri()}';
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
    import {{ createTurnDetector }} from '{_CORE.as_uri()}';
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
    import {{ createTurnDetector }} from '{_CORE.as_uri()}';
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
    import {{ createTurnDetector }} from '{_CORE.as_uri()}';
    const d = createTurnDetector({{ threshold: 0.02, hangoverMs: 1000, minSpeechMs: 300 }});
    let t = 0; const out = [];
    for (let i = 0; i < 60; i++) {{ out.push(d.observe(0.10, t)); t += 50; }}
    console.log(JSON.stringify(out));
    """
    assert 'ended' not in json.loads(_run(js))
