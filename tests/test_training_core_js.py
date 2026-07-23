import json
import subprocess
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "static" / "js" / "trainingCore.js"


def _node(expr):
    script = f"import * as m from {json.dumps(CORE.as_uri())};\n{expr}"
    p = subprocess.run(["node", "--input-type=module", "-e", script],
                       capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def test_parse_params_b():
    out = _node("console.log(JSON.stringify(["
                "m.parseParamsB('Qwen/Qwen2.5-0.5B-Instruct'),"
                "m.parseParamsB('meta-llama/Meta-Llama-3-8B'),"
                "m.parseParamsB('openai-community/gpt2')]));")
    assert json.loads(out) == [0.5, 8, None]


def test_vram_hint_levels():
    out = _node("console.log(JSON.stringify(["
                "m.vramHint(0.5, 6.44).level, m.vramHint(13, 6.44).level, "
                "m.vramHint(0.5, null).level]));")
    assert json.loads(out) == ["fits", "too_big", "unknown"]


def test_form_to_config_steps_xor_epochs():
    out = _node("console.log(JSON.stringify(m.formToConfig("
                "{base_model:'x', dataset_path:'d.jsonl', mode:'steps', steps:'50', "
                "epochs:'3', lora_r:'8', batch_size:'1', learning_rate:'0.0002'})));")
    cfg = json.loads(out)
    assert cfg["steps"] == 50 and cfg.get("epochs") is None
    assert cfg["base_model"] == "x" and cfg["lora_r"] == 8


def test_form_to_config_epochs_mode():
    out = _node("console.log(JSON.stringify(m.formToConfig("
                "{base_model:'x', dataset_path:'d.jsonl', mode:'epochs', steps:'50', epochs:'3'})));")
    cfg = json.loads(out)
    assert cfg["epochs"] == 3 and cfg.get("steps") is None
