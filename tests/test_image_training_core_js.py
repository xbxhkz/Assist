import json
import subprocess
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "static" / "js" / "imageTrainingCore.js"


def _node(expr):
    script = f"import * as m from {json.dumps(CORE.as_uri())};\n{expr}"
    p = subprocess.run(["node", "--input-type=module", "-e", script],
                       capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def test_form_to_config_uses_given_values():
    out = _node("console.log(JSON.stringify(m.formToConfig("
                "{dataset_name:'ds1', output_name:'my-lora', rank:'8', lora_alpha:'8', "
                "learning_rate:'0.0002', steps:'500', resolution:'768'})));")
    cfg = json.loads(out)
    assert cfg == {"dataset_name": "ds1", "output_name": "my-lora", "rank": 8,
                   "lora_alpha": 8, "learning_rate": 0.0002, "steps": 500, "resolution": 768}


def test_form_to_config_blank_numeric_fields_fall_back_to_proven_defaults():
    out = _node("console.log(JSON.stringify(m.formToConfig("
                "{dataset_name:'ds1', output_name:'my-lora', rank:'', lora_alpha:'', "
                "learning_rate:'', steps:'', resolution:''})));")
    cfg = json.loads(out)
    assert cfg["rank"] == 4 and cfg["lora_alpha"] == 4
    assert cfg["learning_rate"] == 0.0001
    assert cfg["steps"] == 1000 and cfg["resolution"] == 1024


def test_form_to_config_trims_names_and_omits_base_model():
    out = _node("console.log(JSON.stringify(m.formToConfig("
                "{dataset_name:'  ds1  ', output_name:'  my-lora  '})));")
    cfg = json.loads(out)
    assert cfg["dataset_name"] == "ds1" and cfg["output_name"] == "my-lora"
    assert "base_model" not in cfg


def test_render_status_line_running():
    out = _node("console.log(m.renderStatusLine("
                "{status:'running', last_step:5, loss:0.42, vram_gb:6.0}));")
    assert out == "status: running · step 5 · loss 0.42 · vram 6 GB"


def test_render_status_line_done_shows_lora_path():
    out = _node("console.log(m.renderStatusLine("
                "{status:'done', peak_vram_gb:6.04, lora_path:'C:/loras/my-lora.safetensors'}));")
    assert out == "status: done · saved: C:/loras/my-lora.safetensors"


def test_render_status_line_error():
    out = _node("console.log(m.renderStatusLine({status:'error', error:'ran out of VRAM'}));")
    assert out == "status: error · (ran out of VRAM)"
