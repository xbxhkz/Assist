import json, subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "static" / "js" / "exportCore.js"


def _node(expr):
    s = f"import * as m from {json.dumps(CORE.as_uri())};\n{expr}"
    p = subprocess.run(["node", "--input-type=module", "-e", s],
                       capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def test_quants_default_first():
    out = _node("console.log(JSON.stringify(m.EXPORT_QUANTS));")
    q = json.loads(out)
    assert q[0] == "Q4_K_M" and set(q) == {"Q4_K_M", "Q5_K_M", "Q8_0", "F16"}


def test_export_button_state():
    out = _node("console.log(JSON.stringify(["
                "m.exportButtonState({complete:true}).canExport,"
                "m.exportButtonState({complete:false}).canExport]));")
    assert json.loads(out) == [True, False]
