import json, subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "static" / "js" / "serveCore.js"


def _node(expr):
    s = f"import * as m from {json.dumps(CORE.as_uri())};\n{expr}"
    p = subprocess.run(["node", "--input-type=module", "-e", s],
                       capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def test_adapter_actions():
    out = _node("console.log(JSON.stringify(["
                "m.adapterActions({complete:true,converted:false}),"
                "m.adapterActions({complete:true,converted:true}),"
                "m.adapterActions({complete:false,converted:false})]));")
    a = json.loads(out)
    assert a[0] == {"canConvert": True, "canServe": False}
    assert a[1] == {"canConvert": False, "canServe": True}
    assert a[2] == {"canConvert": False, "canServe": False}
