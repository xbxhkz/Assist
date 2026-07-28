import json, subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "static" / "js" / "datasetCore.js"


def _node(expr):
    s = f"import * as m from {json.dumps(CORE.as_uri())};\n{expr}"
    p = subprocess.run(["node", "--input-type=module", "-e", s],
                       capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def test_formats():
    out = _node("console.log(JSON.stringify(Object.keys(m.ROW_FORMATS)));")
    assert set(json.loads(out)) == {"text", "instruction", "prompt"}


def test_form_to_row():
    out = _node("console.log(JSON.stringify(["
                "m.formToRow('text', {text:'hi'}),"
                "m.formToRow('instruction', {instruction:'a'}),"
                "m.formToRow('prompt', {prompt:'Q', completion:'A'})]));")
    a = json.loads(out)
    assert a[0]["row"] == {"text": "hi"} and a[0]["error"] is None
    assert a[1]["row"] is None and "response" in a[1]["error"]
    assert a[2]["row"] == {"prompt": "Q", "completion": "A"}
