"""Convert a trained peft LoRA adapter to a GGUF LoRA, INSIDE the Py3.11 sidecar
venv (never imported by the Py3.14 app). Emits one JSON line to stdout:
  {"event":"done","adapter_gguf":<path>}  |  {"event":"error","message":<str>}
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default=None)
    args = ap.parse_args()
    try:
        out_path = os.path.join(args.adapter, "adapter.gguf")
        argv = [sys.executable, os.path.join(HERE, "convert_lora_to_gguf.py"),
                "--outfile", out_path]
        if args.base:
            argv += ["--base", args.base]
        argv += [args.adapter]
        p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if p.returncode != 0 or not os.path.isfile(out_path):
            tail = ((p.stdout or "") + (p.stderr or ""))[-1500:]
            emit({"event": "error", "message": "convert_lora_to_gguf failed: " + tail})
            sys.exit(1)
        emit({"event": "done", "adapter_gguf": out_path})
    except Exception as e:  # noqa: BLE001
        try:
            emit({"event": "error", "message": f"{e}"})
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
