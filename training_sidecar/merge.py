"""Merge a LoRA adapter into its base and export a full F16 GGUF. Runs INSIDE the
Py3.11 CUDA venv (never imported by the Py3.14 app). Emits one JSON line to stdout:
  {"event":"done","f16_gguf":<path>}  |  {"event":"error","message":<str>}
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

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
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--base", default=None)
    args = ap.parse_args()
    tmp = None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base = args.base
        if not base:
            with open(os.path.join(args.adapter, "adapter_config.json"), "r", encoding="utf-8") as f:
                base = json.load(f).get("base_model_name_or_path")
        if not base:
            emit({"event": "error", "message": "base model id not found in adapter_config.json"})
            sys.exit(1)

        tmp = tempfile.mkdtemp(prefix="assist_merge_")
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16)
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        model.save_pretrained(tmp)
        AutoTokenizer.from_pretrained(base).save_pretrained(tmp)

        argv = [sys.executable, os.path.join(HERE, "convert_hf_to_gguf.py"), tmp,
                "--outfile", args.outfile, "--outtype", "f16"]
        p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0 or not os.path.isfile(args.outfile):
            tail = ((p.stdout or "") + (p.stderr or ""))[-1500:]
            emit({"event": "error", "message": "convert_hf_to_gguf failed: " + tail})
            sys.exit(1)
        emit({"event": "done", "f16_gguf": args.outfile})
    except Exception as e:  # noqa: BLE001
        try:
            emit({"event": "error", "message": f"{e}"})
        except Exception:
            pass
        sys.exit(1)
    finally:
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
