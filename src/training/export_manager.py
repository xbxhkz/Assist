"""Orchestrate merge->convert->quantize export of a tuned adapter to a standalone
GGUF. Never imports the merge stack; never raises. Blocking (call via asyncio.to_thread)."""
import json
import os
import re
import subprocess

from src.training.runtime import resolve_merge_script

VALID_QUANTS = ("Q4_K_M", "Q5_K_M", "Q8_0", "F16")


def _default_spawn(argv):
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s or "model").rsplit("/", 1)[-1]).strip("-") or "model"


class AdapterExporter:
    def __init__(self, env=None, spawn=None, exports_dir=None, quantize_resolver=None):
        if env is None:
            from src.training.env import TrainingEnv
            env = TrainingEnv()
        if exports_dir is None:
            from src.constants import DATA_DIR
            exports_dir = os.path.join(DATA_DIR, "training", "exports")
        if quantize_resolver is None:
            from src.localmodels.runtime import resolve_quantize_binary
            quantize_resolver = resolve_quantize_binary
        self._env = env
        self._spawn = spawn or _default_spawn
        self._exports_dir = exports_dir
        self._resolve_quant = quantize_resolver

    def export(self, adapter_dir, quant, base_model=None) -> dict:
        f16 = None
        try:
            if quant not in VALID_QUANTS:
                return {"error": f"invalid quant '{quant}' (allowed: {', '.join(VALID_QUANTS)})"}
            if not (isinstance(adapter_dir, str) and os.path.isdir(adapter_dir)):
                return {"error": "adapter directory not found"}
            ready = self._env.ensure_ready()
            if not ready.get("ready"):
                return {"error": f"training env not ready: {ready.get('error')}"}

            os.makedirs(self._exports_dir, exist_ok=True)
            run_id = os.path.basename(adapter_dir.rstrip("/\\"))
            if not base_model:
                base_model = self._read_base(adapter_dir)
            out_name = f"{_slug(base_model)}-{run_id}-{quant}.gguf"
            final = os.path.join(self._exports_dir, out_name)
            f16 = os.path.join(self._exports_dir, f".{run_id}.f16.gguf")

            # 1) merge + convert -> F16 GGUF (in the venv)
            argv = [self._env.venv_python(), resolve_merge_script(),
                    "--adapter", adapter_dir, "--outfile", f16]
            if base_model:
                argv += ["--base", base_model]
            rc, out = self._spawn(argv)
            ev = _last_json(out)
            if ev.get("event") == "error":
                return {"error": ev.get("message", "merge failed")}
            if not os.path.isfile(f16):
                return {"error": "merge/convert failed: " + (out or "")[-500:]}

            # 2) quantize (or, for F16, the F16 GGUF is the output)
            if quant == "F16":
                os.replace(f16, final)
                f16 = None
                return {"ok": True, "gguf": final}
            qexe = self._resolve_quant()
            rc, out = self._spawn([qexe, f16, final, quant])
            if rc != 0 or not os.path.isfile(final):
                return {"error": "quantize failed: " + (out or "")[-500:]}
            return {"ok": True, "gguf": final}
        except Exception as e:  # noqa: BLE001
            return {"error": f"export error: {e}"}
        finally:
            if f16 and os.path.isfile(f16):
                try:
                    os.remove(f16)
                except Exception:
                    pass

    def _read_base(self, adapter_dir):
        for name, key in (("adapter_config.json", "base_model_name_or_path"),
                          ("run_config.json", "base_model")):
            p = os.path.join(adapter_dir, name)
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        v = json.load(f).get(key)
                    if v:
                        return v
                except Exception:
                    pass
        return None

    def list_exports(self, run_id) -> list:
        out = []
        if not os.path.isdir(self._exports_dir):
            return out
        for fn in sorted(os.listdir(self._exports_dir)):
            if fn.endswith(".gguf") and f"-{run_id}-" in fn:
                out.append(os.path.join(self._exports_dir, fn))
        return out

    def exports_dir(self):
        return self._exports_dir


def _last_json(text):
    ev = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                ev = json.loads(line)
            except Exception:
                pass
    return ev


_exporter = None


def get_adapter_exporter():
    global _exporter
    if _exporter is None:
        _exporter = AdapterExporter()
    return _exporter
