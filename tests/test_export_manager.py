import json
import os
from src.training.export_manager import AdapterExporter, VALID_QUANTS


class FakeEnv:
    def __init__(self, ready=True): self._ready = ready
    def ensure_ready(self, progress=None):
        return {"ready": self._ready, "error": None if self._ready else "no env"}
    def venv_python(self): return "venv/python"


def _adapter(tmp_path):
    d = tmp_path / "run-1"; d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"}))
    (d / "run_config.json").write_text(json.dumps({"base_model": "Qwen/Qwen2.5-0.5B-Instruct"}))
    return str(d)


def test_export_quantized_runs_merge_then_quantize_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.export_manager.resolve_merge_script", lambda: "merge.py")
    exports = tmp_path / "exports"
    calls = []

    def spawn(argv):
        calls.append(argv)
        if argv[1].endswith("merge.py"):
            f16 = argv[argv.index("--outfile") + 1]
            open(f16, "w").close()  # sidecar writes the F16 GGUF
            return (0, json.dumps({"event": "done", "f16_gguf": f16}))
        # llama-quantize: argv = [quantize_exe, <f16>, <out>, <QUANT>]
        open(argv[2], "w").close()
        return (0, "quantized")

    exp = AdapterExporter(env=FakeEnv(), spawn=spawn, exports_dir=str(exports),
                          quantize_resolver=lambda device="cpu": "llama-quantize.exe")
    out = exp.export(_adapter(tmp_path), "Q4_K_M")
    assert out.get("ok") is True and out["gguf"].endswith("Q4_K_M.gguf")
    assert os.path.isfile(out["gguf"])
    # merge ran, then quantize ran
    assert calls[0][1].endswith("merge.py") and calls[1][0] == "llama-quantize.exe" and calls[1][3] == "Q4_K_M"
    # the F16 intermediate was cleaned up
    f16 = calls[0][calls[0].index("--outfile") + 1]
    assert not os.path.isfile(f16)


def test_f16_export_skips_quantize(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.export_manager.resolve_merge_script", lambda: "merge.py")
    def spawn(argv):
        f16 = argv[argv.index("--outfile") + 1]; open(f16, "w").close()
        return (0, json.dumps({"event": "done", "f16_gguf": f16}))
    exp = AdapterExporter(env=FakeEnv(), spawn=spawn, exports_dir=str(tmp_path / "e"),
                          quantize_resolver=lambda device="cpu": "q.exe")
    out = exp.export(_adapter(tmp_path), "F16")
    assert out.get("ok") is True and out["gguf"].endswith("F16.gguf") and os.path.isfile(out["gguf"])


def test_invalid_quant_rejected(tmp_path):
    exp = AdapterExporter(env=FakeEnv(), spawn=lambda a: (0, ""), exports_dir=str(tmp_path))
    out = exp.export(_adapter(tmp_path), "Q3_K_S")
    assert "error" in out and "quant" in out["error"].lower()


def test_env_not_ready_errors(tmp_path):
    exp = AdapterExporter(env=FakeEnv(ready=False), spawn=lambda a: (0, ""), exports_dir=str(tmp_path))
    out = exp.export(_adapter(tmp_path), "Q4_K_M")
    assert "error" in out and "env" in out["error"].lower()


def test_merge_error_surfaced_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.export_manager.resolve_merge_script", lambda: "merge.py")
    def spawn(argv): return (1, json.dumps({"event": "error", "message": "arch unsupported"}))
    exp = AdapterExporter(env=FakeEnv(), spawn=spawn, exports_dir=str(tmp_path / "e"),
                          quantize_resolver=lambda device="cpu": "q.exe")
    out = exp.export(_adapter(tmp_path), "Q4_K_M")
    assert "error" in out and "arch unsupported" in out["error"]


def test_never_raises_on_spawn_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.export_manager.resolve_merge_script", lambda: "merge.py")
    def boom(argv): raise RuntimeError("cannot spawn")
    exp = AdapterExporter(env=FakeEnv(), spawn=boom, exports_dir=str(tmp_path / "e"),
                          quantize_resolver=lambda device="cpu": "q.exe")
    assert "error" in exp.export(_adapter(tmp_path), "Q4_K_M")


def test_valid_quants_shape():
    assert VALID_QUANTS[0] == "Q4_K_M" and set(VALID_QUANTS) == {"Q4_K_M", "Q5_K_M", "Q8_0", "F16"}
