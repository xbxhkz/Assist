import src.imagemodels.manager as im
from src.imagemodels.encoders import MissingEncoderError


class SpyManager:
    def __init__(self, running=None):
        self._running = running          # a model id string, or None
        self.started = []
    def status(self):
        if self._running:
            return {"running": True, "model": self._running, "port": 8100}
        return {"running": False, "model": None, "port": None}
    def list_models(self):
        return [{"name": "flux.gguf", "path": "C:/img/flux.gguf", "size": 1}]
    def start(self, files, device="cpu", steps=None):
        self.started.append((files, device))
        return {"running": True, "model": "flux.gguf", "port": 8100}


def _call(**kw):
    kw.setdefault("resolver", lambda p: {"diffusion_model": p})
    kw.setdefault("probe", lambda port: "sd-cpp-local")
    return im.ensure_image_served("admin", **kw)


def test_reuses_already_running_without_start():
    mgr = SpyManager(running="already.gguf")
    out = _call(manager=mgr, settings={"image_model": "flux.gguf"})
    assert out == {"model": "sd-cpp-local", "error": None, "local": True}
    assert mgr.started == []          # no swap


def test_serves_local_default_when_idle():
    mgr = SpyManager(running=None)
    out = _call(manager=mgr, settings={"image_model": "flux.gguf"},
                lister=mgr.list_models)
    assert out == {"model": "sd-cpp-local", "error": None, "local": True}
    assert len(mgr.started) == 1 and mgr.started[0][1] == "gpu"


def test_external_default_is_noop():
    mgr = SpyManager(running=None)
    out = _call(manager=mgr, settings={"image_model": "gpt-image-1.5"},
                lister=mgr.list_models)
    assert out == {"model": None, "error": None, "local": False}
    assert mgr.started == []


def test_unset_default_is_noop():
    mgr = SpyManager(running=None)
    out = _call(manager=mgr, settings={}, lister=mgr.list_models)
    assert out == {"model": None, "error": None, "local": False}


def test_start_failure_returns_error_never_raises():
    mgr = SpyManager(running=None)
    def boom(files, device="cpu", steps=None):
        raise RuntimeError("sd-server did not start")
    mgr.start = boom
    out = _call(manager=mgr, settings={"image_model": "flux.gguf"},
                lister=mgr.list_models)
    assert out["model"] is None and out["local"] is True
    assert "did not start" in out["error"]


def test_missing_encoder_returns_hint():
    mgr = SpyManager(running=None)
    def boom(p):
        e = MissingEncoderError(["t5xxl"])
        e.hint = "Missing FLUX files: t5xxl. Put t5xxl / clip_l / vae ..."
        raise e
    out = _call(manager=mgr, settings={"image_model": "flux.gguf"},
                lister=mgr.list_models, resolver=boom)
    assert out["model"] is None and out["local"] is True
    assert "t5xxl" in out["error"]


def test_get_manager_failure_never_raises(monkeypatch):
    import src.imagemodels.manager as im2
    def boom():
        raise RuntimeError("no manager")
    monkeypatch.setattr(im2, "get_manager", boom)
    # manager not injected → the real get_manager() default is hit and raises
    out = im2.ensure_image_served("admin")
    assert out == {"model": None, "error": None, "local": False}


def test_serves_local_default_but_probe_fails():
    mgr = SpyManager(running=None)
    out = _call(manager=mgr, settings={"image_model": "flux.gguf"},
                lister=mgr.list_models, probe=lambda port: None)
    assert out == {"model": None, "error": None, "local": True}
