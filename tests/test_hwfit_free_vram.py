import services.hwfit.hardware as hw


def test_free_vram_parses_first_gpu(monkeypatch):
    monkeypatch.setattr(hw, "_run", lambda cmd: "5432\n")
    got = hw.free_vram_gb()
    assert got is not None and abs(got - 5432 / 1024.0) < 0.01


def test_free_vram_multi_gpu_returns_first(monkeypatch):
    monkeypatch.setattr(hw, "_run", lambda cmd: "5432\n8000\n")
    assert abs(hw.free_vram_gb() - 5432 / 1024.0) < 0.01


def test_free_vram_none_when_no_smi(monkeypatch):
    monkeypatch.setattr(hw, "_run", lambda cmd: None)
    assert hw.free_vram_gb() is None


def test_free_vram_none_when_nonnumeric(monkeypatch):
    monkeypatch.setattr(hw, "_run", lambda cmd: "[N/A]\n")
    assert hw.free_vram_gb() is None
