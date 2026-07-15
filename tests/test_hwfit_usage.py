import services.hwfit.hardware as hw


class _FakeVM:
    used = 12 * 1024 ** 3
    total = 64 * 1024 ** 3
    percent = 23.9


class _FakePsutil:
    @staticmethod
    def cpu_percent():
        return 42.0

    @staticmethod
    def virtual_memory():
        return _FakeVM()


def test_usage_parses_cpu_ram_and_gpu(monkeypatch):
    monkeypatch.setattr(hw, "psutil", _FakePsutil)
    monkeypatch.setattr(hw, "_run", lambda cmd: "0, RTX 4050, 515, 6141, 6\n")
    u = hw.usage()
    assert u["cpu_percent"] == 42.0
    assert u["ram_total_gb"] == 64.0 and u["ram_percent"] == 23.9
    g = u["gpus"][0]
    assert g["index"] == 0 and g["name"] == "RTX 4050"
    assert g["vram_total_gb"] == 6.0 and abs(g["vram_percent"] - 8.4) < 0.3
    assert g["util_percent"] == 6.0


def test_usage_multi_gpu_in_order(monkeypatch):
    monkeypatch.setattr(hw, "psutil", None)
    monkeypatch.setattr(hw, "_run", lambda cmd: "0, A, 100, 1000, 10\n1, B, 200, 2000, 20\n")
    g = hw.usage()["gpus"]
    assert [x["index"] for x in g] == [0, 1] and g[1]["name"] == "B"


def test_usage_no_nvidia_smi(monkeypatch):
    monkeypatch.setattr(hw, "psutil", None)
    monkeypatch.setattr(hw, "_run", lambda cmd: None)
    u = hw.usage()
    assert u["gpus"] == [] and u["cpu_percent"] == 0.0


def test_usage_tolerates_na_and_zero_total(monkeypatch):
    monkeypatch.setattr(hw, "psutil", None)
    monkeypatch.setattr(hw, "_run", lambda cmd: "0, Uni, [N/A], [N/A], [N/A]\n1, G, 0, 0, 0\n")
    g = hw.usage()["gpus"]
    assert [x["index"] for x in g] == [1] and g[0]["vram_percent"] == 0.0
