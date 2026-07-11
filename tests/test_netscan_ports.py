import pytest
import src.desktop.netscan as ns


def test_scan_ports_reports_open_with_service():
    opened = {22, 80}
    def connect(host, port, timeout=0.5):
        return port in opened
    got = ns.scan_ports("192.168.1.10", [22, 80, 3389], connect=connect, concurrency=4)
    d = {g["port"]: g for g in got}
    assert set(d) == {22, 80}  # only open ports
    assert d[22]["service"] == "ssh" and d[80]["service"] == "http"
    assert all(g["open"] for g in got)


def test_scan_ports_common_default():
    got = ns.scan_ports("10.0.0.5", connect=lambda h, p, timeout=0.5: p == 443, concurrency=4)
    assert [g["port"] for g in got] == [443]


def test_scan_ports_rejects_public_host():
    with pytest.raises(ValueError):
        ns.scan_ports("8.8.8.8", [80], connect=lambda h, p, timeout=0.5: True)


def test_scan_ports_caps_port_count():
    with pytest.raises(ValueError):
        ns.scan_ports("192.168.1.10", list(range(1, 2000)),
                      connect=lambda h, p, timeout=0.5: False, max_ports=1024)
