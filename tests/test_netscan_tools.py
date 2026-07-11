import asyncio
import json
import src.agent_tools.netscan_tools as it


def test_net_info_formats(monkeypatch):
    monkeypatch.setattr(it.netscan, "net_info",
                        lambda: {"interfaces": [{"iface": "Ethernet", "ipv4": "192.168.1.10",
                                                 "cidr": "192.168.1.0/24", "gateway": "192.168.1.1",
                                                 "dns": ["192.168.1.1"]}],
                                 "neighbors": [{"ip": "192.168.1.1", "mac": "B8:27:EB:11:22:33",
                                                "vendor": "Raspberry Pi Foundation"}]})
    r = asyncio.run(it.NetInfoTool().execute("{}", {}))
    assert r["exit_code"] == 0 and "192.168.1.0/24" in r["output"] and "Raspberry Pi" in r["output"]


def test_discover_defaults_to_own_subnet(monkeypatch):
    monkeypatch.setattr(it.netscan, "local_networks",
                        lambda: [{"iface": "Ethernet", "ipv4": "192.168.1.10",
                                  "cidr": "192.168.1.0/24", "gateway": "", "dns": []}])
    seen = {}
    def fake_discover(cidr, **k):
        seen["cidr"] = cidr
        return [{"ip": "192.168.1.1", "mac": None, "hostname": None, "vendor": None, "os_guess": "unknown"}]
    monkeypatch.setattr(it.netscan, "discover_hosts", fake_discover)
    r = asyncio.run(it.DiscoverHostsTool().execute("{}", {}))
    assert r["exit_code"] == 0 and seen["cidr"] == "192.168.1.0/24"


def test_discover_public_cidr_refused(monkeypatch):
    def boom(cidr, **k):
        raise ValueError("only private/local addresses may be scanned (got 8.8.8.0/24)")
    monkeypatch.setattr(it.netscan, "discover_hosts", boom)
    r = asyncio.run(it.DiscoverHostsTool().execute(json.dumps({"cidr": "8.8.8.0/24"}), {}))
    assert r["exit_code"] == 1 and "private" in r["error"].lower()


def test_scan_ports_requires_host(monkeypatch):
    r = asyncio.run(it.ScanPortsTool().execute("{}", {}))
    assert r["exit_code"] == 1 and "host" in r["error"].lower()


def test_scan_ports_formats(monkeypatch):
    monkeypatch.setattr(it.netscan, "scan_ports",
                        lambda host, ports="common", **k: [{"port": 22, "open": True, "service": "ssh"}])
    r = asyncio.run(it.ScanPortsTool().execute(json.dumps({"host": "192.168.1.10"}), {}))
    assert r["exit_code"] == 0 and "22" in r["output"] and "ssh" in r["output"]
