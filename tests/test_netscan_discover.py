import pytest
import src.desktop.netscan as ns


def test_os_guess_from_port_signatures():
    assert ns._os_guess([445, 139]) == "Windows"
    assert ns._os_guess([22]) == "Linux/Unix"
    assert ns._os_guess([9100]) == "Printer"
    assert ns._os_guess([]) == "unknown"


def test_os_guess_ttl_fallback():
    assert ns._os_guess([], ttl=128) == "Windows"
    assert ns._os_guess([], ttl=64) == "Linux/Unix"
    assert ns._os_guess([], ttl=255) == "Network device"


def test_discover_merges_probe_and_arp():
    up = {"192.168.1.1": (True, [445]), "192.168.1.5": (True, [22])}
    def probe(ip, timeout=0.5):
        return up.get(ip, (False, []))
    def resolve(ip):
        return {"192.168.1.1": "router.lan"}.get(ip)
    arp = lambda: {"192.168.1.1": "B8:27:EB:11:22:33", "192.168.1.9": "AA:BB:CC:DD:EE:FF"}
    got = ns.discover_hosts("192.168.1.0/24", probe=probe, resolve=resolve,
                            arp=arp, max_hosts=1024, concurrency=8)
    ips = {g["ip"]: g for g in got}
    # probe-up hosts AND arp-only host (192.168.1.9) are all reported
    assert set(ips) == {"192.168.1.1", "192.168.1.5", "192.168.1.9"}
    assert ips["192.168.1.1"]["hostname"] == "router.lan"
    assert ips["192.168.1.1"]["vendor"] == "Raspberry Pi Foundation"
    assert ips["192.168.1.1"]["os_guess"] == "Windows"
    assert ips["192.168.1.5"]["os_guess"] == "Linux/Unix"


def test_discover_rejects_oversized_cidr():
    with pytest.raises(ValueError):
        ns.discover_hosts("10.0.0.0/8", probe=lambda ip, timeout=0.5: (False, []),
                          arp=lambda: {}, max_hosts=1024)


def test_discover_rejects_public_cidr():
    with pytest.raises(ValueError):
        ns.discover_hosts("8.8.8.0/24", probe=lambda ip, timeout=0.5: (False, []),
                          arp=lambda: {})
