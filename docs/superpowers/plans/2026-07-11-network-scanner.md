# Network Scanner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three read-only agent tools — `net_info`, `discover_hosts`, `scan_ports` — that inventory the user's OWN local network (devices, ports, this machine's config), stdlib-only, admin-gated, hard-restricted to private address ranges.

**Architecture:** An injectable-primitive backend `src/desktop/netscan.py` (pure logic; tests never touch a real network/subprocess) plus a small bundled `src/desktop/oui.py` vendor table, wrapped by three thin tools in `src/agent_tools/netscan_tools.py`, registered across the tool system's 6 sites under a new `network` domain.

**Tech Stack:** Python stdlib only — `socket`, `ipaddress`, `subprocess` (CREATE_NO_WINDOW), `concurrent.futures`; pytest (`--import-mode=importlib`). No third-party deps, no raw sockets, no Npcap/nmap.

## Global Constraints

- All pytest runs use `--import-mode=importlib` (a global `ultralytics` package shadows `tests/`).
- Every unit test must pass without a real network or subprocess — every backend function takes injectable primitives; the real socket/subprocess boundary is exercised only at the packaging live-verify.
- **Private-range only, backend-enforced:** every scan path calls `_require_private` first; public IPs/CIDRs are refused with a clear error (not merely discouraged in the prompt). A CIDR must be fully `subnet_of` a private block.
- **Bounds:** discover refuses a CIDR with more than `max_hosts=1024` hosts; scan caps at `max_ports=1024`; concurrency `100`; per-connection `timeout=0.5`s. Plain TCP-connect probes — nothing stealthy/spoofed.
- **No session toggle.** Guard = admin gate + private-range restriction. All three tools go in `NON_ADMIN_BLOCKED_TOOLS` AND in `PLAN_MODE_READONLY_TOOLS` (read-only). Every scan logs one INFO line.
- Follow the shipped desktop/input tools as the registration precedent verbatim; anchor edits on the stable `capture_screen` / `keyboard` entries (line numbers shifted after input-automation).
- Stdlib only — do NOT add any dependency to requirements.txt or Assist.spec.

---

### Task 1: `src/desktop/oui.py` — bundled MAC-OUI→vendor table

**Files:**
- Create: `src/desktop/oui.py`
- Test: `tests/test_oui.py`

**Interfaces:**
- Produces: `OUI_VENDORS: dict[str,str]` (first-3-octet prefix `"AA:BB:CC"` uppercase → vendor); `oui_vendor(mac: str|None) -> str|None`.

- [ ] **Step 1 — failing test** `tests/test_oui.py`:

```python
import src.desktop.oui as oui


def test_known_prefix_maps_to_vendor():
    assert oui.oui_vendor("B8:27:EB:12:34:56") == "Raspberry Pi Foundation"


def test_accepts_dash_and_lowercase():
    assert oui.oui_vendor("b8-27-eb-aa-bb-cc") == "Raspberry Pi Foundation"


def test_unknown_prefix_is_none():
    assert oui.oui_vendor("02:00:00:00:00:00") is None


def test_malformed_mac_is_none():
    assert oui.oui_vendor("") is None
    assert oui.oui_vendor(None) is None
    assert oui.oui_vendor("zz") is None


def test_table_keys_are_uppercase_prefixes():
    for k in oui.OUI_VENDORS:
        assert k == k.upper() and len(k.split(":")) == 3
```

- [ ] **Step 2 — run, expect FAIL:** `python -m pytest tests/test_oui.py --import-mode=importlib -q`
- [ ] **Step 3 — implement** `src/desktop/oui.py`:

```python
"""Best-effort MAC-OUI -> vendor lookup. A curated subset of common home/office
device vendors; a miss returns None (never an error). Extend OUI_VENDORS as
needed — accuracy is best-guess, not authoritative."""

# First three octets (uppercase, colon-separated) -> vendor. Real IEEE OUI
# assignments for common consumer/network devices.
OUI_VENDORS = {
    "B8:27:EB": "Raspberry Pi Foundation",
    "DC:A6:32": "Raspberry Pi Trading",
    "E4:5F:01": "Raspberry Pi Trading",
    "A4:83:E7": "Apple",
    "F0:18:98": "Apple",
    "3C:15:C2": "Apple",
    "AC:DE:48": "Apple",
    "24:0A:C4": "Espressif (ESP32)",
    "30:AE:A4": "Espressif (ESP32)",
    "50:C7:BF": "TP-Link",
    "F4:F2:6D": "TP-Link",
    "FC:EC:DA": "Ubiquiti",
    "24:5A:4C": "Ubiquiti",
    "00:1B:21": "Intel",
    "00:1A:11": "Google",
    "3C:5A:B4": "Google",
    "F4:F5:D8": "Google (Nest)",
    "44:65:0D": "Amazon",
    "68:37:E9": "Amazon",
    "EC:1A:59": "Belkin",
    "00:17:88": "Philips (Hue)",
    "B0:39:56": "Netgear",
    "A0:63:91": "Netgear",
    "00:0C:29": "VMware",
    "08:00:27": "VirtualBox",
    "52:54:00": "QEMU/KVM",
    "00:50:56": "VMware",
    "1C:69:7A": "EliteGroup/ECS",
    "D8:3A:DD": "Raspberry Pi Trading",
}


def oui_vendor(mac):
    if not mac:
        return None
    parts = str(mac).replace("-", ":").upper().split(":")
    if len(parts) < 3 or not all(len(p) == 2 for p in parts[:3]):
        return None
    return OUI_VENDORS.get(":".join(parts[:3]))
```

- [ ] **Step 4 — run, expect PASS.** **Step 5 — commit** (stage only the two files): `feat(netscan): bundled OUI vendor table`.

---

### Task 2: `netscan.py` — `_require_private` guard

**Files:**
- Create: `src/desktop/netscan.py`
- Test: `tests/test_netscan_guard.py`

**Interfaces:**
- Produces: `_require_private(target: str) -> None` (raises `ValueError` unless `target` — an IP or CIDR — is fully within a private block).

- [ ] **Step 1 — failing test** `tests/test_netscan_guard.py`:

```python
import pytest
import src.desktop.netscan as ns


@pytest.mark.parametrize("target", [
    "192.168.1.0/24", "10.0.5.10", "172.16.0.0/12", "169.254.1.1",
    "127.0.0.1", "192.168.1.50",
])
def test_private_targets_pass(target):
    ns._require_private(target)  # no raise


@pytest.mark.parametrize("target", [
    "8.8.8.8", "1.1.1.1/24", "93.184.216.34", "0.0.0.0/0", "172.32.0.0/16",
])
def test_public_or_spanning_targets_raise(target):
    with pytest.raises(ValueError):
        ns._require_private(target)


def test_invalid_target_raises():
    with pytest.raises(ValueError):
        ns._require_private("not-an-ip")
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** the top of `src/desktop/netscan.py`:

```python
"""Local-network inventory backend. Pure logic over injectable primitives so
unit tests never touch a real network or subprocess. Every scan path calls
_require_private first: only private/local ranges may ever be reached."""
import ipaddress
import logging
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor

from src.desktop.oui import oui_vendor

logger = logging.getLogger(__name__)

_PRIVATE_BLOCKS = [ipaddress.ip_network(n) for n in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
    "169.254.0.0/16", "127.0.0.0/8",
)]


def _require_private(target):
    """Raise ValueError unless `target` (IP or CIDR string) is fully within a
    private block. For a CIDR the ENTIRE network must be subnet_of a private
    block, so a range that spans public space (e.g. 0.0.0.0/0) is refused."""
    t = str(target).strip()
    try:
        if "/" in t:
            net = ipaddress.ip_network(t, strict=False)
            ok = any(net.version == b.version and net.subnet_of(b) for b in _PRIVATE_BLOCKS)
        else:
            ip = ipaddress.ip_address(t)
            ok = any(ip in b for b in _PRIVATE_BLOCKS)
    except ValueError as e:
        raise ValueError(f"invalid scan target {target!r}: {e}")
    if not ok:
        raise ValueError(f"only private/local addresses may be scanned (got {target})")
```

- [ ] **Step 4 — run, expect PASS.** **Step 5 — commit** `feat(netscan): private-range guard (_require_private)`.

---

### Task 3: `netscan.py` — `net_info` (interfaces + ARP) via injectable runner

**Files:**
- Modify: `src/desktop/netscan.py`
- Test: `tests/test_netscan_info.py`

**Interfaces:**
- Consumes: `oui_vendor` (Task 1).
- Produces: `_run(args) -> str` (default subprocess runner, CREATE_NO_WINDOW; injectable); `local_networks(*, run=_run) -> list[dict{iface, ipv4, cidr, gateway, dns}]`; `arp_table(*, run=_run) -> dict[str,str]` (ip→MAC uppercase-colon); `net_info(*, run=_run, oui=oui_vendor) -> dict{interfaces, neighbors}`.

- [ ] **Step 1 — failing test** `tests/test_netscan_info.py`:

```python
import src.desktop.netscan as ns

FAKE_IPCONFIG = """
Windows IP Configuration

Ethernet adapter Ethernet:
   IPv4 Address. . . . . . . . . . . : 192.168.1.10(Preferred)
   Subnet Mask . . . . . . . . . . . : 255.255.255.0
   Default Gateway . . . . . . . . . : 192.168.1.1
   DNS Servers . . . . . . . . . . . : 192.168.1.1

Wireless LAN adapter Wi-Fi:
   Media State . . . . . . . . . . . : Media disconnected
"""

FAKE_ARP = """
Interface: 192.168.1.10 --- 0x5
  Internet Address      Physical Address      Type
  192.168.1.1           b8-27-eb-11-22-33     dynamic
  192.168.1.20          02-00-00-00-00-00     dynamic
"""


def _fake_run(args):
    return FAKE_IPCONFIG if "ipconfig" in args[0] else FAKE_ARP


def test_local_networks_parses_ipv4_cidr_gateway_dns():
    nets = ns.local_networks(run=_fake_run)
    assert any(n["ipv4"] == "192.168.1.10" and n["cidr"] == "192.168.1.0/24"
               and n["gateway"] == "192.168.1.1" and "192.168.1.1" in n["dns"]
               for n in nets)


def test_arp_table_parses_ip_to_mac_uppercase_colon():
    t = ns.arp_table(run=_fake_run)
    assert t["192.168.1.1"] == "B8:27:EB:11:22:33"


def test_net_info_merges_interfaces_and_vendored_neighbors():
    info = ns.net_info(run=_fake_run)
    assert info["interfaces"]
    n = {x["ip"]: x for x in info["neighbors"]}
    assert n["192.168.1.1"]["mac"] == "B8:27:EB:11:22:33"
    assert n["192.168.1.1"]["vendor"] == "Raspberry Pi Foundation"
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** — append to `src/desktop/netscan.py`:

```python
def _run(args):
    """Run a console command and return stdout text (CREATE_NO_WINDOW so no
    console flashes in the windowed build). Injectable in tests."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    out = subprocess.run(args, capture_output=True, text=True, timeout=15,
                         creationflags=flags)
    return out.stdout or ""


def _norm_mac(raw):
    return raw.replace("-", ":").upper().strip()


def _value_after_colon(line):
    # ipconfig uses dotted leaders then ": value"; take everything after the
    # last ": " and strip a "(Preferred)" suffix.
    if ":" not in line:
        return ""
    return line.split(":", 1)[1].strip().split("(")[0].strip()


def local_networks(*, run=_run):
    text = run(["ipconfig", "/all"])
    nets, cur = [], None
    for line in text.splitlines():
        if line and not line.startswith(" ") and "adapter" in line.lower():
            if cur and cur.get("ipv4"):
                nets.append(cur)
            cur = {"iface": line.rstrip(":").strip(), "ipv4": "", "cidr": "",
                   "gateway": "", "dns": []}
            continue
        if cur is None:
            continue
        low = line.lower()
        if "ipv4 address" in low:
            cur["ipv4"] = _value_after_colon(line)
        elif "subnet mask" in low and cur["ipv4"]:
            mask = _value_after_colon(line)
            try:
                cur["cidr"] = str(ipaddress.ip_interface(f"{cur['ipv4']}/{mask}").network)
            except ValueError:
                cur["cidr"] = ""
        elif "default gateway" in low:
            gw = _value_after_colon(line)
            if gw:
                cur["gateway"] = gw
        elif "dns servers" in low:
            dns = _value_after_colon(line)
            if dns:
                cur["dns"].append(dns)
    if cur and cur.get("ipv4"):
        nets.append(cur)
    return nets


def _looks_ipv4(s):
    try:
        ipaddress.IPv4Address(s)
        return True
    except ValueError:
        return False


def arp_table(*, run=_run):
    text = run(["arp", "-a"])
    table = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and _looks_ipv4(parts[0]) and ("-" in parts[1] or ":" in parts[1]):
            table[parts[0]] = _norm_mac(parts[1])
    return table


def net_info(*, run=_run, oui=oui_vendor):
    ifaces = local_networks(run=run)
    neighbors = [{"ip": ip, "mac": mac, "vendor": oui(mac)}
                 for ip, mac in sorted(arp_table(run=run).items())]
    return {"interfaces": ifaces, "neighbors": neighbors}
```

- [ ] **Step 4 — run, expect PASS.** **Step 5 — commit** `feat(netscan): net_info (interfaces + ARP neighbors)`.

---

### Task 4: `netscan.py` — `discover_hosts` + `_os_guess`

**Files:**
- Modify: `src/desktop/netscan.py`
- Test: `tests/test_netscan_discover.py`

**Interfaces:**
- Consumes: `_require_private`, `arp_table`, `oui_vendor`.
- Produces: `_LIVENESS_PORTS`; `_tcp_probe(ip, timeout=0.5) -> (up: bool, open_ports: list[int])`; `_reverse_dns(ip) -> str|None`; `_os_guess(open_ports, ttl=None) -> str`; `discover_hosts(cidr, *, probe=_tcp_probe, resolve=_reverse_dns, arp=arp_table, oui=oui_vendor, max_hosts=1024, concurrency=100, timeout=0.5) -> list[dict{ip, mac, hostname, vendor, os_guess}]`. `probe`/`resolve` take one host; `arp` is a zero-arg callable returning `{ip:mac}`.

- [ ] **Step 1 — failing test** `tests/test_netscan_discover.py`:

```python
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
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** — append to `src/desktop/netscan.py`:

```python
_LIVENESS_PORTS = (445, 139, 22, 80, 443, 3389, 5353, 9100)

# First matching open port wins (priority order).
_OS_SIGNATURES = [
    (445, "Windows"), (139, "Windows"), (3389, "Windows"),
    (548, "Apple/macOS"), (5353, "Apple/mDNS device"),
    (9100, "Printer"), (22, "Linux/Unix"), (23, "Network device"),
]


def _os_guess(open_ports, ttl=None):
    op = set(open_ports or [])
    for port, name in _OS_SIGNATURES:
        if port in op:
            return name
    if ttl is not None:
        if ttl >= 200:
            return "Network device"
        if ttl >= 100:
            return "Windows"
        if ttl > 0:
            return "Linux/Unix"
    return "unknown"


def _tcp_connect(host, port, timeout=0.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


def _tcp_probe(ip, timeout=0.5):
    open_ports = [p for p in _LIVENESS_PORTS if _tcp_connect(ip, p, timeout)]
    return (bool(open_ports), open_ports)


def _reverse_dns(ip):
    try:
        return socket.gethostbyaddr(ip)[0]
    except (OSError, socket.herror):
        return None


def discover_hosts(cidr, *, probe=_tcp_probe, resolve=_reverse_dns, arp=arp_table,
                   oui=oui_vendor, max_hosts=1024, concurrency=100, timeout=0.5):
    _require_private(cidr)
    net = ipaddress.ip_network(cidr, strict=False)
    hosts = [str(h) for h in net.hosts()]
    if len(hosts) > max_hosts:
        raise ValueError(f"CIDR too large: {len(hosts)} hosts exceeds max {max_hosts}")
    arp_map = arp() if callable(arp) else dict(arp or {})

    def _check(ip):
        up, open_ports = probe(ip, timeout)
        return (ip, open_ports) if (up or ip in arp_map) else None

    found = {}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for r in ex.map(_check, hosts):
            if r:
                found[r[0]] = r[1]
    out = []
    for ip in sorted(found, key=lambda s: ipaddress.IPv4Address(s)):
        mac = arp_map.get(ip)
        out.append({
            "ip": ip, "mac": mac, "hostname": resolve(ip),
            "vendor": oui(mac) if mac else None,
            "os_guess": _os_guess(found[ip]),
        })
    logger.info("discover_hosts %s -> %d host(s)", cidr, len(out))
    return out
```

- [ ] **Step 4 — run, expect PASS.** **Step 5 — commit** `feat(netscan): discover_hosts + OS heuristic`.

---

### Task 5: `netscan.py` — `scan_ports` + service map

**Files:**
- Modify: `src/desktop/netscan.py`
- Test: `tests/test_netscan_ports.py`

**Interfaces:**
- Consumes: `_require_private`, `_tcp_connect`.
- Produces: `COMMON_PORTS` (list); `_SERVICES` (port→name); `scan_ports(host, ports="common", *, connect=_tcp_connect, concurrency=100, timeout=0.5, max_ports=1024) -> list[dict{port, open, service}]` (only OPEN ports returned).

- [ ] **Step 1 — failing test** `tests/test_netscan_ports.py`:

```python
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
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** — append to `src/desktop/netscan.py`:

```python
_SERVICES = {
    21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns", 80: "http",
    110: "pop3", 135: "msrpc", 139: "netbios", 143: "imap", 443: "https",
    445: "smb", 515: "printer", 548: "afp", 631: "ipp", 993: "imaps",
    995: "pop3s", 1433: "mssql", 3306: "mysql", 3389: "rdp", 5353: "mdns",
    5432: "postgres", 5900: "vnc", 8080: "http-alt", 8443: "https-alt",
    9100: "jetdirect", 32400: "plex",
}
COMMON_PORTS = sorted(_SERVICES)


def scan_ports(host, ports="common", *, connect=_tcp_connect, concurrency=100,
               timeout=0.5, max_ports=1024):
    _require_private(host)
    plist = COMMON_PORTS if ports in (None, "common") else [int(p) for p in ports]
    if len(plist) > max_ports:
        raise ValueError(f"too many ports: {len(plist)} exceeds max {max_ports}")

    def _check(port):
        return (port, connect(host, port, timeout))

    out = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for port, is_open in ex.map(_check, plist):
            if is_open:
                out.append({"port": port, "open": True,
                            "service": _SERVICES.get(port, "unknown")})
    out.sort(key=lambda d: d["port"])
    logger.info("scan_ports %s -> %d open", host, len(out))
    return out
```

- [ ] **Step 4 — run, expect PASS.** **Step 5 — commit** `feat(netscan): scan_ports + service map`.

---

### Task 6: `netscan_tools.py` — the three tool wrappers

**Files:**
- Create: `src/agent_tools/netscan_tools.py`
- Test: `tests/test_netscan_tools.py`

**Interfaces:**
- Consumes: `src.desktop.netscan` (`net_info`, `discover_hosts`, `scan_ports`, `local_networks`).
- Produces: `NetInfoTool`, `DiscoverHostsTool`, `ScanPortsTool` with `async execute(content, ctx) -> {output|error, exit_code}`. Module-level name `netscan` (tests monkeypatch `it.netscan.*`).

- [ ] **Step 1 — failing test** `tests/test_netscan_tools.py`:

```python
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
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `src/agent_tools/netscan_tools.py`:

```python
"""Network-inventory tools: net_info / discover_hosts / scan_ports. Read-only,
admin-gated at the registry layer, hard-restricted to private ranges in the
backend (src.desktop.netscan). Thin wrappers: parse args, call backend, format,
surface the private-range ValueError as a clean error."""
import json
import logging

from src.desktop import netscan

logger = logging.getLogger(__name__)


def _args(content):
    try:
        return json.loads(content) if content.strip().startswith("{") else {}
    except (json.JSONDecodeError, TypeError):
        return {}


class NetInfoTool:
    async def execute(self, content, ctx):
        try:
            info = netscan.net_info()
        except Exception as e:
            return {"error": f"net_info: {e}", "exit_code": 1}
        lines = []
        for n in info["interfaces"]:
            lines.append(f"{n['iface']}: {n['ipv4']} ({n['cidr']}) gw={n['gateway']} "
                         f"dns={','.join(n['dns'])}")
        lines.append(f"neighbors ({len(info['neighbors'])}):")
        for nb in info["neighbors"]:
            lines.append(f"  {nb['ip']}  {nb['mac']}  {nb.get('vendor') or ''}")
        return {"output": "\n".join(lines) or "(no network info)", "exit_code": 0}


class DiscoverHostsTool:
    async def execute(self, content, ctx):
        a = _args(content)
        cidr = (a.get("cidr") or "").strip()
        if not cidr:
            subs = [n["cidr"] for n in netscan.local_networks() if n.get("cidr")]
            if not subs:
                return {"error": "discover_hosts: could not determine local subnet; pass a cidr",
                        "exit_code": 1}
            cidr = subs[0]
        try:
            hosts = netscan.discover_hosts(cidr)
        except ValueError as e:
            return {"error": f"discover_hosts: {e}", "exit_code": 1}
        except Exception as e:
            return {"error": f"discover_hosts: {e}", "exit_code": 1}
        logger.info("discover_hosts %s -> %d", cidr, len(hosts))
        if not hosts:
            return {"output": f"No live hosts found on {cidr}", "exit_code": 0}
        lines = [f"{h['ip']}  {h.get('hostname') or ''}  {h.get('mac') or ''}  "
                 f"{h.get('vendor') or ''}  [{h['os_guess']}]" for h in hosts]
        return {"output": f"{len(hosts)} host(s) on {cidr}:\n" + "\n".join(lines), "exit_code": 0}


class ScanPortsTool:
    async def execute(self, content, ctx):
        a = _args(content)
        host = (a.get("host") or "").strip()
        if not host:
            return {"error": "scan_ports: host required", "exit_code": 1}
        ports = a.get("ports", "common")
        try:
            open_ports = netscan.scan_ports(host, ports)
        except ValueError as e:
            return {"error": f"scan_ports: {e}", "exit_code": 1}
        except Exception as e:
            return {"error": f"scan_ports: {e}", "exit_code": 1}
        logger.info("scan_ports %s -> %d open", host, len(open_ports))
        if not open_ports:
            return {"output": f"No open ports on {host}", "exit_code": 0}
        lines = [f"{p['port']}/{p['service']}" for p in open_ports]
        return {"output": f"{len(open_ports)} open on {host}: " + ", ".join(lines), "exit_code": 0}
```

- [ ] **Step 4 — run, expect PASS.** Also `python -c "import src.agent_tools.netscan_tools"`. **Step 5 — commit** `feat(netscan): net_info/discover_hosts/scan_ports tools`.

---

### Task 7: Register the three tools across the 6 sites (new `network` domain) + guard test

**Files:**
- Modify: `src/agent_tools/__init__.py`, `src/agent_loop.py`, `src/tool_schemas.py`, `src/tool_index.py`, `src/tool_execution.py`, `src/tool_security.py`
- Test: `tests/test_netscan_registration.py`

**Interfaces:**
- Consumes: the three tool classes (Task 6).
- Produces: the three tools callable through the agent path, admin-gated, plan-mode read-only.

- [ ] **Step 1 — failing guard test** `tests/test_netscan_registration.py`:

```python
import src.agent_tools as agent_tools
import src.tool_security as ts
import src.tool_index as ti
import src.agent_loop as al

NET = {"net_info", "discover_hosts", "scan_ports"}


def test_all_in_handlers_and_tags():
    for n in NET:
        assert n in agent_tools.TOOL_HANDLERS and n in agent_tools.TOOL_TAGS


def test_all_admin_blocked_and_plan_readonly():
    for n in NET:
        assert n in ts.NON_ADMIN_BLOCKED_TOOLS
        assert n in ts.PLAN_MODE_READONLY_TOOLS


def test_index_and_prompt_sections():
    for n in NET:
        assert n in ti.BUILTIN_TOOL_DESCRIPTIONS and n in al.TOOL_SECTIONS
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** the six edits (mirror the `capture_screen` / `keyboard` precedent at each site — grep those names to find the insertion points):

**`src/agent_tools/__init__.py`** — add the import near the other desktop-tool imports:
```python
from .netscan_tools import NetInfoTool, DiscoverHostsTool, ScanPortsTool
```
add to `TOOL_HANDLERS` (after the `"keyboard": KeyboardTool().execute,` line):
```python
    "net_info": NetInfoTool().execute,
    "discover_hosts": DiscoverHostsTool().execute,
    "scan_ports": ScanPortsTool().execute,
```
add the three names to the `TOOL_TAGS` set (extend the desktop/input line).

**`src/tool_security.py`** — add all three to `NON_ADMIN_BLOCKED_TOOLS` (after `"keyboard",`) AND to `PLAN_MODE_READONLY_TOOLS` (after the `"list_ui_elements",` entry):
```python
    "net_info",
    "discover_hosts",
    "scan_ports",
```
(They are read-only, so they belong in BOTH sets. Do NOT add them to `_PLAN_MODE_KNOWN_MUTATORS`.)

**`src/tool_index.py`** — add to `BUILTIN_TOOL_DESCRIPTIONS` (after the `keyboard` entry):
```python
    "net_info": "Show this machine's own network config (interfaces, IP, subnet, gateway, DNS) and ARP neighbors. Read-only.",
    "discover_hosts": "Discover live devices on a private LAN subnet (defaults to your own subnet): IP, MAC, hostname, vendor, best-guess OS. Private ranges only.",
    "scan_ports": "TCP-connect scan a private host's ports (default = common set) and report which are open. Private ranges only.",
```

**`src/tool_execution.py`** — extend the direct-dispatch tuple (the `elif tool in (... "mouse", "keyboard"):` branch) to include the three names:
```python
                 "net_info", "discover_hosts", "scan_ports"):
```

**`src/agent_loop.py`** — add a new `network` entry to the domain→tools map (the dict that has `"desktop": {...}`):
```python
    "network": {"net_info", "discover_hosts", "scan_ports"},
```
add a `network` entry to `_DOMAIN_RULES`:
```python
    "network": """\
- `net_info` shows this machine's own network; `discover_hosts`/`scan_ports` only work on PRIVATE/local ranges (public IPs are refused).
- `discover_hosts` defaults to your own subnet; scans are read-only inventory. Prefer `discover_hosts` before `scan_ports` to find a target.""",
```
add three `TOOL_SECTIONS` entries mirroring the `capture_screen` fenced style:
```python
    "net_info": """\
```net_info
{}
```
Show this machine's network config + ARP neighbors. Read-only.""",
    "discover_hosts": """\
```discover_hosts
{"cidr": "192.168.1.0/24"}  // optional; defaults to your own subnet
```
List live devices on a PRIVATE subnet (IP, MAC, hostname, vendor, best-guess OS).""",
    "scan_ports": """\
```scan_ports
{"host": "192.168.1.1", "ports": [22,80,443]}  // ports optional; default = common set
```
TCP-connect scan a PRIVATE host's ports; reports open ports + service.""",
```

**`src/tool_schemas.py`** — add three function schemas mirroring the existing desktop entries (grep `capture_screen`), and add the three names to the desktop/input content-marshalling tuple (the one ending `"mouse", "keyboard")`):
```python
        {"type": "function", "function": {
            "name": "net_info",
            "description": "Show this machine's own network config (interfaces, IP, subnet, gateway, DNS) + ARP neighbors.",
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "discover_hosts",
            "description": "Discover live devices on a PRIVATE LAN subnet (defaults to your own subnet).",
            "parameters": {"type": "object", "properties": {
                "cidr": {"type": "string", "description": "Private CIDR, e.g. 192.168.1.0/24 (optional)"}}}}},
        {"type": "function", "function": {
            "name": "scan_ports",
            "description": "TCP-connect scan a PRIVATE host's ports; report which are open.",
            "parameters": {"type": "object", "properties": {
                "host": {"type": "string", "description": "Private host IP"},
                "ports": {"description": "List of ports, or 'common' (default)"}},
                "required": ["host"]}}},
```
(Insert into the same list the existing desktop schemas live in; match the surrounding brackets exactly.)

- [ ] **Step 4 — run the guard test, expect PASS.** Then `python -c "import src.agent_tools, src.agent_loop, src.tool_schemas, src.tool_execution, src.tool_security, src.tool_index"` (import src.agent_tools FIRST — the tool_schemas↔agent_tools cluster only resolves cleanly when entered via agent_tools). Also run all netscan tests: `python -m pytest tests/test_oui.py tests/test_netscan_guard.py tests/test_netscan_info.py tests/test_netscan_discover.py tests/test_netscan_ports.py tests/test_netscan_tools.py tests/test_netscan_registration.py --import-mode=importlib -q`.
- [ ] **Step 5 — commit** `feat(netscan): register network tools across the tool system`.

---

### Task 8: Package + live-verify

- [ ] **Step 1 — full affected suite:** the pytest command from Task 7 Step 4 → all green.
- [ ] **Step 2 — build:** full clean `.\build-installer.ps1` (never `-Fast`). Confirm compile succeeds.
- [ ] **Step 3 — boot-verify the frozen exe** via `Assist.exe --run-py <probe.py> <out>`. The probe must, inside the frozen process: (a) confirm the three tools are in `TOOL_HANDLERS`; (b) call the REAL `src.desktop.netscan.net_info()` (parses this machine's real `ipconfig`/`arp`) and assert it returns at least one interface with a `cidr`; (c) assert `netscan._require_private("8.8.8.8")` raises and `_require_private("192.168.0.0/16")` does not. Print `NETSCAN_BOOT=OK` / `FAIL`. (Do NOT run a full LAN discovery in the probe — that's the user's live test.)
- [ ] **Step 4 — user manual test** (packaged exe): ask the agent `net_info` (shows real interfaces/gateway/DNS + neighbors); `discover_hosts` with no cidr (finds real devices on the LAN with hostnames/vendors/OS guesses); `scan_ports` on the router IP (shows open ports); ask it to scan a public IP like 8.8.8.8 → refused. Confirm Desktop Control + Input Automation + Plugins still work.
- [ ] **Step 5 — commit** the installer: `git add -f installer/Output/Assist-Setup.exe && git commit -m "build: Assist-Setup.exe with network scanner"`.

## Self-Review

- **Spec coverage:** OUI table (T1), `_require_private` CIDR-subnet_of guard (T2), `net_info`/`local_networks`/`arp_table` via injectable runner (T3), `discover_hosts`+`_os_guess`+bounds (T4), `scan_ports`+service map+caps (T5), the 3 tools with own-subnet default + public-refusal (T6), registration across 6 sites under a new `network` domain + admin-gate + plan-mode-readonly + guard test (T7), audit logging (T4/T5/T6 `logger.info`), stdlib-only (no requirements/spec changes), package + live-verify with the real private-range guard + net_info exercised in the frozen exe (T8). All spec sections covered.
- **Placeholders:** none — every code step carries complete code. The optional `ctypes GetAdaptersAddresses` fallback (spec plan-time-verif #1) is deliberately NOT built (ipconfig parsing is the v1 path; the fallback is a documented contingency, not a task).
- **Type consistency:** `_require_private(target)` used identically in T2/T4/T5; `discover_hosts` returns `{ip,mac,hostname,vendor,os_guess}` consistently across T4 (impl+tests) and T6 (tool tests); `scan_ports` returns `{port,open,service}` in T5 and T6; `probe(ip,timeout)->(up,open_ports)` and `arp` zero-arg callable consistent between T4 impl and its tests; the module-level name `netscan` (tool module) matches the monkeypatch targets in T6; tool names (`net_info`,`discover_hosts`,`scan_ports`) identical across T6-T8.
