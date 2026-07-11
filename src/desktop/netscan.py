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
