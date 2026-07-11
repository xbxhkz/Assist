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
