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
