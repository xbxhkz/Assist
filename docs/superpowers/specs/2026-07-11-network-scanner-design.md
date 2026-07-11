# Network Scanner — Design Spec

**Status:** Approved for planning (2026-07-11)
**Sub-project 3 of 5** in the desktop-agent initiative (see the Desktop Control spec's "Program context"). Independent of the screen/input/operator cluster.

## Goal

Give the Assist agent a **read-only inventory of the user's own local network**: which devices are up (IP, MAC, hostname, best-guess OS and vendor), which ports they expose, and this machine's own network configuration — entirely local, stdlib-only, admin-gated, and hard-restricted to private/local address ranges.

## Scope

**In scope:** three read-only agent tools (`net_info`, `discover_hosts`, `scan_ports`) over a new injectable `src/desktop/netscan.py` backend; a backend-enforced private-range guardrail with per-scan bounds; a small bundled MAC-OUI→vendor table; registration across the tool system's six sites under a new `network` domain.

**Out of scope (YAGNI / not this sub-project):** raw-packet / SYN scanning, nmap or scapy, Npcap; authoritative OS stack fingerprinting (we do best-guess heuristics only); scanning public/internet ranges; continuous monitoring or change-alerting; vulnerability assessment; IPv6 host enumeration (IPv6 subnets are too large to sweep — `net_info` may *report* IPv6 addresses but discovery/port-scan target IPv4). No new session-consent toggle (admin gate + private-range restriction is the guard).

## Architecture

Mirror the shipped `src/desktop/` pattern: a backend module with **injectable primitives so unit tests never touch a real network**, and thin tool wrappers that parse args, enforce gates, call the backend, and format results.

- **`src/desktop/netscan.py` (new)** — pure logic over injected primitives:
  - `_require_private(target) -> None` — raise `ValueError` unless `target` is contained in a private block: RFC1918 (10/8, 172.16/12, 192.168/16), link-local (169.254/16), or loopback (127/8). For a single IP, "contained" means membership; for a **CIDR, the ENTIRE network must be a subnet of one private block** (use `ipaddress`'s `subnet_of`) — so a range that merely overlaps private space but also spans public addresses (e.g. `0.0.0.0/0`) is refused, never partially scanned. The single chokepoint every scan path calls first.
  - `local_networks(*, run=<subprocess runner>) -> list[dict]` — parse `ipconfig /all` into `[{iface, ipv4, cidr, gateway, dns}]`; helper for `net_info` and for defaulting `discover_hosts` to the machine's own subnet(s).
  - `arp_table(*, run=…) -> dict[str, str]` — parse `arp -a` into `{ip: mac}`.
  - `net_info(*, run=…) -> dict` — `{interfaces: local_networks(...), neighbors: [{ip, mac, vendor?}]}` (neighbors from the ARP table + OUI vendor lookup).
  - `discover_hosts(cidr, *, probe=…, resolve=…, arp=…, oui=…, max_hosts=1024, concurrency=100, timeout=0.5) -> list[dict]` — `_require_private(cidr)`; reject a CIDR whose host count exceeds `max_hosts`; enumerate hosts; liveness via the injected TCP-connect `probe(ip) -> (up: bool, ttl: int|None)` on a small common-port set, merged with the ARP table (an ARP entry ⇒ up); for each up host: `resolve(ip)` reverse-DNS hostname, MAC from `arp`, `oui(mac)` best-effort vendor, and `_os_guess(ttl, open_ports)` best-guess OS. Concurrency- and timeout-bounded. Returns `[{ip, mac, hostname, vendor, os_guess}]` (fields absent/`None` when unknown).
  - `scan_ports(host, ports, *, connect=…, concurrency=100, timeout=0.5, max_ports=1024) -> list[dict]` — `_require_private(host)`; cap `ports` at `max_ports`; TCP-connect each via the injected `connect(host, port, timeout) -> bool`; return `[{port, open, service}]` (service from a well-known-port map).
  - `_os_guess(ttl, open_ports) -> str` — heuristic: TTL≈128→"Windows", ≈64→"Linux/Unix/Android", ≈255→"network device"; refined by port signatures (445/139→Windows, 22→Linux/SSH, 5353/548→Apple, 9100→printer). Returns `"unknown"` when nothing matches.
  - Default primitives (real ones) are module-level and injectable: `_tcp_probe`, `_reverse_dns`, `_tcp_connect`, `_run` (subprocess with `CREATE_NO_WINDOW`), `_oui_lookup`.
- **`src/desktop/oui.py` (new)** — a small curated `OUI_VENDORS: dict[str, str]` mapping the first 3 MAC octets (e.g. `"XX:XX:XX"`, uppercase) to common home/office vendors (Apple, Samsung, Intel, TP-Link, Ubiquiti, Raspberry Pi, Amazon, Google, Espressif, …). `oui_vendor(mac) -> str|None`. Best-effort — a miss returns `None`, never an error.
- **`src/agent_tools/netscan_tools.py` (new)** — `NetInfoTool`, `DiscoverHostsTool`, `ScanPortsTool`, standard `async execute(content, ctx) -> {output|error, exit_code}`.

**Registration** (the six sites the tool system requires, per the `capture_screen` / input-tools precedent, enforced by a guard test): `src/agent_tools/__init__.py` (`TOOL_HANDLERS`, `TOOL_TAGS`), `src/agent_loop.py` (`TOOL_SECTIONS` + a new `network` domain in the domain map + `_DOMAIN_RULES`), `src/tool_schemas.py` (function schemas + content-marshalling tuple), `src/tool_index.py` (`BUILTIN_TOOL_DESCRIPTIONS`), `src/tool_execution.py` (direct-dispatch branch), `src/tool_security.py` (all three in `NON_ADMIN_BLOCKED_TOOLS` **and** in `PLAN_MODE_READONLY_TOOLS` — they are read-only).

## The three tools

1. **`net_info`** — `{}`. This machine's network config: interfaces (name, IPv4, subnet CIDR, gateway, DNS) + ARP neighbors (ip, mac, best-effort vendor). Read-only, admin-gated.
2. **`discover_hosts`** — `{"cidr": "192.168.1.0/24"}` (optional; defaults to the machine's own auto-detected subnet(s) from `local_networks`). Validates the target is private and not larger than `max_hosts`; returns the live-device inventory `[{ip, mac, hostname, vendor, os_guess}]`. Read-only, admin-gated.
3. **`scan_ports`** — `{"host": "192.168.1.10", "ports": [22,80,443] | "common"}` (ports optional; default = a curated common set). Validates the host is private; returns `[{port, open, service}]` for the open ports. Read-only, admin-gated.

Each tool, given a public/non-private target, refuses with a clear message (e.g. `"scan_ports: only private/local addresses may be scanned (got 8.8.8.8)"`).

## Permission / guardrail model

The chosen model is **admin gating + a hard backend private-range restriction + bounds** — no session-consent toggle (scanning is read-only inventory; this is consistent with the other read-only desktop tools `find_files` / `list_windows`, which are admin-only but not behind a session switch).

- **Private-range only (backend-enforced):** every scan path calls `_require_private` first; public IPs/CIDRs are refused. This is enforced in the backend, not merely the prompt, so a jailbroken prompt still cannot scan the internet.
- **Default to own network:** `discover_hosts` with no `cidr` targets the machine's own subnet(s).
- **Bounded, non-aggressive:** max hosts per discover (`/22` ≈ 1024, larger refused), max ports per scan (1024), concurrency cap (~100 threads via `concurrent.futures`), short per-connection timeout (~0.5 s), and an overall wall-clock cap. Plain TCP-connect probes — nothing stealthy, evasive, or spoofed.
- **Admin-gated:** all three tools in `NON_ADMIN_BLOCKED_TOOLS`.
- **Plan mode:** all three are read-only → in `PLAN_MODE_READONLY_TOOLS` (usable for investigation in plan mode).
- **Auditability:** every scan logs one INFO line to `app.log` (tool + target + result counts).

## Dependency strategy

- **Stdlib only:** `socket` (TCP connect + reverse DNS), `ipaddress` (private-range checks, CIDR enumeration + size bound), `subprocess` (`ipconfig`/`arp` with `CREATE_NO_WINDOW`), `concurrent.futures` (bounded parallel probing). Optional **`ctypes`** (iphlpapi `GetAdaptersAddresses`) only if `ipconfig` parsing proves fragile at implementation time — a fallback, not a v1 requirement.
- **No third-party dependency, no raw sockets, no Npcap/nmap** — so there is **no frozen-build dependency risk** (unlike the comtypes path in sub-project 2). The bundled OUI table is a plain Python dict in `src/desktop/oui.py` (a few hundred entries, best-effort).
- All subprocess calls use `CREATE_NO_WINDOW`; nothing spawns `sys.executable` (no fork risk).

## Data flow

```
model → tool call → netscan tool .execute
   → admin gate (NON_ADMIN_BLOCKED_TOOLS, enforced by the tool layer)
   → src/desktop/netscan  → _require_private(target)  [refuse public]
                          → bounded TCP-connect probes / ipconfig+arp parse
   → structured result  ──────────────► back to the model
```

## Testing strategy (TDD, injected primitives — no real network / subprocess)

- **`_require_private`:** accepts `192.168.1.0/24`, `10.0.5.10`, `172.16.0.0/12`, `169.254.1.1`, `127.0.0.1`; rejects `8.8.8.8`, `1.1.1.1/24`, `93.184.216.34`.
- **`discover_hosts`:** with an injected `probe`/`arp`/`resolve`/`oui`, returns the merged inventory (ARP-only host counts as up; probe-up host resolved + vendored + os-guessed); **refuses an oversized CIDR** (e.g. `/8` > `max_hosts`) with a clear error; a public CIDR raises.
- **`scan_ports`:** injected `connect` returns open/closed per port → only open ports reported with the right service names; a public host raises; `ports` capped at `max_ports`.
- **`net_info`:** injected `run` returns canned `ipconfig /all` + `arp -a` text → parsed into interfaces (ipv4/cidr/gateway/dns) + neighbors with vendors.
- **`_os_guess`:** TTL 128→Windows, 64→Linux, 255→network device; port-signature refinement (445→Windows, 22→Linux); unknown→"unknown".
- **`oui.py`:** `oui_vendor("A4:83:E7:...")` → the mapped vendor; an unknown prefix → `None`.
- **Tools:** `discover_hosts` defaults to own subnet when `cidr` omitted (injected `local_networks`); public target → refusal message; results formatted; admin gating via the registration guard.
- **Registration guard test** (like the input-tools guard): the three tools present at all six sites; all three in `NON_ADMIN_BLOCKED_TOOLS` and in `PLAN_MODE_READONLY_TOOLS`.
- **Live verification on the packaged exe:** `net_info` shows the real interfaces/gateway/DNS + ARP neighbors; `discover_hosts` (no cidr) finds real devices on the user's LAN with hostnames/vendors; `scan_ports` on the router shows its open ports; a public IP is refused.

## Security considerations

This is the user's own machine and the user's own private network, operated by the admin user of a local-first app — a personal network-inventory / defensive capability. Guardrails: admin-only tools; a hard, backend-enforced private-range restriction (public ranges are impossible to reach, not merely discouraged); bounded, non-evasive plain TCP-connect scans with concurrency/timeout/host caps; full audit logging; and read-only semantics (the scanner changes nothing on any device). No raw-socket/spoofing/stealth capability is provided.

## Plan-time verifications (flagged, not hidden)

1. Confirm `ipconfig /all` and `arp -a` output parse reliably into `{iface, ipv4, cidr, gateway, dns}` / `{ip: mac}` on the target Windows build (localized Windows can translate field labels — parse on structure/format, not localized words where possible; fall back to `ctypes` `GetAdaptersAddresses` if fragile).
2. Confirm a bounded `concurrent.futures` TCP-connect sweep of a `/24` completes within the wall-clock cap on the user's machine and does not exhaust sockets/file handles.
3. Confirm the private-range guard covers the CIDR case correctly (a CIDR that merely *overlaps* private space but includes public addresses — e.g. a bad `0.0.0.0/0` — is refused, not partially scanned).

## Program context

Sub-project 3 of 5. Independent (no dependency on Desktop Control / Input Automation). Remaining after this: **AI Operator** (#5 — the continuous screen-watch loop that consumes Desktop Control's screen reading + Input Automation's actuators + a local VLM). Shipped: #1 Desktop Control, #2 Input Automation, #4 Plugin/Connector Hub.
