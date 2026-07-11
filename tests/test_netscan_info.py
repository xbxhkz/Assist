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
