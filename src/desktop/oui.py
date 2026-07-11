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
