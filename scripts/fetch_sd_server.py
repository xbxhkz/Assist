"""Vendor prebuilt stable-diffusion.cpp `sd-server` (CPU + Vulkan) into
build_assets/sd/ at build time.

Downloads the pinned Windows CPU (avx2) and Vulkan release zips and extracts
`sd-server.exe` (+ its DLLs) into build_assets/sd/cpu/ and build_assets/sd/vulkan/.
Vulkan gives GPU acceleration via the driver with no CUDA toolkit to bundle; a
user-installed CUDA sd-server on PATH is preferred at runtime when present.

Pinned for reproducibility — bump SD_RELEASE_TAG/HASH deliberately after
verifying a build. FLUX 2 support depends on the pinned sd.cpp version.
"""
import io
import os
import sys
import urllib.request
import zipfile

SD_TAG = os.getenv("SD_RELEASE_TAG", "master-765-bb84971")
SD_HASH = os.getenv("SD_RELEASE_HASH", "bb84971")
_BASE = f"https://github.com/leejet/stable-diffusion.cpp/releases/download/{SD_TAG}"
BUILDS = {
    "cpu": f"{_BASE}/sd-master-{SD_HASH}-bin-win-cpu-x64.zip",
    "vulkan": f"{_BASE}/sd-master-{SD_HASH}-bin-win-vulkan-x64.zip",
}
ASSET_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "build_assets", "sd")
)


def _fetch_build(name: str, url: str) -> bool:
    dest = os.path.join(ASSET_ROOT, name)
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading sd-server ({name}) from {url} ...")
    with urllib.request.urlopen(url) as resp:  # noqa: S310
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            base = os.path.basename(member)
            if not base:
                continue
            # sd-server exe + every DLL (runtime deps) at any depth.
            if base == "sd-server.exe" or base.lower().endswith(".dll"):
                with zf.open(member) as src, open(os.path.join(dest, base), "wb") as dst:
                    dst.write(src.read())
    exe = os.path.join(dest, "sd-server.exe")
    if not os.path.isfile(exe):
        print(f"ERROR: sd-server.exe not found in {name} zip", file=sys.stderr)
        return False
    print(f"  sd-server ({name}) vendored into {dest}")
    return True


def main() -> int:
    return 0 if all(_fetch_build(n, u) for n, u in BUILDS.items()) else 1


if __name__ == "__main__":
    sys.exit(main())
