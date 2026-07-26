"""Vendor prebuilt llama-server (CPU + Vulkan) into build_assets/llama/.

Downloads pinned llama.cpp Windows release zips and extracts llama-server.exe
(plus its DLLs) into build_assets/llama/cpu/ and build_assets/llama/vulkan/.
Pinned for reproducibility; bump LLAMA_TAG deliberately after verifying a
build. Keep it recent: an old build cannot load newer model architectures
(e.g. b4589 rejected Qwen3.6 'qwen35moe' with "unknown model architecture").
"""
import io
import os
import sys
import urllib.request
import zipfile

LLAMA_TAG = os.getenv("LLAMA_RELEASE_TAG", "b9867")
_BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}"
ZIPS = {
    "cpu": f"{_BASE}/llama-{LLAMA_TAG}-bin-win-cpu-x64.zip",
    "vulkan": f"{_BASE}/llama-{LLAMA_TAG}-bin-win-vulkan-x64.zip",
}
ASSET_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "build_assets", "llama")
)


def _fetch(kind: str, url: str) -> bool:
    dest = os.path.join(ASSET_ROOT, kind)
    exe = os.path.join(dest, "llama-server.exe")
    if os.path.isfile(exe):
        print(f"{kind}: already vendored, skipping")
        return True
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading {kind} llama-server from {url} ...")
    with urllib.request.urlopen(url) as resp:  # noqa: S310
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            base = os.path.basename(member)
            if not base:
                continue
            # Take the server exe and every DLL (runtime deps) at any depth.
            if base in ("llama-server.exe", "llama-quantize.exe") or base.lower().endswith(".dll"):
                with zf.open(member) as src, open(os.path.join(dest, base), "wb") as dst:
                    dst.write(src.read())
    if not os.path.isfile(exe):
        print(f"ERROR: llama-server.exe not found in {kind} zip", file=sys.stderr)
        return False
    print(f"{kind} llama-server vendored into {dest}")
    return True


def main() -> int:
    ok = all(_fetch(k, u) for k, u in ZIPS.items())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
