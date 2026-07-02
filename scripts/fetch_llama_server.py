"""Vendor a prebuilt CPU llama-server into build_assets/llama/ at build time.

Downloads a pinned llama.cpp Windows CPU release zip and extracts
llama-server.exe (plus its DLLs) into build_assets/llama/. Pinned for
reproducibility; bump LLAMA_RELEASE deliberately after verifying a build.
"""
import io
import os
import sys
import urllib.request
import zipfile

# Pinned llama.cpp release asset (Windows x64 CPU build). Update deliberately.
LLAMA_RELEASE = os.getenv(
    "LLAMA_RELEASE_URL",
    "https://github.com/ggml-org/llama.cpp/releases/download/b4589/"
    "llama-b4589-bin-win-avx2-x64.zip",
)
ASSET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "build_assets", "llama")
)


def main() -> int:
    os.makedirs(ASSET_DIR, exist_ok=True)
    print(f"Downloading llama-server from {LLAMA_RELEASE} ...")
    with urllib.request.urlopen(LLAMA_RELEASE) as resp:  # noqa: S310
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            base = os.path.basename(member)
            if not base:
                continue
            # Take the server exe and every DLL (runtime deps) at any depth.
            if base == "llama-server.exe" or base.lower().endswith(".dll"):
                with zf.open(member) as src, open(os.path.join(ASSET_DIR, base), "wb") as dst:
                    dst.write(src.read())
    exe = os.path.join(ASSET_DIR, "llama-server.exe")
    if not os.path.isfile(exe):
        print("ERROR: llama-server.exe not found in release zip", file=sys.stderr)
        return 1
    print(f"llama-server vendored into {ASSET_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
