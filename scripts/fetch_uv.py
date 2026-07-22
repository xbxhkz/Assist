"""Vendor the `uv` binary (Astral) into build_assets/uv/uv.exe.

uv sets up the on-demand Python 3.11 CUDA training venv on the user's machine.
Downloads the Windows x64 zip from GitHub releases and extracts uv.exe. Uses the
`latest` asset by default; pin via UV_RELEASE_TAG for reproducible builds.
"""
import io
import os
import sys
import urllib.request
import zipfile

TAG = os.getenv("UV_RELEASE_TAG", "").strip()
ASSET = "uv-x86_64-pc-windows-msvc.zip"
URL = (f"https://github.com/astral-sh/uv/releases/download/{TAG}/{ASSET}" if TAG
       else f"https://github.com/astral-sh/uv/releases/latest/download/{ASSET}")
DEST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build_assets", "uv"))


def main() -> int:
    exe = os.path.join(DEST, "uv.exe")
    if os.path.isfile(exe):
        print("uv: already vendored, skipping")
        return 0
    os.makedirs(DEST, exist_ok=True)
    print(f"Downloading uv from {URL} ...")
    with urllib.request.urlopen(URL) as resp:  # noqa: S310
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if os.path.basename(member).lower() == "uv.exe":
                with zf.open(member) as src, open(exe, "wb") as dst:
                    dst.write(src.read())
    if not os.path.isfile(exe):
        print("ERROR: uv.exe not found in the release zip", file=sys.stderr)
        return 1
    print(f"uv vendored into {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
