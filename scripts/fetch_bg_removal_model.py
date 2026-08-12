"""Fetch the U2Net ONNX background-removal model into build_assets/ so it
can be bundled into the frozen build (Assist.spec collects the whole
build_assets/bg_removal/ directory, mirroring build_assets/yolo/). Mirrors
scripts/fetch_sd_server.py's raw-download pattern -- a single file here, no
zip extraction needed.

NOTE: MODEL_URL below is from training-knowledge memory of where the rembg
project (danielgatis/rembg) hosts its pre-exported ONNX models as GitHub
Release assets -- NOT freshly verified. Confirm with `curl -sIL <url>`
before relying on this in a real release build. No checksum verification
exists here, matching every sibling fetch_*.py script in this repo (none of
them verify a hash either -- only existence-after-write is checked).
"""
import os
import sys
import urllib.request

ASSET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "build_assets", "bg_removal")
)
MODEL_URL = os.getenv(
    "BG_REMOVAL_MODEL_URL",
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
)
MODEL_PATH = os.path.join(ASSET_DIR, "u2net.onnx")


def main() -> int:
    if os.path.isfile(MODEL_PATH):
        print(f"Background-removal model already present at {MODEL_PATH}, skipping.")
        return 0

    os.makedirs(ASSET_DIR, exist_ok=True)
    print(f"Downloading background-removal model from {MODEL_URL} ...")
    try:
        with urllib.request.urlopen(MODEL_URL) as resp:  # noqa: S310
            data = resp.read()
    except Exception as e:
        print(f"ERROR: failed to download model: {e}", file=sys.stderr)
        return 1

    with open(MODEL_PATH, "wb") as f:
        f.write(data)

    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH} after download", file=sys.stderr)
        return 1

    print(f"Background-removal model saved to {MODEL_PATH} ({len(data)} bytes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
