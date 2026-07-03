"""Vendor the default embedding model into build_assets/fastembed_cache.

Run at build time (before PyInstaller). Populates the cache using fastembed's
own layout by triggering a real embed, so the frozen app can load it offline
when FASTEMBED_CACHE_PATH points at the bundled copy.

The Hugging Face cache stores snapshot files as symlinks into a `blobs/` dir.
PyInstaller's COLLECT tries to recreate those symlinks under `dist/`, which
fails on Windows without Developer Mode/admin (WinError 1314), and symlinks are
undesirable in a distributable anyway. So after fetching we de-symlink the
cache: every symlink is replaced with a real copy of its target.
"""
import os
import shutil
import sys

ASSET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "build_assets", "fastembed_cache")
)
MODEL = os.getenv("FASTEMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _desymlink(root: str) -> int:
    """Replace every symlinked file under `root` with a real copy of its target.

    Makes the bundled cache safe for PyInstaller COLLECT on Windows (no symlink
    privilege needed) and self-contained (no dangling links in the installer).
    """
    count = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                target = os.path.realpath(path)
                os.unlink(path)
                shutil.copy2(target, path)
                count += 1
    return count


def main() -> int:
    os.makedirs(ASSET_DIR, exist_ok=True)
    from fastembed import TextEmbedding

    print(f"Fetching embedding model {MODEL} into {ASSET_DIR} ...")
    emb = TextEmbedding(model_name=MODEL, cache_dir=ASSET_DIR)
    # Force the model files to download+materialize in the cache.
    list(emb.embed(["warmup"]))
    replaced = _desymlink(ASSET_DIR)
    print(f"Embedding model cached (de-symlinked {replaced} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
