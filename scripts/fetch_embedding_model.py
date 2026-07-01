"""Vendor the default embedding model into build_assets/fastembed_cache.

Run at build time (before PyInstaller). Populates the cache using fastembed's
own layout by triggering a real embed, so the frozen app can load it offline
when FASTEMBED_CACHE_PATH points at the bundled copy.
"""
import os
import sys

ASSET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "build_assets", "fastembed_cache")
)
MODEL = os.getenv("FASTEMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def main() -> int:
    os.makedirs(ASSET_DIR, exist_ok=True)
    os.environ["FASTEMBED_CACHE_PATH"] = ASSET_DIR
    from fastembed import TextEmbedding

    print(f"Fetching embedding model {MODEL} into {ASSET_DIR} ...")
    emb = TextEmbedding(model_name=MODEL, cache_dir=ASSET_DIR)
    # Force the model files to download+materialize in the cache.
    list(emb.embed(["warmup"]))
    print("Embedding model cached.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
