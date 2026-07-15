"""LoRA registry + atomic streaming download. sd-server resolves a
<lora:name:weight> prompt tag against the --lora-model-dir this exposes."""
import os
from contextlib import contextmanager

from src.constants import IMAGE_MODELS_DIR


def loras_dir() -> str:
    d = os.path.join(IMAGE_MODELS_DIR, "loras")
    os.makedirs(d, exist_ok=True)
    return d


def _safe_stem_file(name: str) -> str:
    """Return a safe `<stem>.safetensors` basename, or raise ValueError."""
    if not name or "/" in name or "\\" in name or ".." in name:
        raise ValueError("unsafe lora name")
    base = os.path.basename(name)
    if not base.lower().endswith(".safetensors"):
        base += ".safetensors"
    return base


def list_loras() -> list:
    d = loras_dir()
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.lower().endswith(".safetensors"):
            p = os.path.join(d, fn)
            out.append({"name": os.path.splitext(fn)[0], "filename": fn,
                        "size": os.path.getsize(p)})
    return out


def delete_lora(name: str) -> bool:
    p = os.path.join(loras_dir(), _safe_stem_file(name))
    if os.path.isfile(p):
        os.remove(p)
        return True
    return False


@contextmanager
def _default_http_stream(url, headers):
    import httpx
    with httpx.stream("GET", url, headers=headers or {}, timeout=None,
                      follow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        yield total, r.iter_bytes()


def download_to_loras(url, filename, *, headers=None, http_stream=None) -> dict:
    """Stream `url` into loras/<safe filename> atomically (.part -> rename).
    `http_stream` is an injectable context manager yielding (total, chunks)."""
    http_stream = http_stream or _default_http_stream
    fn = _safe_stem_file(filename)
    dest = os.path.join(loras_dir(), fn)
    part = dest + ".part"
    try:
        with http_stream(url, headers) as (_total, chunks):
            with open(part, "wb") as f:
                for chunk in chunks:
                    f.write(chunk)
        os.replace(part, dest)
    finally:
        if os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass
    return {"name": os.path.splitext(fn)[0], "filename": fn,
            "size": os.path.getsize(dest)}
