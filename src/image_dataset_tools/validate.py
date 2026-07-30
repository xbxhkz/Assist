"""Validate an image dataset (all entries, non-raising): corrupt/unreadable
images, missing captions, exact duplicates, resolution floor. Pure (Pillow is
lazy-imported, mirroring routes/gallery/gallery_routes.py's convention)."""
import hashlib
import os


def _hash_file(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return None


def validate_image_set(entries, *, min_dimension=256):
    """entries: [{"id","path","caption"}, ...]. Never raises."""
    entries = entries if isinstance(entries, list) else []
    try:
        min_dim = max(1, int(min_dimension))
    except Exception:  # noqa: BLE001
        min_dim = 256

    from PIL import Image, UnidentifiedImageError

    errors = []
    seen_hashes = {}
    duplicates = 0
    missing_captions = 0
    valid = 0

    for i, entry in enumerate(entries):
        eid = entry.get("id", i) if isinstance(entry, dict) else i
        if not isinstance(entry, dict):
            errors.append({"id": eid, "message": "not a valid entry"})
            continue
        path = entry.get("path")
        caption = entry.get("caption")
        if not isinstance(path, str) or not os.path.isfile(path):
            errors.append({"id": eid, "message": "image file not found"})
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img2:
                w, h = img2.size
        except (UnidentifiedImageError, OSError, ValueError):
            errors.append({"id": eid, "message": "image is corrupt or unreadable"})
            continue
        except Exception:  # noqa: BLE001
            errors.append({"id": eid, "message": "image is corrupt or unreadable"})
            continue

        bad = []
        if min(w, h) < min_dim:
            bad.append(f"resolution {w}x{h} is below the {min_dim}px floor")
        if not isinstance(caption, str) or not caption.strip():
            missing_captions += 1
            bad.append("caption is missing or empty")
        digest = _hash_file(path)
        if digest:
            if digest in seen_hashes:
                duplicates += 1
                bad.append(f"duplicate of image '{seen_hashes[digest]}'")
            else:
                seen_hashes[digest] = eid
        if bad:
            errors.append({"id": eid, "message": "; ".join(bad)})
        else:
            valid += 1

    return {
        "total": len(entries),
        "valid": valid,
        "invalid": len(errors),
        "errors": errors,
        "stats": {"duplicates": duplicates, "missing_captions": missing_captions},
    }
