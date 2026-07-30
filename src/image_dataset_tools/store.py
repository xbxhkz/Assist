"""Save/list/load/delete image LoRA-training datasets under <DATA_DIR>/
training/image_datasets/<name>/, in the kohya-ss/diffusers convention
(image + same-basename .txt caption). Path-safe. Never raises."""
import json
import os
import re
import shutil


def _default_dir():
    from src.constants import DATA_DIR
    return os.path.join(DATA_DIR, "training", "image_datasets")


def _safe_name(name):
    base = os.path.basename(str(name or "").strip())
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-.")
    return base


class ImageDatasetStore:
    def __init__(self, base_dir=None):
        self._dir = base_dir or _default_dir()

    def _dataset_dir(self, name):
        safe = _safe_name(name)
        return safe, (os.path.join(self._dir, safe) if safe else None)

    def save(self, name, entries, trigger_word="") -> dict:
        safe, path = self._dataset_dir(name)
        if not safe:
            return {"error": "invalid dataset name"}
        if not isinstance(entries, list) or not entries:
            return {"error": "entries must be a non-empty list"}
        tmp = path + ".tmp"
        try:
            os.makedirs(self._dir, exist_ok=True)
            if os.path.isdir(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            saved = []
            for i, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    continue
                src = entry.get("path")
                caption = entry.get("caption")
                if not isinstance(src, str) or not os.path.isfile(src):
                    continue  # skip missing sources -- not fatal
                ext = os.path.splitext(src)[1].lower() or ".png"
                base_name = f"{i:04d}{ext}"
                shutil.copyfile(src, os.path.join(tmp, base_name))
                with open(os.path.join(tmp, f"{i:04d}.txt"), "w", encoding="utf-8") as f:
                    f.write(caption if isinstance(caption, str) else "")
                saved.append(base_name)
            meta = {"trigger_word": trigger_word if isinstance(trigger_word, str) else "",
                    "images": len(saved)}
            with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            os.replace(tmp, path)
            return {"ok": True, "path": path, "name": safe}
        except Exception as e:  # noqa: BLE001
            if os.path.isdir(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
            return {"error": f"save failed: {e}"}

    def list(self) -> list:
        out = []
        try:
            if not os.path.isdir(self._dir):
                return out
            for fn in sorted(os.listdir(self._dir)):
                p = os.path.join(self._dir, fn)
                if not os.path.isdir(p) or fn.endswith(".tmp"):
                    continue
                try:
                    n_images = sum(1 for f in os.listdir(p) if not f.endswith((".txt", ".json")))
                    size = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p))
                    out.append({"name": fn, "path": p, "images": n_images, "size": size})
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        return out

    def load(self, name) -> dict:
        safe, path = self._dataset_dir(name)
        if not path or not os.path.isdir(path):
            return {"error": "dataset not found"}
        try:
            meta = {"trigger_word": "", "images": 0}
            meta_path = os.path.join(path, "meta.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta.update(json.load(f))
                except Exception:  # noqa: BLE001
                    pass
            images = []
            for fn in sorted(os.listdir(path)):
                if fn.endswith((".txt", ".json")):
                    continue
                stem = os.path.splitext(fn)[0]
                cap_path = os.path.join(path, stem + ".txt")
                caption = ""
                if os.path.isfile(cap_path):
                    try:
                        with open(cap_path, "r", encoding="utf-8") as f:
                            caption = f.read()
                    except Exception:  # noqa: BLE001
                        pass
                images.append({"filename": fn, "caption": caption})
            return {"name": safe, "path": path, "trigger_word": meta.get("trigger_word", ""), "images": images}
        except Exception as e:  # noqa: BLE001
            return {"error": f"load failed: {e}"}

    def delete(self, name) -> dict:
        safe, path = self._dataset_dir(name)
        try:
            if path and os.path.isdir(path):
                shutil.rmtree(path)
                return {"ok": True}
            return {"error": "dataset not found"}
        except Exception as e:  # noqa: BLE001
            return {"error": f"delete failed: {e}"}


_store = None


def get_image_dataset_store():
    global _store
    if _store is None:
        _store = ImageDatasetStore()
    return _store
