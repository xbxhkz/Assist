"""Save/list/load/delete training datasets as JSONL under <DATA_DIR>/training/
datasets/. Path-safe (sanitized names, no traversal). Never raises."""
import json
import os
import re


def _default_dir():
    from src.constants import DATA_DIR
    return os.path.join(DATA_DIR, "training", "datasets")


def _safe_name(name):
    base = os.path.basename(str(name or "").strip())
    base = re.sub(r"\.jsonl$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-.")
    return base


class DatasetStore:
    def __init__(self, base_dir=None):
        self._dir = base_dir or _default_dir()

    def _path(self, name):
        safe = _safe_name(name)
        return safe, (os.path.join(self._dir, safe + ".jsonl") if safe else None)

    def save(self, name, rows) -> dict:
        try:
            safe, path = self._path(name)
            if not safe:
                return {"error": "invalid dataset name"}
            if not isinstance(rows, list) or not rows:
                return {"error": "rows must be a non-empty list"}
            os.makedirs(self._dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            return {"ok": True, "path": path, "name": safe}
        except Exception as e:  # noqa: BLE001
            return {"error": f"save failed: {e}"}

    def list(self) -> list:
        out = []
        try:
            if not os.path.isdir(self._dir):
                return out
            for fn in sorted(os.listdir(self._dir)):
                if not fn.lower().endswith(".jsonl"):
                    continue
                p = os.path.join(self._dir, fn)
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        n = sum(1 for ln in f if ln.strip())
                    out.append({"name": fn[:-6], "path": p, "rows": n, "size": os.path.getsize(p)})
                except Exception:
                    pass
        except Exception:
            pass
        return out

    def load(self, name) -> dict:
        try:
            safe, path = self._path(name)
            if not path or not os.path.isfile(path):
                return {"error": "dataset not found"}
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        rows.append(json.loads(ln))
                    except Exception:
                        pass
            return {"rows": rows, "name": safe, "path": path}
        except Exception as e:  # noqa: BLE001
            return {"error": f"load failed: {e}"}

    def delete(self, name) -> dict:
        try:
            safe, path = self._path(name)
            if path and os.path.isfile(path):
                os.remove(path)
                return {"ok": True}
            return {"error": "dataset not found"}
        except Exception as e:  # noqa: BLE001
            return {"error": f"delete failed: {e}"}


_store = None


def get_dataset_store():
    global _store
    if _store is None:
        _store = DatasetStore()
    return _store
