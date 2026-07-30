"""Ephemeral staging area for images being prepared into a dataset --
uploaded/gallery-sourced images live here between "add" and "save". A
working set is identified by an opaque id; each image gets its own
sanitized-basename subfolder holding the image bytes + a state.json with its
caption. Path-safe. Never raises."""
import json
import os
import re
import shutil
import uuid


def _default_dir():
    from src.constants import DATA_DIR
    return os.path.join(DATA_DIR, "training", "image_datasets", "_working")


def _safe_id(value):
    s = re.sub(r"[^A-Za-z0-9_-]+", "", str(value or ""))
    return s


def _ws_dir(working_set_id):
    safe = _safe_id(working_set_id)
    if not safe:
        return None
    return os.path.join(_default_dir(), safe)


def _state_path(ws_dir):
    return os.path.join(ws_dir, "state.json")


def _read_state(ws_dir):
    try:
        with open(_state_path(ws_dir), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {"images": []}
    except Exception:  # noqa: BLE001
        return {"images": []}


def _write_state(ws_dir, state):
    try:
        os.makedirs(ws_dir, exist_ok=True)
        tmp = _state_path(ws_dir) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, _state_path(ws_dir))
    except Exception:  # noqa: BLE001
        pass


def new_working_set() -> str:
    return uuid.uuid4().hex


def add_images(working_set_id, images) -> list:
    """`images`: [(filename, bytes)] or [(filename, bytes, caption)]. Returns
    the newly-added [{"id","caption"}]. Never raises; [] on any failure."""
    ws_dir = _ws_dir(working_set_id)
    if not ws_dir or not isinstance(images, list):
        return []
    try:
        os.makedirs(ws_dir, exist_ok=True)
        state = _read_state(ws_dir)
        state.setdefault("images", [])
        added = []
        for item in images:
            try:
                if not isinstance(item, tuple) or len(item) < 2:
                    continue
                filename, data = item[0], item[1]
                caption = item[2] if len(item) > 2 and isinstance(item[2], str) else ""
                if not isinstance(filename, str) or not isinstance(data, (bytes, bytearray)):
                    continue
                ext = os.path.splitext(filename)[1].lower() or ".png"
                image_id = uuid.uuid4().hex
                with open(os.path.join(ws_dir, image_id + ext), "wb") as f:
                    f.write(data)
                entry = {"id": image_id, "filename": image_id + ext, "caption": caption}
                state["images"].append(entry)
                added.append({"id": image_id, "caption": caption})
            except Exception:  # noqa: BLE001
                continue
        _write_state(ws_dir, state)
        return added
    except Exception:  # noqa: BLE001
        return []


def list_images(working_set_id) -> list:
    ws_dir = _ws_dir(working_set_id)
    if not ws_dir or not os.path.isdir(ws_dir):
        return []
    state = _read_state(ws_dir)
    return [{"id": img.get("id"), "caption": img.get("caption", "")}
            for img in state.get("images", []) if isinstance(img, dict)]


def get_image_path(working_set_id, image_id):
    ws_dir = _ws_dir(working_set_id)
    safe_id = _safe_id(image_id)
    if not ws_dir or not safe_id:
        return None
    state = _read_state(ws_dir)
    for img in state.get("images", []):
        if isinstance(img, dict) and img.get("id") == safe_id:
            path = os.path.join(ws_dir, img.get("filename", ""))
            return path if os.path.isfile(path) else None
    return None


def set_caption(working_set_id, image_id, caption) -> bool:
    ws_dir = _ws_dir(working_set_id)
    safe_id = _safe_id(image_id)
    if not ws_dir or not safe_id or not os.path.isdir(ws_dir):
        return False
    state = _read_state(ws_dir)
    found = False
    for img in state.get("images", []):
        if isinstance(img, dict) and img.get("id") == safe_id:
            img["caption"] = caption if isinstance(caption, str) else ""
            found = True
            break
    if found:
        _write_state(ws_dir, state)
    return found


def delete_working_set(working_set_id) -> bool:
    ws_dir = _ws_dir(working_set_id)
    if not ws_dir or not os.path.isdir(ws_dir):
        return False
    try:
        shutil.rmtree(ws_dir)
        return True
    except Exception:  # noqa: BLE001
        return False
