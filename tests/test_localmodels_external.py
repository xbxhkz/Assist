"""Linked (external) .gguf registry: add/remove/list/prune, no file deletion."""
import os

import pytest
from fastapi import HTTPException

import src.localmodels.external as ext


def _setup(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(ext, "MODELS_DIR", str(models))
    return models


def _gguf(dir_, name):
    p = dir_ / name
    p.write_bytes(b"gg")
    return str(p)


def test_add_returns_entry_and_persists(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    d = tmp_path / "ext"
    d.mkdir()
    path = _gguf(d, "m.gguf")
    entry = ext.add_external_model(path)
    assert entry["external"] is True
    assert entry["name"] == "m.gguf"
    assert entry["path"] == os.path.realpath(path)
    assert entry["size"] == 2
    assert any(e["path"] == os.path.realpath(path)
               for e in ext.list_external_models())


def test_add_rejects_non_gguf(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    d = tmp_path / "ext"
    d.mkdir()
    p = d / "note.txt"
    p.write_text("x")
    with pytest.raises(ValueError):
        ext.add_external_model(str(p))


def test_add_rejects_missing_file(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        ext.add_external_model(str(tmp_path / "nope.gguf"))


def test_add_skips_path_inside_models_dir(tmp_path, monkeypatch):
    models = _setup(tmp_path, monkeypatch)
    inside = _gguf(models, "d.gguf")
    entry = ext.add_external_model(inside)
    assert entry["external"] is False           # already a managed model
    assert ext.list_external_models() == []      # not registered as external


def test_add_dedupes(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    d = tmp_path / "ext"
    d.mkdir()
    path = _gguf(d, "m.gguf")
    ext.add_external_model(path)
    ext.add_external_model(path)
    assert len(ext.list_external_models()) == 1


def test_list_prunes_missing(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    d = tmp_path / "ext"
    d.mkdir()
    path = _gguf(d, "m.gguf")
    ext.add_external_model(path)
    os.remove(path)
    assert ext.list_external_models() == []
    assert ext.is_registered_external(path) is False   # pruned


def test_remove_unlinks_without_deleting(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    d = tmp_path / "ext"
    d.mkdir()
    path = _gguf(d, "m.gguf")
    ext.add_external_model(path)
    ext.remove_external_model(path)
    assert ext.is_registered_external(path) is False
    assert os.path.isfile(path)                          # file untouched


def test_is_registered_external(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    d = tmp_path / "ext"
    d.mkdir()
    path = _gguf(d, "m.gguf")
    assert ext.is_registered_external(path) is False
    ext.add_external_model(path)
    assert ext.is_registered_external(path) is True


def test_manager_list_merges_downloaded_and_linked(tmp_path, monkeypatch):
    import src.localmodels.manager as mgr_mod
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(ext, "MODELS_DIR", str(models))
    monkeypatch.setattr(mgr_mod, "MODELS_DIR", str(models))
    (models / "dl.gguf").write_bytes(b"gg")           # downloaded
    d = tmp_path / "ext"
    d.mkdir()
    ext.add_external_model(_gguf(d, "lk.gguf"))        # linked elsewhere
    by_name = {m["name"]: m for m in mgr_mod.LocalModelManager().list_models()}
    assert by_name["dl.gguf"]["external"] is False
    assert by_name["lk.gguf"]["external"] is True


def test_serve_validation_accepts_registered_external_rejects_arbitrary(tmp_path, monkeypatch):
    import routes.localmodels_routes as lr
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(ext, "MODELS_DIR", str(models))
    monkeypatch.setattr(lr, "MODELS_DIR", str(models))
    d = tmp_path / "ext"
    d.mkdir()
    linked = _gguf(d, "lk.gguf")
    ext.add_external_model(linked)
    assert lr._validate_model_path(linked) == os.path.realpath(linked)
    arbitrary = _gguf(d, "other.gguf")  # exists, .gguf, but not registered
    with pytest.raises(HTTPException):
        lr._validate_model_path(arbitrary)
