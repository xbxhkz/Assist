"""DownloadManager tests: synchronous spawn + fake stream, no network."""
import os
from contextlib import contextmanager

import pytest

import src.localmodels.downloader as dl


def make_stream(total, chunks, on_chunk=None):
    @contextmanager
    def _stream(url, headers):
        def gen():
            for c in chunks:
                if on_chunk:
                    on_chunk()
                yield c
        yield total, gen()
    return _stream


def make_manager(tmp_path, stream, **kw):
    return dl.DownloadManager(
        http_stream=stream,
        spawn=lambda fn: fn(),          # synchronous → deterministic
        dest_dir=str(tmp_path),
        headers_provider=lambda: {},
        **kw,
    )


def test_safe_filename():
    assert dl._safe_filename("m.gguf") == "m.gguf"
    assert dl._safe_filename("../m.gguf") is None
    assert dl._safe_filename("a/b.gguf") is None
    assert dl._safe_filename("a\\b.gguf") is None
    assert dl._safe_filename("m.txt") is None
    assert dl._safe_filename("") is None


def test_download_completes_and_renames(tmp_path):
    mgr = make_manager(tmp_path, make_stream(4, [b"aa", b"bb"]))
    mgr.start("https://huggingface.co/x/resolve/main/m.gguf", "m.gguf")
    st = mgr.status()
    assert st["error"] is None
    assert (tmp_path / "m.gguf").read_bytes() == b"aabb"
    assert not (tmp_path / "m.gguf.part").exists()
    assert st["bytes"] == 4 and st["total"] == 4 and st["pct"] == 100.0
    assert st["downloading"] is False


def test_download_rejects_bad_filename(tmp_path):
    mgr = make_manager(tmp_path, make_stream(1, [b"x"]))
    with pytest.raises(ValueError):
        mgr.start("https://huggingface.co/x/resolve/main/e.gguf", "../evil.gguf")


def test_second_download_rejected_while_active(tmp_path):
    mgr = make_manager(tmp_path, make_stream(1, [b"x"]))
    mgr._active = True  # simulate an in-flight download
    with pytest.raises(RuntimeError):
        mgr.start("https://huggingface.co/x/resolve/main/m.gguf", "m.gguf")


def test_cancel_removes_partial(tmp_path):
    # Cancel fires on the first chunk → transfer aborts, .part cleaned up.
    def on_chunk():
        mgr.cancel()
    mgr = make_manager(tmp_path, make_stream(10, [b"aa", b"bb", b"cc"], on_chunk=on_chunk))
    mgr.start("https://huggingface.co/x/resolve/main/m.gguf", "m.gguf")
    assert not (tmp_path / "m.gguf").exists()
    assert not (tmp_path / "m.gguf.part").exists()


def test_status_idle():
    mgr = dl.DownloadManager(spawn=lambda fn: fn())
    assert mgr.status() == {"downloading": False, "filename": None, "bytes": 0,
                            "total": None, "pct": None, "error": None}


def test_start_sweeps_stale_partials(tmp_path):
    (tmp_path / "old-model.gguf.part").write_bytes(b"stale")
    mgr = make_manager(tmp_path, make_stream(2, [b"aa"]))
    mgr.start("https://huggingface.co/x/resolve/main/new.gguf", "new.gguf")
    assert not (tmp_path / "old-model.gguf.part").exists()  # stale swept
    assert (tmp_path / "new.gguf").read_bytes() == b"aa"     # new completed
