import os
import src.desktop.filesearch as fs


def test_uses_index_when_available(tmp_path):
    a = tmp_path / "report.txt"; a.write_text("x")
    got = fs.search("report", roots=[str(tmp_path)],
                    searcher=lambda q, ext, n: [str(a)],
                    is_sensitive=lambda p: False)
    assert [h["path"] for h in got] == [str(a)]
    assert got[0]["size"] == 1


def test_falls_back_to_walk_when_searcher_none(tmp_path):
    (tmp_path / "notes.md").write_text("y")
    (tmp_path / "other.txt").write_text("z")
    got = fs.search("notes", roots=[str(tmp_path)], searcher=None,
                    is_sensitive=lambda p: False)
    assert [os.path.basename(h["path"]) for h in got] == ["notes.md"]


def test_falls_back_when_searcher_raises(tmp_path):
    (tmp_path / "keep.log").write_text("y")
    def boom(q, ext, n): raise OSError("index off")
    got = fs.search("keep", roots=[str(tmp_path)], searcher=boom,
                    is_sensitive=lambda p: False)
    assert len(got) == 1 and got[0]["path"].endswith("keep.log")


def test_ext_filter_in_walk(tmp_path):
    (tmp_path / "a.py").write_text("1")
    (tmp_path / "a.txt").write_text("1")
    got = fs.search("a", roots=[str(tmp_path)], ext="py", searcher=None,
                    is_sensitive=lambda p: False)
    assert [os.path.basename(h["path"]) for h in got] == ["a.py"]


def test_sensitive_paths_filtered(tmp_path):
    (tmp_path / "id_rsa").write_text("secret")
    got = fs.search("id_rsa", roots=[str(tmp_path)], searcher=None,
                    is_sensitive=lambda p: p.endswith("id_rsa"))
    assert got == []


def test_max_results_caps_walk(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("x")
    got = fs.search("f", roots=[str(tmp_path)], max_results=3, searcher=None,
                    is_sensitive=lambda p: False)
    assert len(got) == 3
