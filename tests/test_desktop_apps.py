import src.desktop.apps as apps


def test_resolves_start_menu_shortcut():
    sm = {"notepad": r"C:\ProgramData\...\Notepad.lnk"}
    got = apps.resolve_app("Notepad", start_menu=sm, which=lambda n: None, registry={})
    assert got == {"name": "Notepad", "target": sm["notepad"], "kind": "shortcut"}


def test_resolves_registry_app_paths():
    reg = {"code": r"C:\Program Files\Microsoft VS Code\Code.exe"}
    got = apps.resolve_app("code", start_menu={}, which=lambda n: None, registry=reg)
    assert got["kind"] == "exe" and got["target"] == reg["code"]


def test_resolves_from_path():
    got = apps.resolve_app("python", start_menu={}, registry={},
                           which=lambda n: r"C:\Py\python.exe" if n == "python" else None)
    assert got["kind"] == "exe" and got["target"].endswith("python.exe")


def test_existing_path_opens_as_path(tmp_path):
    f = tmp_path / "report.pdf"; f.write_text("x")
    got = apps.resolve_app(str(f), start_menu={}, which=lambda n: None, registry={})
    assert got["kind"] == "path" and got["target"] == str(f)


def test_url_is_recognized():
    got = apps.resolve_app("https://example.com", start_menu={}, which=lambda n: None, registry={})
    assert got["kind"] == "url"


def test_unknown_returns_none():
    assert apps.resolve_app("nonesuchapp", start_menu={}, which=lambda n: None, registry={}) is None


def test_launch_shortcut_uses_startfile():
    calls = []
    apps.launch({"target": "x.lnk", "kind": "shortcut"}, startfile=calls.append)
    assert calls == ["x.lnk"]


def test_launch_exe_uses_popen():
    spawned = []
    apps.launch({"target": "x.exe", "kind": "exe"},
                startfile=lambda p: (_ for _ in ()).throw(AssertionError("should popen")),
                popen=lambda argv, **k: spawned.append(argv))
    assert spawned == [["x.exe"]]
