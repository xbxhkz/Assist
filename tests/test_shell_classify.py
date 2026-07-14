from src.shell_exec.classify import classify_command as c


def test_powershell_reads():
    assert c("Get-ChildItem", "powershell") == "read"
    assert c("get-process", "powershell") == "read"           # case-insensitive
    assert c("Test-Path C:\\x", "powershell") == "read"
    assert c("Format-Table", "powershell") == "read"


def test_powershell_writes_and_unknown():
    assert c("Remove-Item x", "powershell") == "write"
    assert c("Set-Content x y", "powershell") == "write"
    assert c("frobnicate", "powershell") == "write"           # unknown → write


def test_cmd_reads_and_writes():
    assert c("dir", "cmd") == "read"
    assert c("type foo.txt", "cmd") == "read"
    assert c("del foo.txt", "cmd") == "write"
    assert c("set FOO=bar", "cmd") == "write"                 # not on allowlist


def test_compose_chars_force_write():
    assert c("Get-Content x | Remove-Item", "powershell") == "write"   # pipe
    assert c("echo x > f.txt", "cmd") == "write"                        # redirect
    assert c("dir && del x", "cmd") == "write"                          # chain
    assert c("Get-ChildItem; Remove-Item x", "powershell") == "write"   # semicolon
    assert c("dir & whoami", "cmd") == "write"                          # background/sep


def test_empty_is_write():
    assert c("", "powershell") == "write"
    assert c("   ", "cmd") == "write"
