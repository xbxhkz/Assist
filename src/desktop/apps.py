"""Resolve an app/file/URL name to a launch target and launch it. OS calls are
injected so tests never touch the real Start Menu / registry / shell."""
import os
import re
import shutil
import subprocess

_URL_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


def resolve_app(name, *, start_menu=None, which=shutil.which, registry=None):
    raw = (name or "").strip()
    if not raw:
        return None
    if _URL_RE.match(raw):
        return {"name": raw, "target": raw, "kind": "url"}
    if os.path.exists(raw):
        return {"name": os.path.basename(raw) or raw, "target": raw, "kind": "path"}
    key = raw.lower()
    sm = start_menu if start_menu is not None else _default_start_menu()
    if key in sm:
        return {"name": raw, "target": sm[key], "kind": "shortcut"}
    reg = registry if registry is not None else _default_app_paths()
    if key in reg:
        return {"name": raw, "target": reg[key], "kind": "exe"}
    found = which(raw) or which(raw + ".exe")
    if found:
        return {"name": raw, "target": found, "kind": "exe"}
    return None


def launch(target, startfile=None, popen=None):
    startfile = startfile or getattr(os, "startfile", None)
    kind, tgt = target["kind"], target["target"]
    if kind in ("shortcut", "path", "url"):
        if startfile is None:
            raise RuntimeError("os.startfile unavailable on this platform")
        startfile(tgt)
        return
    # exe/uwp: detached, hidden console, never waited on.
    runner = popen or (lambda argv, **k: subprocess.Popen(
        argv, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    runner([tgt])


def _default_start_menu():
    """{app_name_lower: shortcut_path} from the system + user Start Menu."""
    out = {}
    roots = [os.path.join(os.environ.get("ProgramData", r"C:\ProgramData"),
                          r"Microsoft\Windows\Start Menu\Programs"),
             os.path.join(os.environ.get("APPDATA", ""),
                          r"Microsoft\Windows\Start Menu\Programs")]
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dp, _dns, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith(".lnk"):
                    out.setdefault(os.path.splitext(fn)[0].lower(),
                                   os.path.join(dp, fn))
    return out


def _default_app_paths():
    """{exe_stem_lower: full_path} from HKLM/HKCU App Paths."""
    out = {}
    try:
        import winreg
    except ImportError:
        return out
    sub = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
    for hive in (getattr(winreg, "HKEY_LOCAL_MACHINE", None),
                 getattr(winreg, "HKEY_CURRENT_USER", None)):
        if hive is None:
            continue
        try:
            with winreg.OpenKey(hive, sub) as k:
                for i in range(winreg.QueryInfoKey(k)[0]):
                    name = winreg.EnumKey(k, i)
                    try:
                        with winreg.OpenKey(k, name) as sk:
                            val = winreg.QueryValue(sk, None)
                        if val:
                            out[os.path.splitext(name)[0].lower()] = val
                    except OSError:
                        continue
        except OSError:
            continue
    return out
