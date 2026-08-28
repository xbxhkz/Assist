# Unslothed Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Unslothed as a self-contained Windows installer and two Docker images, under the product name "Unslothed", without disturbing the upstream merge seam.

**Architecture:** Everything is additive — new files under `packaging/` plus four narrow edits to branding config and locale strings. No upstream Python source is edited. The installer is PyInstaller + Inno Setup (Assist's proven pipeline); Docker is one multi-stage Dockerfile producing `:cpu` and `:cuda`.

**Tech Stack:** PyInstaller, Inno Setup 6 (already installed at `C:\Program Files (x86)\Inno Setup 6\ISCC.exe`), Docker multi-stage builds, Node 22 for the frontend build stage.

**Spec:** `docs/superpowers/specs/2026-08-28-unslothed-packaging-design.md`

**Target repo:** `C:\Users\Admin\unsloth` — **not** the repo this plan lives in. Branch from `assist-vision-tools`, which now carries both prior sub-projects.

## Global Constraints

- **The upstream seam must not grow.** `core/inference/tools.py` stays at +8/−0 in 3 hunks; `routes/inference.py` stays at exactly one re-add hunk with `if tools_on and tools:` intact. If a task seems to need a change there, stop and report.
- **No Python source outside `packaging/` is edited** except the four branding surfaces named in Task 2.
- **The Tauri `identifier` (`ai.unsloth.studio`) must NOT change.** It is the OS-level app identity; changing it orphans settings on upgrade.
- **"Unslothed" is untranslated in all 12 locales** — a proper noun.
- **Do not run the full backend suite** — upstream tests fabricate non-sparse GGUF fixtures up to 40 GB and have already caused a disk-full incident. Scope every run with `-k`.
- Run pytest with `--import-mode=importlib` from `studio/backend`.
- Versions to pin against, measured on this machine: **torch 2.10.0+cu130**, **torchvision 0.25.0+cu130**.
- Tool counts for assertions, verified against the installed package: **17 total = 7 upstream + 10 ours** (5 vision, 5 code).
- SPDX header on every new Python file:
  ```
  # SPDX-License-Identifier: AGPL-3.0-only
  # Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
  ```
- Stage specific files; never `git add -A`.
- End every commit message with: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

---

## File Structure

| File | Responsibility |
|---|---|
| `studio/backend/requirements/base.txt` | pin `torchvision` (Task 1) |
| `studio/frontend/index.html` | page title (Task 2) |
| `studio/src-tauri/tauri.conf.json` | `productName`, window title (Task 2) |
| `studio/frontend/src/i18n/locales/*.ts` | 12 locale files, message values only (Task 2) |
| `packaging/check_branding.py` | asserts what changed and what did not (Task 2) |
| `packaging/Unslothed.spec` | PyInstaller spec (Task 3) |
| `packaging/Unslothed.iss` | Inno Setup script (Task 4) |
| `packaging/build-installer.ps1` | drives frontend → PyInstaller → ISCC (Task 4) |
| `packaging/smoke_installed.py` | runs the INSTALLED exe and asserts tools register (Task 5) |
| `packaging/Dockerfile` | multi-stage, both tags (Task 6) |
| `packaging/docker-entrypoint.sh` | startup, data-volume init (Task 6) |
| `packaging/build-docker.ps1` | builds both tags (Task 6) |
| `packaging/verify_docker.ps1` | runs both images and asserts (Task 7) |

---

## Task 1: Pin torchvision

**Files:**
- Modify: `studio/backend/requirements/base.txt`
- Test: `studio/backend/tests/test_requirements_pinned.py`

**Interfaces:**
- Produces: nothing importable. This is a defect fix that Task 6 depends on.

**Why first:** `torchvision` is the only unpinned entry across all eight requirements files, and sub-project 1 added it. Assist hit this exact shape once already — a bare `mcp` resolved to 2.0.0 on a fresh install and killed all four built-in MCP servers. In a Docker image rebuilt months from now, a bare `torchvision` can resolve to a version incompatible with the pinned torch, and the failure presents as a mysterious CUDA error rather than a version mismatch.

- [ ] **Step 1: Write the failing test**

```python
# studio/backend/tests/test_requirements_pinned.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every runtime dependency must be pinned.

A bare package name resolves to whatever is newest at install time. This
project has already been bitten once: a bare ``mcp`` in Assist's requirements
resolved to 2.0.0 on a fresh install and broke all four built-in MCP servers.
The failure did not look like a version problem, which is what made it
expensive.
"""
import pathlib
import re

REQ_DIR = pathlib.Path(__file__).resolve().parents[1] / "requirements"

# A bare name: letters/digits/._- and nothing else. No comparator, no extras,
# no environment marker, no URL.
_BARE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*$")


def _entries(path):
    for raw in path.read_text(encoding = "utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        yield line


def test_no_requirement_is_left_unpinned():
    unpinned = []
    for path in sorted(REQ_DIR.glob("*.txt")):
        for entry in _entries(path):
            if _BARE.match(entry):
                unpinned.append(f"{path.name}: {entry}")
    assert not unpinned, (
        "unpinned requirements resolve to whatever is newest at install time:\n  "
        + "\n  ".join(unpinned)
    )
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_requirements_pinned.py --import-mode=importlib -q`
Expected: FAIL, naming `base.txt: torchvision`

- [ ] **Step 3: Pin it**

In `studio/backend/requirements/base.txt`, replace the bare `torchvision` line with:

```
# Pinned to the torchvision that pairs with the pinned torch (2.10.0). A bare
# name here resolves to whatever is newest at install time, which in a Docker
# image rebuilt months later can pull a torchvision incompatible with torch --
# surfacing as a CUDA error rather than a version mismatch.
torchvision==0.25.0
```

- [ ] **Step 4: Run it and watch it pass**

Run: `python -m pytest tests/test_requirements_pinned.py --import-mode=importlib -q`
Expected: PASS

- [ ] **Step 5: Confirm nothing else broke**

Run: `python -m pytest tests/ -k "assist_code or assist_vision" --import-mode=importlib -q --ignore=tests/test_mcp_server.py --ignore=tests/test_rag_ocr_fallback.py`
Expected: the same counts as before this task — no new failures.

- [ ] **Step 6: Commit**

```bash
git add studio/backend/requirements/base.txt studio/backend/tests/test_requirements_pinned.py
git commit -m "fix(deps): pin torchvision, the only unpinned requirement"
```

---

## Task 2: Branding

**Files:**
- Modify: `studio/frontend/index.html`
- Modify: `studio/src-tauri/tauri.conf.json`
- Modify: `studio/frontend/src/i18n/locales/*.ts` (12 files)
- Create: `packaging/check_branding.py`

**Interfaces:**
- Produces: `packaging/check_branding.py`, runnable standalone, exit 0 on pass.

**The measurement that shapes this task:** the frontend has **3089** occurrences of `unsloth`, of which **2351 sit outside the locale files** — import paths, API routes, CSS classes, the Tauri identifier. A blanket replace breaks the application. Change only user-visible strings.

- [ ] **Step 1: Write the verification script**

```python
# packaging/check_branding.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Assert the branding changed where it should and nowhere else.

Run from the repo root:  python packaging/check_branding.py

The risk this guards is not "did we rename enough" but "did we rename too
much". Of 3089 `unsloth` occurrences in the frontend, 2351 are import paths,
API routes and identifiers. Renaming those breaks the app, and the breakage
would not be obvious from a diff.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "studio" / "frontend"
LOCALES = FRONTEND / "src" / "i18n" / "locales"

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)


# 1. The page title.
index_html = (FRONTEND / "index.html").read_text(encoding = "utf-8")
check("<title>Unslothed</title>" in index_html,
      "index.html: <title> is not Unslothed")

# 2. Tauri product name and window title -- but NOT the identifier.
conf = json.loads((ROOT / "studio" / "src-tauri" / "tauri.conf.json").read_text(encoding = "utf-8"))
check(conf.get("productName") == "Unslothed",
      f"tauri.conf.json: productName is {conf.get('productName')!r}, expected 'Unslothed'")
titles = [w.get("title") for w in (conf.get("app") or {}).get("windows") or []]
check(all(t == "Unslothed" for t in titles),
      f"tauri.conf.json: window titles are {titles!r}, expected all 'Unslothed'")

# The identifier is the OS-level application identity. Changing it makes
# Windows and macOS treat an upgrade as a different app, orphaning settings.
check(conf.get("identifier") == "ai.unsloth.studio",
      f"tauri.conf.json: identifier is {conf.get('identifier')!r} -- it MUST stay "
      "'ai.unsloth.studio'; changing it orphans user settings on upgrade")

# 3. Nothing that is not a user-visible string may have changed. If any import
#    path, API route or CSS class picked up the new name, the app is broken.
DANGEROUS = [
    (re.compile(r"from\s+['\"][^'\"]*unslothed[^'\"]*['\"]"), "an import path"),
    (re.compile(r"['\"]/api/[^'\"]*unslothed[^'\"]*['\"]"), "an API route"),
    (re.compile(r"ai\.unslothed\."), "the Tauri identifier"),
]
for path in FRONTEND.joinpath("src").rglob("*.ts*"):
    text = path.read_text(encoding = "utf-8", errors = "replace")
    for pattern, what in DANGEROUS:
        if pattern.search(text):
            failures.append(f"{path.relative_to(ROOT)}: {what} was renamed -- this breaks the app")

# 4. The AGPL attribution names upstream and must survive.
en = (LOCALES / "en.ts").read_text(encoding = "utf-8")
check("Unsloth AI Inc." in en,
      "en.ts: the AGPL attribution to 'Unsloth AI Inc.' was removed -- that is "
      "upstream's legal notice, not product branding")

# 5. The brand is untranslated: every locale carries the same literal.
missing = [p.name for p in sorted(LOCALES.glob("*.ts"))
           if "Unslothed" not in p.read_text(encoding = "utf-8", errors = "replace")]
check(not missing, f"locales missing the 'Unslothed' literal: {missing}")

if failures:
    print("BRANDING CHECK FAILED")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("branding check passed")
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python packaging/check_branding.py`
Expected: exit 1, reporting the title, productName and window title are not yet `Unslothed`.

- [ ] **Step 3: Change the four surfaces**

`studio/frontend/index.html` — `<title>Unsloth</title>` becomes `<title>Unslothed</title>`.

`studio/src-tauri/tauri.conf.json` — set `productName` to `"Unslothed"` and every `app.windows[].title` to `"Unslothed"`. **Leave `identifier` exactly as it is.**

For the 12 locale files, change user-facing message **values** only. Never touch a key. Never touch the AGPL attribution on line 2. Work from `en.ts` first and mirror the same literal into the other 11 — the brand is untranslated.

Examples of what changes (values seen in `en.ts`):

| Before | After |
|---|---|
| `Unsloth home` | `Unslothed home` |
| `Unsloth preferences.` | `Unslothed preferences.` |
| `Unsloth Desktop App keeps signing in automatically.` | `Unslothed Desktop App keeps signing in automatically.` |
| `Unsloth at login` | `Unslothed at login` |

And what does **not**:

| Leave alone | Why |
|---|---|
| `Unsloth AI Inc. team. All rights reserved.` | upstream's AGPL attribution |
| `ai.unsloth.studio` | OS-level app identity |
| model-catalogue names, `alpaca_unsloth.json` | these name upstream, not this product |

- [ ] **Step 4: Run the check and the i18n parity check**

```bash
python packaging/check_branding.py
cd studio/frontend && npm run i18n:check
```
Expected: both pass. Parity matters because 12 files were edited by hand.

- [ ] **Step 5: Negative control — prove the check catches over-renaming**

Temporarily add a fake renamed import to any frontend `.ts` file:

```ts
import { nothing } from "./unslothed-does-not-exist";
```

Run `python packaging/check_branding.py`. It must FAIL with "an import path was renamed". Remove the line and confirm it passes again.

If it does not fail, the check is not testing what it claims — diagnose and fix it before continuing. Six negative controls in this project's history turned out inert.

- [ ] **Step 6: Commit**

```bash
git add studio/frontend/index.html studio/src-tauri/tauri.conf.json \
        studio/frontend/src/i18n/locales packaging/check_branding.py
git commit -m "feat(branding): Unslothed as the product name, in the four places users see it"
```

---

## Task 3: PyInstaller spec

**Files:**
- Create: `packaging/Unslothed.spec`

**Interfaces:**
- Produces: a PyInstaller spec that builds a portable app folder at `packaging/dist/Unslothed/`.

**The three hard parts, and why each is handled the way it is:**

**torch cannot be analysed.** It loads extensions dynamically and ships CUDA DLLs static analysis never sees. Exclude it from `Analysis` and copy the built tree in wholesale, with the `nvidia/` CUDA tree.

**Lazily imported modules are invisible.** `assist_vision` and `assist_code` import heavy dependencies *inside function bodies* deliberately — a module-scope `import torch` measured a ~2.5s ASGI event-loop stall in the source project. Static analysis cannot see them, so each needs a `hiddenimports` entry. Measured lazy imports: `cv2`, `onnxruntime`, `numpy`, `torch`, `torchvision.models.detection`, `torchvision.transforms.functional`, `ultralytics`, `insightface.app`, `insightface.model_zoo`.

**Language servers stay out.** They install on first use via npm and `dotnet tool`. Freezing them means shipping Node and the .NET SDK to duplicate a path that already works.

- [ ] **Step 1: Write the spec file**

```python
# packaging/Unslothed.spec
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# PyInstaller spec for the self-contained Unslothed Windows build.
#
# Run via packaging/build-installer.ps1, not directly -- the frontend must be
# built first or the app ships with no UI.

import os
import site
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).resolve().parents[0].parent
BACKEND = ROOT / "studio" / "backend"
SITE = Path(site.getsitepackages()[-1])

# --- data ------------------------------------------------------------------
# The frontend dist is the UI. Without it the app starts and serves nothing,
# which looks like a backend fault and is not.
datas = [
    (str(ROOT / "studio" / "frontend" / "dist"), "studio/frontend/dist"),
    (str(BACKEND / "assets"), "studio/backend/assets"),
    (str(BACKEND / "requirements"), "studio/backend/requirements"),
]
for optional in ("vendor", "plugins"):
    src = BACKEND / optional
    if src.is_dir():
        datas.append((str(src), f"studio/backend/{optional}"))

# torch and its CUDA payload are copied wholesale rather than analysed. Torch
# loads extensions dynamically and ships CUDA DLLs that static analysis never
# sees; letting PyInstaller try produces a build that imports and then fails at
# the first real call.
for heavy in ("torch", "torchvision", "nvidia", "functorch", "torchgen"):
    src = SITE / heavy
    if src.is_dir():
        datas.append((str(src), heavy))

# --- hidden imports --------------------------------------------------------
# Every entry below is imported INSIDE a function body somewhere in
# assist_vision or assist_code, deliberately, to keep startup fast. That is
# exactly what PyInstaller's static analysis cannot see. Without these the
# frozen build registers the tool and then raises ModuleNotFoundError on the
# first call -- a failure that looks like a tool bug, not a packaging bug.
hiddenimports = [
    # assist_vision
    "cv2",
    "numpy",
    "onnxruntime",
    "torch",
    "torchvision",
    "torchvision.models.detection",
    "torchvision.transforms.functional",
    "ultralytics",
    "insightface",
    "insightface.app",
    "insightface.model_zoo",
    # assist_code needs no third-party imports -- it is stdlib only, and its
    # language servers are external processes installed on first use.
]
hiddenimports += collect_submodules("studio.backend.core.inference.assist_vision")
hiddenimports += collect_submodules("studio.backend.core.inference.assist_code")

a = Analysis(
    [str(BACKEND / "app.py")],
    pathex = [str(ROOT), str(BACKEND)],
    binaries = [],
    datas = datas,
    hiddenimports = hiddenimports,
    hookspath = [],
    runtime_hooks = [],
    # Excluded because they are copied in via `datas` above. Leaving them in
    # Analysis doubles the build time and produces a broken CUDA payload.
    excludes = ["torch", "torchvision", "nvidia", "functorch", "torchgen"],
    noarchive = False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries = True,
    name = "Unslothed",
    debug = False,
    strip = False,
    upx = False,          # UPX corrupts CUDA DLLs; never enable it here.
    console = True,       # the backend logs to stdout; a windowed build hides startup failures
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip = False,
    upx = False,
    name = "Unslothed",
)
```

- [ ] **Step 2: Verify the entry point exists**

Run: `ls studio/backend/app.py`

If it does not exist, find the real ASGI entry point (`grep -rn "app = FastAPI" studio/backend/*.py | head -3`) and correct the `Analysis` first argument. Report what you found and what you changed.

- [ ] **Step 3: Commit the spec**

```bash
git add packaging/Unslothed.spec
git commit -m "build(installer): PyInstaller spec, with torch copied rather than analysed"
```

The spec is not run in this task — Task 4 supplies the driver that runs it in the right order.

---

## Task 4: Inno Setup script and build driver

**Files:**
- Create: `packaging/Unslothed.iss`
- Create: `packaging/build-installer.ps1`

**Interfaces:**
- Consumes: `packaging/Unslothed.spec` (Task 3)
- Produces: `packaging/Output/Unslothed-Setup.exe`

Per-user install, no elevation. Assist's equivalent installs to `{autopf}` (Program Files) and requires admin; this one deliberately does not.

- [ ] **Step 1: Write the Inno Setup script**

```
; packaging/Unslothed.iss
; SPDX-License-Identifier: AGPL-3.0-only
; Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
;
; Per-user install, no elevation. Studio keeps its data in %USERPROFILE%\.unsloth\studio
; and this installer must never write there -- installing must not disturb an
; existing setup's models, chats or auth.

#define MyAppName "Unslothed"
#define MyAppExeName "Unslothed.exe"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={localappdata}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputBaseFilename=Unslothed-Setup
OutputDir=Output
Compression=lzma2/max
SolidCompression=yes
; lowest = install for this user only, no UAC prompt. Matches DefaultDirName.
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
UninstallDisplayName={#MyAppName}

[Files]
Source: "dist\Unslothed\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{userdesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
```

- [ ] **Step 2: Write the build driver**

```powershell
# packaging/build-installer.ps1
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Builds the self-contained Unslothed Windows installer.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build-installer.ps1
#
# Stages: frontend -> PyInstaller -> Inno Setup. The frontend must come first;
# a build without studio/frontend/dist produces an app that starts and serves
# no UI, which reads as a backend fault and is not.

$ErrorActionPreference = "Stop"
function Write-Step($m) { Write-Host ""; Write-Host ("==> " + $m) -ForegroundColor Cyan }
function Fail($m) { Write-Host ""; Write-Host ("ERROR: " + $m) -ForegroundColor Red; exit 1 }

$Root = Split-Path -Parent $PSScriptRoot
$Packaging = Join-Path $Root "packaging"

Write-Step "Building the frontend"
Push-Location (Join-Path $Root "studio\frontend")
if (-not (Test-Path "node_modules")) { npm ci }
npm run build
Pop-Location
$dist = Join-Path $Root "studio\frontend\dist\index.html"
if (-not (Test-Path $dist)) { Fail "frontend build produced no dist/index.html" }

Write-Step "Resolving version"
$Version = (git -C $Root describe --tags --always 2>$null)
if (-not $Version) { $Version = "0.0.0" }
$Version = $Version -replace '[^0-9A-Za-z.\-]', ''
Write-Host "  version: $Version"

Write-Step "Running PyInstaller"
Push-Location $Packaging
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
pyinstaller --noconfirm --clean "Unslothed.spec"
Pop-Location
$appExe = Join-Path $Packaging "dist\Unslothed\Unslothed.exe"
if (-not (Test-Path $appExe)) { Fail "PyInstaller produced no Unslothed.exe" }

Write-Step "Locating Inno Setup"
$iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) { $iscc = "$env:ProgramFiles\Inno Setup 6\ISCC.exe" }
if (-not (Test-Path $iscc)) { Fail "ISCC.exe not found -- install Inno Setup 6" }

Write-Step "Compiling the installer"
Push-Location $Packaging
& $iscc "/DMyAppVersion=$Version" "Unslothed.iss"
if ($LASTEXITCODE -ne 0) { Fail "ISCC failed with exit code $LASTEXITCODE" }
Pop-Location

$setup = Join-Path $Packaging "Output\Unslothed-Setup.exe"
if (-not (Test-Path $setup)) { Fail "no installer at $setup" }
$sizeGb = [math]::Round((Get-Item $setup).Length / 1GB, 2)
Write-Step "Done: $setup ($sizeGb GB)"
```

- [ ] **Step 3: Install PyInstaller into the build environment**

```bash
python -m pip install pyinstaller
```

This belongs to the build environment, not the shipped one. Do **not** add it to `studio/backend/requirements/`.

- [ ] **Step 4: Run the build**

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build-installer.ps1
```

Expect this to take a long time and to be the riskiest step in the plan — a 2.61 GB torch tree with dynamically loaded CUDA DLLs is where PyInstaller builds usually go wrong. If it fails, report the actual error rather than guessing; common causes are a missing hidden import (add it to the spec) and a torch tree that was analysed rather than copied (check `excludes`).

- [ ] **Step 5: Commit**

```bash
git add packaging/Unslothed.iss packaging/build-installer.ps1
git commit -m "build(installer): Inno Setup script and build driver, per-user install"
```

Add `packaging/dist/`, `packaging/build/` and `packaging/Output/` to `.gitignore` in the same commit — build artefacts must not be committed.

---

## Task 5: Installer smoke test

**Files:**
- Create: `packaging/smoke_installed.py`

**Interfaces:**
- Consumes: an installed `Unslothed.exe`
- Produces: exit 0 on pass

**Why this test and not a lighter one:** a frozen build can register a tool and then raise `ModuleNotFoundError` on its first call, because the tool's dependencies are imported inside the function. Checking that the process starts proves nothing about that. This project has already shipped two frozen-build failures of exactly this shape — a `subprocess.run` hang from a missing `stdin=DEVNULL`, and an undeclared `psutil`.

- [ ] **Step 1: Write the smoke test**

```python
# packaging/smoke_installed.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise the INSTALLED build, not the source tree.

    python packaging/smoke_installed.py "%LOCALAPPDATA%\\Unslothed\\Unslothed.exe"

Asserts the backend starts, all 17 tools register (7 upstream + 10 ours), and
a lazily imported module actually loads. That last assertion is the point: a
frozen build can register a tool whose dependencies are imported inside its
function body and only fail on the first real call.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

EXPECTED_TOTAL = 17
OURS = {
    "remove_background", "detect_shapes", "webcam_look",
    "edit_image_prompt", "face_swap",
    "code_diagnostics", "code_definition", "code_references",
    "code_hover", "code_symbols",
}
PORT = int(os.environ.get("UNSLOTHED_SMOKE_PORT", "7860"))
BOOT_BUDGET = 180.0


def wait_for_health(url, budget):
    deadline = time.monotonic() + budget
    last = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout = 5) as r:
                if r.status == 200:
                    return True
        except Exception as e:  # noqa: BLE001 - any failure just means "not yet"
            last = e
        time.sleep(2)
    print(f"backend never became healthy within {budget:.0f}s (last: {last})")
    return False


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    exe = sys.argv[1]
    if not os.path.isfile(exe):
        print(f"no such executable: {exe}")
        return 2

    env = dict(os.environ, UNSLOTH_STUDIO_PORT = str(PORT))
    proc = subprocess.Popen([exe], env = env,
                            stdin = subprocess.DEVNULL,
                            stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    try:
        if not wait_for_health(f"http://127.0.0.1:{PORT}/health", BOOT_BUDGET):
            out = proc.stdout.read(4000).decode("utf-8", "replace") if proc.stdout else ""
            print("--- backend output ---")
            print(out)
            return 1

        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/tools", timeout = 15) as r:
            names = {t["function"]["name"] for t in json.loads(r.read())}

        print(f"tools registered: {len(names)}")
        missing = OURS - names
        if missing:
            print(f"FAIL: our tools missing from the frozen build: {sorted(missing)}")
            return 1
        if len(names) != EXPECTED_TOTAL:
            print(f"FAIL: expected {EXPECTED_TOTAL} tools, got {len(names)}")
            return 1

        # The assertion that matters. code_diagnostics on a nonexistent path
        # must return the confinement/not-found message -- reaching that means
        # the module and its lazy imports genuinely loaded. A frozen build with
        # a missing hidden import fails here, not above.
        payload = json.dumps({
            "name": "code_diagnostics",
            "arguments": {"path": "definitely-not-a-real-file.ts"},
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/api/tools/execute", data = payload,
            headers = {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout = 60) as r:
            result = json.loads(r.read())
        text = json.dumps(result).lower()
        if "modulenotfounderror" in text or "traceback" in text:
            print(f"FAIL: a lazily imported module is missing from the build:\n{result}")
            return 1

        print("smoke test passed")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout = 20)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify the endpoints exist**

The script assumes `/health`, `/api/tools` and `/api/tools/execute`. Confirm against the real routes:

```bash
grep -rnE "\"/health\"|/api/tools" studio/backend/routes/*.py studio/backend/app.py | head -5
```

If the paths differ, correct the script and say what you found. Do **not** leave it asserting against endpoints that do not exist — it would fail for the wrong reason and look like a packaging bug.

- [ ] **Step 3: Install and run it**

Run `packaging\Output\Unslothed-Setup.exe`, then:

```bash
python packaging/smoke_installed.py "$LOCALAPPDATA/Unslothed/Unslothed.exe"
```
Expected: `smoke test passed`

- [ ] **Step 4: Negative control**

Remove one hidden import from the spec (`cv2` is a good choice — `remove_background` needs it), rebuild, reinstall, and confirm the smoke test FAILS on the tool-call assertion rather than at startup. Restore the import and rebuild.

This is expensive because it means a second full build. It is also the only way to know the test can detect the failure it exists for. If a shorter control can be devised that genuinely exercises the same path, use it and explain why it is equivalent.

- [ ] **Step 5: Commit**

```bash
git add packaging/smoke_installed.py
git commit -m "test(installer): smoke-test the installed build, including a real tool call"
```

---

## Task 6: Dockerfile, both tags

**Files:**
- Create: `packaging/Dockerfile`
- Create: `packaging/docker-entrypoint.sh`
- Create: `packaging/build-docker.ps1`
- Create: `packaging/.dockerignore`

**Interfaces:**
- Produces: images `unslothed:cpu` and `unslothed:cuda`

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
# packaging/Dockerfile
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Two tags from one file:
#   docker build -f packaging/Dockerfile --build-arg VARIANT=cpu  -t unslothed:cpu  .
#   docker build -f packaging/Dockerfile --build-arg VARIANT=cuda -t unslothed:cuda .
#
# :cuda needs the NVIDIA Container Toolkit on the host and `--gpus all` at run
# time. It will not run on a NAS; :cpu is the image for that.

# --- frontend -------------------------------------------------------------
FROM node:22-slim AS frontend
WORKDIR /src
COPY studio/frontend/package.json studio/frontend/package-lock.json ./
RUN npm ci
COPY studio/frontend/ ./
RUN npm run build

# --- runtime bases --------------------------------------------------------
FROM python:3.14-slim AS base-cpu
ENV UNSLOTHED_VARIANT=cpu
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

FROM nvidia/cuda:12.6.2-runtime-ubuntu24.04 AS base-cuda
ENV UNSLOTHED_VARIANT=cuda
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

ARG VARIANT=cpu
FROM base-${VARIANT} AS runtime

# Both language servers are pre-installed. Inside a container the tools'
# "not installed, run npm install -g ..." message is useless -- the user cannot
# easily act on it and the result would not persist. An image where a headline
# feature silently does not work is worse than the ~250 MB this costs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g typescript-language-server typescript@5 \
    && curl -fsSL https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh \
    && bash /tmp/dotnet-install.sh --channel 8.0 --runtime dotnet --install-dir /usr/share/dotnet \
    && ln -sf /usr/share/dotnet/dotnet /usr/local/bin/dotnet \
    && rm -f /tmp/dotnet-install.sh \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.dotnet/tools:${PATH}"
RUN dotnet tool install --global csharp-ls || \
    echo "csharp-ls install failed; code_* C# tools will report unavailable"

WORKDIR /app

COPY studio/backend/requirements/ ./studio/backend/requirements/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" \
         --extra-index-url https://pypi.org/simple \
         -r studio/backend/requirements/base.txt \
    && pip install --no-cache-dir -r studio/backend/requirements/studio.txt

COPY studio/ ./studio/
COPY unsloth/ ./unsloth/
COPY pyproject.toml ./
COPY --from=frontend /src/dist ./studio/frontend/dist

COPY packaging/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Data lives in a volume so containers stay disposable and a NAS deployment can
# point at persistent storage.
ENV UNSLOTH_STUDIO_HOME=/data
VOLUME ["/data"]
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsS http://127.0.0.1:7860/health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
```

- [ ] **Step 2: Write the entrypoint**

```bash
#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
set -euo pipefail

mkdir -p "${UNSLOTH_STUDIO_HOME}"

echo "Unslothed (${UNSLOTHED_VARIANT}) starting"
echo "  data:   ${UNSLOTH_STUDIO_HOME}"
echo "  tsserver: $(command -v typescript-language-server || echo 'NOT FOUND')"
echo "  csharp-ls: $(command -v csharp-ls || echo 'NOT FOUND')"

exec python -m uvicorn studio.backend.app:app --host 0.0.0.0 --port 7860 "$@"
```

- [ ] **Step 3: Write `.dockerignore`**

```
**/node_modules
**/__pycache__
**/.pytest_cache
studio/frontend/dist
packaging/dist
packaging/build
packaging/Output
.git
.superpowers
```

`studio/frontend/dist` is excluded deliberately — the frontend stage builds it, and copying a stale host build in would silently ship the wrong UI.

- [ ] **Step 4: Write the build script**

```powershell
# packaging/build-docker.ps1
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
foreach ($variant in @("cpu", "cuda")) {
    Write-Host ""; Write-Host ("==> building unslothed:" + $variant) -ForegroundColor Cyan
    docker build -f packaging/Dockerfile --build-arg "VARIANT=$variant" -t "unslothed:$variant" .
    if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Host "build failed for $variant" -ForegroundColor Red; exit 1 }
}
Pop-Location
docker images unslothed --format "  {{.Tag}}  {{.Size}}"
```

- [ ] **Step 5: Verify the uvicorn entry point**

The entrypoint assumes `studio.backend.app:app`. Confirm it:

```bash
grep -rn "^app = \|app = FastAPI" studio/backend/app.py | head -3
```

Correct the entrypoint if the module path or variable name differs, and say what you found.

- [ ] **Step 6: Build both**

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build-docker.ps1
```

Expect `:cuda` to be well north of 8 GB. If a build fails, report the actual error — do not silently drop a stage.

- [ ] **Step 7: Commit**

```bash
git add packaging/Dockerfile packaging/docker-entrypoint.sh \
        packaging/build-docker.ps1 packaging/.dockerignore
git commit -m "build(docker): cpu and cuda images, both with the language servers preinstalled"
```

---

## Task 7: Docker verification

**Files:**
- Create: `packaging/verify_docker.ps1`

**Interfaces:**
- Consumes: images `unslothed:cpu` and `unslothed:cuda`

- [ ] **Step 1: Write the verification script**

```powershell
# packaging/verify_docker.ps1
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
#   powershell -ExecutionPolicy Bypass -File packaging\verify_docker.ps1
#
# Asserts, for each tag: the tools register, both language servers are on PATH,
# and data written to the volume survives a container restart. The last one is
# the difference between a usable deployment and one that loses everything on
# every upgrade.

$ErrorActionPreference = "Stop"
$failures = @()
function Check($ok, $msg) { if (-not $ok) { $script:failures += $msg; Write-Host "  FAIL: $msg" -ForegroundColor Red } else { Write-Host "  ok: $msg" -ForegroundColor Green } }

foreach ($tag in @("cpu", "cuda")) {
    Write-Host ""; Write-Host ("==> unslothed:" + $tag) -ForegroundColor Cyan
    $img = "unslothed:$tag"
    if (-not (docker image inspect $img 2>$null)) { $failures += "$img not built"; continue }

    # Language servers must be present -- inside a container the install-on-
    # first-use path is useless.
    $ts = docker run --rm $img bash -lc "command -v typescript-language-server || true"
    Check ($ts -match "typescript-language-server") "$tag : typescript-language-server on PATH"
    $cs = docker run --rm $img bash -lc "command -v csharp-ls || true"
    Check ($cs -match "csharp-ls") "$tag : csharp-ls on PATH"

    # Tools register inside the image.
    $py = 'import sys; sys.path.insert(0, "/app/studio/backend"); ' +
          'from core.inference.tools import ALL_TOOLS, ASSIST_CODE_TOOL_NAMES, ASSIST_VISION_TOOL_NAMES; ' +
          'n={t["function"]["name"] for t in ALL_TOOLS}; ' +
          'ours=ASSIST_CODE_TOOL_NAMES|ASSIST_VISION_TOOL_NAMES; ' +
          'print(len(n), len(ours-n))'
    $out = docker run --rm $img python -c $py 2>&1 | Select-Object -Last 1
    Check ($out -match "^17 0") "$tag : 17 tools register, none of ours missing (got '$out')"

    # Data survives a restart. Write through one container, read from another.
    $vol = "unslothed-verify-$tag"
    docker volume rm $vol 2>$null | Out-Null
    docker run --rm -v "${vol}:/data" $img bash -lc "echo persisted > /data/marker.txt" | Out-Null
    $read = docker run --rm -v "${vol}:/data" $img bash -lc "cat /data/marker.txt 2>/dev/null || true"
    Check ($read -match "persisted") "$tag : data survives a container restart"
    docker volume rm $vol 2>$null | Out-Null
}

Write-Host ""
if ($failures.Count -gt 0) { Write-Host "DOCKER VERIFICATION FAILED ($($failures.Count))" -ForegroundColor Red; exit 1 }
Write-Host "docker verification passed" -ForegroundColor Green
```

- [ ] **Step 2: Run it**

```powershell
powershell -ExecutionPolicy Bypass -File packaging\verify_docker.ps1
```
Expected: all checks pass for both tags.

Note `:cuda` checks run without `--gpus all` — they verify the image is built correctly, not that CUDA works. Testing actual GPU access needs the NVIDIA Container Toolkit and is out of scope for this script; say so in the report if you cannot test it.

- [ ] **Step 3: Negative control**

Temporarily remove the `npm install -g typescript-language-server` line from the Dockerfile, rebuild `:cpu` only, and confirm the verification FAILS on the tsserver check. Restore and rebuild.

- [ ] **Step 4: Commit**

```bash
git add packaging/verify_docker.ps1
git commit -m "test(docker): verify tools, language servers, and volume persistence in both tags"
```

---

## Self-Review

**Spec coverage.** Branding four surfaces → Task 2, with the identifier and AGPL attribution explicitly protected. Installer, self-contained, CUDA bundled, per-user → Tasks 3–4. Installer smoke test exercising a real tool call → Task 5. Docker two tags with both language servers → Task 6. Docker verification including volume persistence → Task 7. The `torchvision` pin the spec requires "before the images are built" → Task 1, first.

**Placeholder scan.** No TBDs. Every step carries the actual file content or the actual command. Three steps (3.2, 5.2, 6.5) verify an assumption against the real repo and instruct the implementer to correct the artefact and report — those are deliberate, because the entry-point and route paths were not confirmed while writing this plan.

**Type consistency.** Tool counts are 17 total / 10 ours / 7 upstream everywhere they appear (Global Constraints, Task 5, Task 7). Image tags are `unslothed:cpu` and `unslothed:cuda` in Tasks 6 and 7. The installed path is `%LOCALAPPDATA%\Unslothed\Unslothed.exe` in both Task 4 and Task 5. The PyInstaller output name `Unslothed` matches `Unslothed.iss`'s `Source: "dist\Unslothed\*"`.

**One known weakness, stated rather than hidden.** Task 5's negative control needs a second full PyInstaller build, which is slow. The step says so and permits an equivalent shorter control if one can be justified. Every other negative control in this plan is cheap.
