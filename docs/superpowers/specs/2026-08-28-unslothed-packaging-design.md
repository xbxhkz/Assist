# Unslothed Packaging — Design Spec

## Context

Third sub-project of the initiative to fork [Unsloth](https://github.com/unslothai/unsloth/) and
extend it into a local AI agent platform. Sub-projects 1 (vision tools) and 2 (code intelligence)
are merged into `assist-vision-tools` and open as
[PR #1](https://github.com/xbxhkz/unslothed/pull/1).

This sub-project makes the fork *distributable*: it acquires a product identity, a Windows
installer, and Docker images. It adds no agent capabilities.

It is the first of three tracks derived from a 48-section master specification the owner supplied.
The decomposition, recorded here so the ordering is not re-litigated later:

| Track | Contents | Status |
|---|---|---|
| **Packaging** | branding, Windows installer, Docker | **this spec** |
| Capability ports | workflow engine + editor + triggers, scheduler, desktop control, multi-persona crew, hardware monitor, tool-call audit log, notifications | later; each a port of already-reviewed Assist code |
| Net-new | automatic tool acquisition, capability discovery, agent workspaces | last |

Automatic tool acquisition is deliberately last. An agent that installs its own software is the
most security-sensitive item in the master spec — the spec says so itself — and it depends on
permission and audit layers that the capability track brings into Studio.

## Goal

Ship Unslothed as a self-contained Windows installer and as two Docker images, under the product
name "Unslothed", without disturbing the upstream merge seam.

## The constraint that governs every decision here

Both prior sub-projects were built to keep the fork mergeable against an actively developed
upstream. The entire seam is **8 lines in `core/inference/tools.py` (3 hunks)** and **one hunk in
`routes/inference.py`**. That was validated against 69 commits of real upstream drift, which
changed those two files by 1,341 and 194 lines respectively — and the branch still merged cleanly.

Everything in this sub-project is therefore **additive**: new files at the repo root and under
`packaging/`. No upstream Python source is edited.

### The rename that is NOT happening, and why

The owner asked for the project to be "renamed to unslothed". Clarified: **user-visible product
name and branding only, not the Python package**.

That distinction is load-bearing. Renaming the `unsloth` package would touch every `import
unsloth` across the upstream codebase — thousands of lines — and every future upstream merge would
then conflict on nearly every file. It converts a maintainable fork into a permanent hard fork.

Measured: the frontend contains **3089** occurrences of `unsloth`, of which **2351 sit outside the
locale files** — import paths, API routes, CSS classes, the Tauri identifier. A blanket replace
breaks the application. The genuinely user-visible set is **46 strings in `en.ts`**, mirrored
across 12 locales, plus three config values.

## Deliverable 1 — Branding

Four surfaces change:

| Surface | Change |
|---|---|
| `studio/frontend/index.html` | `<title>Unsloth</title>` → `Unslothed` |
| `studio/src-tauri/tauri.conf.json` | `productName` and the window `title` → `Unslothed` |
| `studio/frontend/src/i18n/locales/*.ts` (12 files) | user-facing message **values** only, never keys |
| Installer and Docker | product name, shortcut name, image tag (Deliverables 2 and 3) |

"Unslothed" stays **untranslated in all 12 locales**, treated as a proper noun.

### Deliberately unchanged

**The Tauri `identifier` (`ai.unsloth.studio`).** It is the application's OS-level identity on
Windows and macOS. Changing it makes the OS treat an upgrade as a different application,
orphaning settings and leaving a duplicate entry. It is not user-visible.

**Attribution and licence text.** `en.ts:2` carries the AGPL copyright line — "Unsloth AI Inc.
team. All rights reserved." The project is AGPL-3.0; rewriting upstream's legal attribution would
be wrong irrespective of branding.

**References to upstream as a project**, e.g. model-catalogue entries and dataset filenames such
as `alpaca_unsloth.json`. Those name upstream, not this product.

## Deliverable 2 — Windows installer

Self-contained, CUDA bundled, per-user. No elevation required.

### Pipeline

Four stages, mirroring Assist's proven build (`build-installer.ps1` is 40 lines, `Assist.iss` 37):

```
studio/frontend            -> npm run build         (dist/, ~697 files)
packaging/Unslothed.spec   -> pyinstaller           (portable app folder)
packaging/Unslothed.iss    -> ISCC.exe              (Unslothed-Setup.exe)
packaging/build-installer.ps1  drives all three and resolves the version
```

Inno Setup 6 is already installed at
`C:\Program Files (x86)\Inno Setup 6\ISCC.exe`. PyInstaller belongs to the **build** environment,
not the shipped one.

### Why not Tauri

`studio/src-tauri` exists and is configured to emit an NSIS installer, but its `bundle.externalBin`
and `bundle.resources` are both `None` — the shell *spawns and health-watches* a Python backend it
does not ship. Its installer would install a launcher for a Studio the user must install
separately, while adding a Rust and MSVC toolchain of several GB. It does not solve the packaging
problem.

### The three things that make this harder than Assist

**torch is 2.61 GB and PyInstaller cannot analyse it reliably.** Torch loads extensions
dynamically and ships CUDA DLLs static analysis never sees. Do not fight this: exclude torch from
`Analysis` and copy the built tree in wholesale, and the `nvidia/` CUDA tree with it.

**Lazily imported modules must be declared.** `assist_vision` and `assist_code` import heavy
dependencies *inside function bodies* deliberately, to keep startup fast — a module-scope `import
torch` was measured at a ~2.5s ASGI event-loop stall in the source project. Static analysis cannot
see those imports, so every one needs a `hiddenimports` entry. The five vision dependencies
(`torchvision`, `ultralytics`, `onnxruntime`, `opencv-python`, `insightface`) are included.

**Language servers stay out of the bundle.** `typescript-language-server` and `csharp-ls` install
on first use through npm and `dotnet tool`. Freezing them would mean shipping Node and the .NET
SDK to duplicate a path the design already handles, with a disclosed log line and an actionable
failure message.

### Output

`Unslothed-Setup.exe`, installing to `%LOCALAPPDATA%\Unslothed` with a Start Menu shortcut, an
optional desktop shortcut, and an uninstaller. Expected size ~4.5 GB with CUDA bundled.

**Data is untouched.** Studio keeps models, chats, auth, and exports in `~/.unsloth/studio`. The
installer must not write there, so installing does not disturb an existing setup.

## Deliverable 3 — Docker

Two tags from one multi-stage Dockerfile, differing only in base image and torch wheels:

```
Stage 1  node:22-slim              -> npm ci && npm run build      (frontend dist)
Stage 2  base varies               -> pip install requirements     (runtime)
         :cpu   python:3.14-slim            + torch CPU wheels
         :cuda  nvidia/cuda:12.x-runtime    + torch CUDA wheels
```

Assist's Dockerfile (111 lines) demonstrates the pattern: a wheel-building stage, apt
dependencies, layered pip installs, an entrypoint script, and a healthcheck.

**Both language servers are pre-installed** into both images — Node for
`typescript-language-server`, the .NET runtime for `csharp-ls`, roughly +250 MB. Inside a
container the tools' "not installed, run `npm install -g …`" message is useless: the user cannot
easily act on it and the result would not persist. An image where a headline feature silently does
not work is worse than 250 MB.

**Data lives in a mounted volume**, not the image, so containers stay disposable and a NAS
deployment can point at persistent storage — the shape Assist's Container Manager deployment
already uses.

**`:cuda` requires `--gpus all` and the NVIDIA Container Toolkit on the host.** It will not run on
a Synology NAS; `:cpu` is the image for that.

### One dependency must be pinned first

`torchvision` is the **only** unpinned entry across all eight of Studio's requirements files, and
sub-project 1 added it. Every other entry is pinned.

This is the shape of a failure this project has already hit once: a bare `mcp` in Assist's
requirements resolved to 2.0.0 on a fresh install and killed all four built-in MCP servers. A bare
`torchvision` in an image rebuilt months later can resolve to a version incompatible with the
pinned torch, and the failure will present as a mysterious CUDA error rather than a version
mismatch.

Pin it to match the torch it is built against, in this sub-project, before the images are built.

## Verification

Every deliverable gets a check that can fail, and a negative control proving it can.

**Branding.** A script asserting the four surfaces changed, that `identifier` did **not**, that no
import path or API route changed, and that `npm run i18n:check` passes across all 12 locales.
Negative control: a deliberately over-broad replace must trip the import-path assertion.

**Installer.** A smoke test that runs the **installed executable**, not the source tree: start the
backend, assert all 17 tools register (the 12 upstream plus our 10 — 5 vision, 5 code), and assert
a lazily imported module actually loads by making a real tool call. A frozen build can register a
tool that explodes on first use; only exercising one detects that.

**Docker.** Build both tags. For each: run it, assert the tools register, assert
`typescript-language-server` and `csharp-ls` are on `PATH`, and assert data written to the mounted
volume survives a container restart.

## Risks

**PyInstaller and torch is the substantive risk** — not the build scripting, which is small and
proven, but a 2.61 GB dependency tree with dynamically loaded CUDA DLLs. The exclude-and-copy
mitigation is standard, but this is where the sub-project may need extra fix rounds.

**The `:cuda` image will be large** — an NVIDIA runtime base plus CUDA torch wheels lands well
north of 8 GB.

**Frozen-build failures arrive late and read as unrelated.** This project has hit a
`subprocess.run` hang from a missing `stdin=DEVNULL` and an undeclared `psutil`, both surfacing
only in the frozen build. That history is why the installer smoke test makes a real tool call
rather than checking that a process starts.

## Out of scope

- Every capability port (workflows, scheduler, desktop control, crew, hardware monitor, audit log,
  notifications) — the second track, each its own spec
- Automatic tool acquisition, capability discovery, agent workspaces — the third track
- Renaming the Python package or the Tauri identifier
- macOS and Linux installers; Tauri already emits `dmg`, `deb`, and `appimage` targets, but only
  Windows is in scope here
- Code signing. `Unslothed-Setup.exe` will be unsigned and will raise a SmartScreen warning, the
  same as Assist's installer does today
