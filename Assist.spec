# -*- mode: python ; coding: utf-8 -*-
# Committed PyInstaller spec for the Assist native Windows build (Phase 2).
from PyInstaller.utils.hooks import collect_all

# Heavy, dynamically-imported packages PyInstaller's static analysis misses.
# collect_all pulls their submodules, data files, and native binaries.
_collected_datas = []
_collected_binaries = []
_collected_hidden = []
for _pkg in ("chromadb", "onnxruntime", "fastembed", "tokenizers",
             # Local Speech-to-Text (voice input): faster-whisper runs on the
             # CTranslate2 backend and decodes mic audio (webm/opus) via PyAV.
             # All three ship native DLLs that PyInstaller's static analysis
             # misses, so collect_all pulls their binaries + data.
             "faster_whisper", "ctranslate2", "av",
             # Webcam object detection (webcam_look tool): YOLO via ultralytics.
             # collect_all pulls its submodules/data (torch + cv2 already handled).
             "ultralytics"):
    _d, _b, _h = collect_all(_pkg)
    _collected_datas += _d
    _collected_binaries += _b
    _collected_hidden += _h

datas = [
    ('static', 'static'),
    ('scripts', 'scripts'),
    ('mcp_servers', 'mcp_servers'),
    ('services/hwfit/data', 'services/hwfit/data'),
    ('config', 'config'),
    ('.env.example', '.env.example'),
    # Offline embedding model (populated by scripts/fetch_embedding_model.py).
    ('build_assets/fastembed_cache', 'fastembed_cache'),
    # Bundled CPU llama-server for native local models (Phase 3a).
    ('build_assets/llama', 'llama'),
    # Bundled sd-server (stable-diffusion.cpp) for native image generation:
    # CPU (avx2) + Vulkan GPU builds, populated by scripts/fetch_sd_server.py.
    ('build_assets/sd', 'sd'),
    # Bundled YOLO weight (yolov8n.pt) so webcam object detection works offline.
    ('build_assets/yolo', 'yolo'),
] + _collected_datas

hiddenimports = [
    'webview', 'webview.platforms.edgechromium',
    # mss (screen capture) is imported lazily inside a function
    # (src/desktop/capture.py _default_grabber), so PyInstaller's static
    # analysis can't see it — declare it explicitly or capture_screen fails
    # at runtime in the frozen build with ModuleNotFoundError: mss.
    'mss', 'mss.windows',
    # comtypes (UI Automation backend, src/desktop/uia.py _real_automation)
    # is imported lazily so its COM plumbing never runs in unit tests — but
    # that also hides it from PyInstaller's static analysis. Declare it
    # explicitly or the frozen build fails at runtime with ModuleNotFoundError.
    'comtypes', 'comtypes.client', 'comtypes.gen',
] + _collected_hidden

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=_collected_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Assist',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX can corrupt onnxruntime / native DLLs; keep off for safety.
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['static\\icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Assist',
)
