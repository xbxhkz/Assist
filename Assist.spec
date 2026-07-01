# -*- mode: python ; coding: utf-8 -*-
# Committed PyInstaller spec for the Assist native Windows build (Phase 2).
from PyInstaller.utils.hooks import collect_all

# Heavy, dynamically-imported packages PyInstaller's static analysis misses.
# collect_all pulls their submodules, data files, and native binaries.
_collected_datas = []
_collected_binaries = []
_collected_hidden = []
for _pkg in ("chromadb", "onnxruntime", "fastembed", "tokenizers"):
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
] + _collected_datas

hiddenimports = [
    'webview', 'webview.platforms.edgechromium',
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
