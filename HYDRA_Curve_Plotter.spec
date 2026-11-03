# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['hydra_curve_plotter.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['openpyxl', 'matplotlib.backends.backend_qtagg'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6', 'tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='HYDRA_Curve_Plotter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
