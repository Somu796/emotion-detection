# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\Sudipta\\Documents\\emotion_detection\\models\\emotion-ferplus-8.onnx', 'models'), ('C:\\Users\\Sudipta\\Documents\\emotion_detection\\models\\RFB-320\\RFB-320.caffemodel', 'models/RFB-320'), ('C:\\Users\\Sudipta\\Documents\\emotion_detection\\models\\RFB-320\\RFB-320.prototxt', 'models/RFB-320')],
    hiddenimports=['cv2', 'numpy', 'tkinter', 'onnx'],
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
    a.binaries,
    a.datas,
    [],
    name='EmotionDetection',
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
