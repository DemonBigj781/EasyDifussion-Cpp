from pathlib import Path

source = Path(SPECPATH)
a = Analysis([str(source / "trainer.py")], pathex=[str(source)],
             binaries=[], datas=[], hiddenimports=[], hookspath=[],
             excludes=["torch", "numpy", "onnxruntime", "transformers", "diffusers"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="sdkit-trainer", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="sdkit-trainer")
