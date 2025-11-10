import os
import subprocess

UI_DIR = "ui/"
RES_DIR = "resources/"

print("=== Converting .ui and .qrc files ===")

# Convert all .ui → _ui.py
for file in os.listdir(UI_DIR):
    if file.endswith(".ui"):
        ui_path = os.path.join(UI_DIR, file)
        py_path = os.path.join(UI_DIR, file.replace(".ui", "_ui.py"))

        print(f"[UI ] {file} -> {py_path}")
        subprocess.call(["pyuic5", ui_path, "-o", py_path])


# Convert all .qrc → _rc.py
for root, dirs, files in os.walk(RES_DIR):
    for file in files:
        if file.endswith(".qrc"):
            qrc_path = os.path.join(root, file)
            py_path = os.path.join(UI_DIR, file.replace(".qrc", "_rc.py"))

            print(f"[QRC] {file} -> {py_path}")
            subprocess.call(["pyrcc5", qrc_path, "-o", py_path])

print("=== Done ===")
