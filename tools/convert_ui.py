import os
import subprocess
import sys

UI_DIR = "ui/"
RES_DIR = "resources/"

def convert_ui(file_path):
    py_path = os.path.join(UI_DIR, os.path.basename(file_path).replace(".ui", "_ui.py"))
    print(f"[UI ] {file_path} -> {py_path}")
    subprocess.call(["pyuic5", file_path, "-o", py_path])

def convert_qrc(file_path):
    py_path = os.path.join(UI_DIR, os.path.basename(file_path).replace(".qrc", "_rc.py"))
    print(f"[QRC] {file_path} -> {py_path}")
    subprocess.call(["pyrcc5", file_path, "-o", py_path])

def main():
    # Lấy argument từ command line
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if input_file.endswith(".ui"):
            convert_ui(os.path.join(UI_DIR, input_file))
        elif input_file.endswith(".qrc"):
            convert_qrc(os.path.join(RES_DIR, input_file))
        else:
            print("Only .ui or .qrc files are supported.")
    else:
        print("=== Converting all .ui and .qrc files ===")
        # Convert tất cả .ui
        for file in os.listdir(UI_DIR):
            if file.endswith(".ui"):
                convert_ui(os.path.join(UI_DIR, file))
        # Convert tất cả .qrc
        for root, dirs, files in os.walk(RES_DIR):
            for file in files:
                if file.endswith(".qrc"):
                    convert_qrc(os.path.join(root, file))
        print("=== Done ===")

if __name__ == "__main__":
    main()
