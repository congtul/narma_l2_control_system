from PyQt5 import QtWidgets
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.advance_mode_ui import Ui_MainWindow as Ui_Advance

class AdvanceModeWindow(QtWidgets.QMainWindow, Ui_Advance):
    saved_file_path = None

    def __init__(self, parent_window=None):
        super().__init__(parent_window)
        self.setupUi(self)
        self.setWindowTitle("Advance Mode Window")
        self.parent_window = parent_window

        # Buttons
        self.Import_btn.clicked.connect(self.import_file)
        self.Apply_advancemode_btn.clicked.connect(self.apply_imported_file)
        self.Clear_code_btn.clicked.connect(self.clear_imported_file)
        self.OK_advancemode_btn.clicked.connect(self.on_ok_clicked)

        if AdvanceModeWindow.saved_file_path:
            self.imported_file = AdvanceModeWindow.saved_file_path
            self.Status_import_label.setText(
                f"Applied File: {os.path.basename(self.imported_file)}")
            self.Status_import_label.setStyleSheet("color: green; font-weight: bold;")
            self.Apply_advancemode_btn.setEnabled(False)
        else:
            self.imported_file = None
            self.Status_import_label.setText("No file imported")
            self.Status_import_label.setStyleSheet("color: gray; font-style: italic;")
            self.Apply_advancemode_btn.setEnabled(False)

    def import_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Source Code", "", "Source Files (*.py *.c *.cpp);;All Files (*)")
        if not file_path:
            self.Status_import_label.setText("Import cancelled")
            self.Status_import_label.setStyleSheet("color: gray; font-style: italic;")
            return
        if file_path.endswith((".py", ".c", ".cpp")):
            self.imported_file = file_path
            self.Status_import_label.setText(f"File imported: {os.path.basename(file_path)}")
            self.Status_import_label.setStyleSheet("color: green; font-weight: bold;")
            if file_path != AdvanceModeWindow.saved_file_path:
                self.Apply_advancemode_btn.setEnabled(True)
            else:
                self.Apply_advancemode_btn.setEnabled(False)
        else:
            QtWidgets.QMessageBox.critical(
                self, "Invalid File", "Please import a Python (.py), C (.c), or C++ (.cpp) file."
            )

    def apply_imported_file(self):
        if not self.imported_file:
            QtWidgets.QMessageBox.warning(
                self, "No File", "No file has been imported yet.")
            return
        AdvanceModeWindow.saved_file_path = self.imported_file
        self.Status_import_label.setText(f"Input values are set ({os.path.basename(self.imported_file)})")
        self.Status_import_label.setStyleSheet("color: blue; font-weight: bold;")
        self.Apply_advancemode_btn.setEnabled(False)

    def clear_imported_file(self):
        self.imported_file = None
        AdvanceModeWindow.saved_file_path = None
        self.Status_import_label.setText("All values cleared")
        self.Status_import_label.setStyleSheet("color: red; font-weight: bold;")
        self.Apply_advancemode_btn.setEnabled(False)

    def on_ok_clicked(self):
        if self.imported_file and self.Apply_advancemode_btn.isEnabled():
            self.apply_imported_file()
        self.close()

    def closeEvent(self, event):
        if self.parent_window:
            self.parent_window.show()
        event.accept()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = AdvanceModeWindow()
    win.show()
    sys.exit(app.exec_())