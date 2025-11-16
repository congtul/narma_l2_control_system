from PyQt5 import QtWidgets, QtCore
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.input_ref_ui import Ui_MainWindow as Ui_Input
from windows.normal_mode_window import NormalModeWindow
from windows.advance_mode_window import AdvanceModeWindow

class InputWindow(QtWidgets.QMainWindow, Ui_Input):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Input Window")
        self.main_window_ref = None 

        # connect button clicks (single click)
        self.Normal_mode_btn.clicked.connect(self.open_normal_mode)
        self.Advance_mode_btn.clicked.connect(self.open_advance_mode)

    def open_normal_mode(self):
        self.normal_window = NormalModeWindow(self)
        self.normal_window.show()
        self.hide()

    def open_advance_mode(self):
        self.advance_window = AdvanceModeWindow(self)
        self.advance_window.main_window_ref = self.main_window_ref
        self.advance_window.show()
        self.hide()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = InputWindow()
    win.show()
    sys.exit(app.exec_())
