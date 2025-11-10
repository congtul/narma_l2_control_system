from PyQt5 import QtWidgets, QtCore
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.input_ref_ui import Ui_MainWindow as Ui_Input
from windows.normal_mode_window import NormalModeWindow
from windows.advance_mode_window import AdvanceModeWindow

class InputWindow(QtWidgets.QMainWindow, Ui_Input):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Input Window")

        # install double-click filters for its own buttons
        self.Normal_mode_btn.installEventFilter(self)
        self.Advance_mode_btn.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonDblClick and event.button() == QtCore.Qt.LeftButton:
            if obj == self.Normal_mode_btn:
                self.open_normal_mode()
            elif obj == self.Advance_mode_btn:
                self.open_advance_mode()
            return True
        return super().eventFilter(obj, event)

    def open_normal_mode(self):
        self.normal_window = NormalModeWindow(self)
        self.normal_window.show()
        self.hide()

    def open_advance_mode(self):
        self.advance_window = AdvanceModeWindow(self)
        self.advance_window.show()
        self.hide()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = InputWindow()
    win.show()
    sys.exit(app.exec_())
