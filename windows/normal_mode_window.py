from PyQt5 import QtWidgets
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.normal_mode_ui import Ui_MainWindow as Ui_Normal

class NormalModeWindow(QtWidgets.QMainWindow, Ui_Normal):
    saved_mode = None
    saved_random_values = {"max_vel": "", "min_vel": ""}
    saved_manual_values = {"stime": "", "veloval": ""}

    def __init__(self, parent_window=None):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Normal Mode Window")
        self.parent_window = parent_window
        self.active_mode = None

        restored = self.restore_previous_state()
        if not restored:
            # Disable all editable fields initially
            self.RD_max_vel_line_edit.setEnabled(False)
            self.RD_min_vel_line_edit.setEnabled(False)
            self.man_Stime_lineedit.setEnabled(False)
            self.man_Veloval_lineedit.setEnabled(False)

        # Connect buttons
        self.Random_gen_btn.clicked.connect(self.use_random_mode)
        self.Manual_gen_btn.clicked.connect(self.use_manual_mode)
        self.Reset_all_val_btn.clicked.connect(self.reset_all_values)
        self.Apply_normalmode_btn.clicked.connect(self.apply_normal_mode)
        self.OK_normalmode_btn_2.clicked.connect(self.on_ok_clicked)

        # Connect text change signals
        self.RD_max_vel_line_edit.textChanged.connect(self.update_apply_button_state)
        self.RD_min_vel_line_edit.textChanged.connect(self.update_apply_button_state)
        self.man_Stime_lineedit.textChanged.connect(self.update_apply_button_state)
        self.man_Veloval_lineedit.textChanged.connect(self.update_apply_button_state)

    def restore_previous_state(self):
        if NormalModeWindow.saved_mode == "random":
            self.active_mode = "random"
            self.use_random_mode()
            self.RD_max_vel_line_edit.setText(NormalModeWindow.saved_random_values["max_vel"])
            self.RD_min_vel_line_edit.setText(NormalModeWindow.saved_random_values["min_vel"])
            self.man_Stime_lineedit.setText(NormalModeWindow.saved_manual_values["stime"])
            self.man_Veloval_lineedit.setText(NormalModeWindow.saved_manual_values["veloval"])
            return True
        elif NormalModeWindow.saved_mode == "manual":
            self.active_mode = "manual"
            self.use_manual_mode()
            self.man_Stime_lineedit.setText(NormalModeWindow.saved_manual_values["stime"])
            self.man_Veloval_lineedit.setText(NormalModeWindow.saved_manual_values["veloval"])
            self.RD_max_vel_line_edit.setText(NormalModeWindow.saved_random_values["max_vel"])
            self.RD_min_vel_line_edit.setText(NormalModeWindow.saved_random_values["min_vel"])
            return True
        else:
            self.active_mode = None
            self.Apply_normalmode_btn.setEnabled(False)
            return False

    def use_random_mode(self):
        self.active_mode = "random"
        self.Status_nor_label.setText("Random mode is chosen!")
        self.Status_nor_label.setStyleSheet("color: #ffa23a; font-weight: bold;")
        self.RD_max_vel_line_edit.setEnabled(True)
        self.RD_min_vel_line_edit.setEnabled(True)
        self.man_Stime_lineedit.setEnabled(False)
        self.man_Veloval_lineedit.setEnabled(False)
        self.update_apply_button_state()

    def use_manual_mode(self):
        self.active_mode = "manual"
        self.Status_nor_label.setText("Manual mode is chosen!")
        self.Status_nor_label.setStyleSheet("color: blue; font-weight: bold;")
        self.man_Stime_lineedit.setEnabled(True)
        self.man_Veloval_lineedit.setEnabled(True)
        self.RD_max_vel_line_edit.setEnabled(False)
        self.RD_min_vel_line_edit.setEnabled(False)
        self.update_apply_button_state()

    def reset_all_values(self):
        self.active_mode = None
        self.RD_max_vel_line_edit.clear()
        self.RD_min_vel_line_edit.clear()
        self.man_Stime_lineedit.clear()
        self.man_Veloval_lineedit.clear()
        self.RD_max_vel_line_edit.setEnabled(False)
        self.RD_min_vel_line_edit.setEnabled(False)
        self.man_Stime_lineedit.setEnabled(False)
        self.man_Veloval_lineedit.setEnabled(False)
        self.Apply_normalmode_btn.setEnabled(False)
        self.Status_nor_label.setText("Reset all values")
        self.Status_nor_label.setStyleSheet("color: red; font-weight: bold;")

    def update_apply_button_state(self):
        if self.active_mode == "random":
            self.Apply_normalmode_btn.setEnabled(
                bool(self.RD_max_vel_line_edit.text().strip() and
                     self.RD_min_vel_line_edit.text().strip()))
        elif self.active_mode == "manual":
            self.Apply_normalmode_btn.setEnabled(
                bool(self.man_Stime_lineedit.text().strip() and
                     self.man_Veloval_lineedit.text().strip()))
        else:
            self.Apply_normalmode_btn.setEnabled(False)

    def apply_normal_mode(self):
        if self.active_mode == "random":
            NormalModeWindow.saved_mode = "random"
            NormalModeWindow.saved_random_values = {
                "max_vel": self.RD_max_vel_line_edit.text(),
                "min_vel": self.RD_min_vel_line_edit.text(),
            }
        elif self.active_mode == "manual":
            NormalModeWindow.saved_mode = "manual"
            NormalModeWindow.saved_manual_values = {
                "stime": self.man_Stime_lineedit.text(),
                "veloval": self.man_Veloval_lineedit.text(),
            }
        self.Apply_normalmode_btn.setEnabled(False)
        if self.active_mode:
            self.Status_nor_label.setText(f"{self.active_mode.capitalize()} values applied successfully!")
            self.Status_nor_label.setStyleSheet("color: green; font-weight: bold;")

    def on_ok_clicked(self):
        if self.Apply_normalmode_btn.isEnabled():
            self.apply_normal_mode()
        self.close()

    def closeEvent(self, event):
        if self.parent_window:
            self.parent_window.show()
        event.accept()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = NormalModeWindow()
    win.show()
    sys.exit(app.exec_())
