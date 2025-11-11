# -*- coding: utf-8 -*-
import sys, os
from PyQt5 import QtCore, QtGui, QtWidgets
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.model_config_ui import Ui_MainWindow  # UI đã convert từ Qt Designer
from windows.model_train_window import ModelTrainWindow

class ModelConfigWindow(QtWidgets.QMainWindow):
    """Window cấu hình model, tương đương với file config hiện tại + Utils"""
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Internal state
        self.required_edits_all = [
            self.ui.lineEdit_6, self.ui.lineEdit_7, self.ui.lineEdit_8,
            self.ui.lineEdit, self.ui.lineEdit_2, self.ui.lineEdit_3,
            self.ui.lineEdit_4, self.ui.lineEdit_5, self.ui.lineEdit_9,
            self.ui.lineEdit_10, self.ui.lineEdit_11
        ]
        self.required_edits_no_epoch = [w for w in self.required_edits_all if w is not self.ui.lineEdit_11]

        self._setup_validators()
        self._connect_signals()
        self.update_buttons_state()

        # Placeholder for child window
        self.train_win = None

        # Gắn signal trực tiếp
        self.ui.pushButton.clicked.connect(self.open_network_weight)
        self.ui.pushButton_2.clicked.connect(self.run_generate_data)
        self.ui.pushButton_4.clicked.connect(self.open_network_weight)
        self.ui.pushButton_5.clicked.connect(self.close)
        self.ui.pushButton_6.clicked.connect(self.run_model_train)
        self.ui.pushButton_7.clicked.connect(self.close)
        self.ui.pushButton_8.clicked.connect(self.set_default_parameters)

    # ---------------- Validators ----------------
    def _setup_validators(self):
        int_validator = QtGui.QIntValidator(-1_000_000, 1_000_000, self)
        double_validator = QtGui.QDoubleValidator(-1e9, 1e9, 6, self)
        double_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)

        for w in [self.ui.lineEdit_6, self.ui.lineEdit_7, self.ui.lineEdit_8,
                  self.ui.lineEdit, self.ui.lineEdit_11]:
            w.setValidator(int_validator)
        for w in [self.ui.lineEdit_2, self.ui.lineEdit_3, self.ui.lineEdit_4,
                  self.ui.lineEdit_5, self.ui.lineEdit_9, self.ui.lineEdit_10]:
            w.setValidator(double_validator)

    def _connect_signals(self):
        for w in self.required_edits_all:
            w.textChanged.connect(self.update_buttons_state)

    @staticmethod
    def _is_valid(widget):
        text = widget.text().strip()
        if not text:
            return False
        v = widget.validator()
        if v is None:
            return True
        state, _, _ = v.validate(text, 0)
        return state == QtGui.QValidator.Acceptable

    # ---------------- State ----------------
    def all_valid(self):
        return all(self._is_valid(w) for w in self.required_edits_all)

    def all_valid_no_epoch(self):
        return all(self._is_valid(w) for w in self.required_edits_no_epoch)

    def update_buttons_state(self):
        self.ui.pushButton_6.setEnabled(self.all_valid())           # Train
        self.ui.pushButton_2.setEnabled(self.all_valid_no_epoch())  # Generate Data

    # ---------------- Collect parameters ----------------
    def collect_common_parameters(self):
        ui = self.ui
        return {
            "hidden": int(ui.lineEdit_6.text()),
            "delay_in": int(ui.lineEdit_7.text()),
            "delay_out": int(ui.lineEdit_8.text()),
            "train_samples": int(ui.lineEdit.text()),
            "max_in_l": float(ui.lineEdit_2.text()),
            "min_in_l": float(ui.lineEdit_3.text()),
            "max_int_l": float(ui.lineEdit_4.text()),
            "min_in_r": float(ui.lineEdit_5.text()),
            "max_in_r": float(ui.lineEdit_9.text()),
            "max_int_r": float(ui.lineEdit_10.text()),
            "use_val": ui.checkBox_2.isChecked(),
            "use_test": ui.checkBox_3.isChecked(),
        }

    def collect_train_parameters(self):
        p = self.collect_common_parameters()
        p["epochs"] = int(self.ui.lineEdit_11.text())
        return p

    # ---------------- Actions ----------------
    def run_model_train(self):
        if not self.all_valid():
            QtWidgets.QMessageBox.warning(self, "Invalid input", "Please fill all fields correctly.")
            return

        # Giữ reference để cửa sổ không tự tắt
        if getattr(self, "train_win", None) is None:
            self.train_win = ModelTrainWindow(parent=self)
            self.destroyed.connect(self.train_win.close)
            self.train_win.destroyed.connect(lambda: setattr(self, "train_win", None))

        # Bạn có thể thêm method set_parameters trong ModelTrainWindow để truyền params
        # self.train_win.set_parameters(self.collect_train_parameters())

        self.train_win.show()

    def run_generate_data(self):
        if not self.all_valid_no_epoch():
            QtWidgets.QMessageBox.warning(self, "Invalid input", "Please fill all fields (epochs optional).")
            return
        import subprocess
        import json
        import sys
        params = self.collect_common_parameters()
        subprocess.Popen([sys.executable, "generate_data.py", json.dumps(params)],
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def open_network_weight(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Weight File", "", "All Files (*);;HDF5 Files (*.h5);;Text Files (*.txt)"
        )
        if path:
            print(f"Selected file: {path}")

    def set_default_parameters(self):
        ui = self.ui
        ui.lineEdit_6.setText("9")
        ui.lineEdit_7.setText("3")
        ui.lineEdit_8.setText("2")
        ui.lineEdit.setText("100000")
        ui.lineEdit_2.setText("4")
        ui.lineEdit_3.setText("-1")
        ui.lineEdit_4.setText("60")
        ui.lineEdit_5.setText("100000")
        ui.lineEdit_9.setText("0")
        ui.lineEdit_10.setText("0")
        ui.lineEdit_11.setText("100")
        ui.checkBox_2.setChecked(True)
        ui.checkBox_3.setChecked(True)
        self.update_buttons_state()


# ---------------- Main ----------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = ModelConfigWindow()
    win.show()
    sys.exit(app.exec_())
