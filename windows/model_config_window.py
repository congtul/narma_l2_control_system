# -*- coding: utf-8 -*-
import sys, os
from PyQt5 import QtCore, QtGui, QtWidgets
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.model_config_ui import Ui_MainWindow  # UI đã convert từ Qt Designer
from windows.model_train_window import ModelTrainWindow

class ModelConfigWindow(QtWidgets.QMainWindow):
    """Window cấu hình model, tương đương với file config hiện tại + Utils"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Internal state
        self.required_edits_all = [
            self.ui.hidden_layers_input, self.ui.delayed_inputs_input, self.ui.delayed_outputs_input,
            self.ui.training_samples_input, self.ui.max_plant_input, self.ui.min_plant_input,
            self.ui.max_interval_left_input, self.ui.max_plant_output, self.ui.min_plant_output,
            self.ui.max_interval_right_input, self.ui.training_epochs_input
        ]
        self.required_edits_no_epoch = [w for w in self.required_edits_all if w is not self.ui.training_epochs_input]

        self._setup_validators()
        self._connect_signals()
        self.update_buttons_state()

        # Placeholder for child window
        self.train_win = None

        # Gắn signal trực tiếp
        self.ui.import_weight_btn.clicked.connect(self.open_network_weight)
        self.ui.generate_data_btn.clicked.connect(self.run_generate_data)
        self.ui.import_data_btn.clicked.connect(self.open_network_weight)
        self.ui.cancel_btn.clicked.connect(self.close)
        self.ui.train_btn.clicked.connect(self.run_model_train)
        self.ui.ok_btn.clicked.connect(self.close)
        self.ui.default_btn.clicked.connect(self.set_default_parameters)

    # ---------------- Validators ----------------
    def _setup_validators(self):
        int_validator = QtGui.QIntValidator(-1_000_000, 1_000_000, self)
        double_validator = QtGui.QDoubleValidator(-1e9, 1e9, 6, self)
        double_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)

        for w in [self.ui.hidden_layers_input, self.ui.delayed_inputs_input, self.ui.delayed_outputs_input,
                  self.ui.training_samples_input, self.ui.training_epochs_input]:
            w.setValidator(int_validator)
        for w in [self.ui.max_plant_input, self.ui.min_plant_input, self.ui.max_interval_left_input,
                  self.ui.max_plant_output, self.ui.min_plant_output, self.ui.max_interval_right_input]:
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
        self.ui.train_btn.setEnabled(self.all_valid())           # Train
        self.ui.generate_data_btn.setEnabled(self.all_valid_no_epoch())  # Generate Data

    # ---------------- Collect parameters ----------------
    def collect_common_parameters(self):
        ui = self.ui
        return {
            "hidden": int(ui.hidden_layers_input.text()),
            "delay_in": int(ui.delayed_inputs_input.text()),
            "delay_out": int(ui.delayed_outputs_input.text()),
            "train_samples": int(ui.training_samples_input.text()),
            "max_in_l": float(ui.max_plant_input.text()),
            "min_in_l": float(ui.min_plant_input.text()),
            "max_int_l": float(ui.max_interval_left_input.text()),
            "min_in_r": float(ui.max_plant_output.text()),
            "max_in_r": float(ui.min_plant_output.text()),
            "max_int_r": float(ui.max_interval_right_input.text()),
            "use_val": ui.use_validation_checkbox.isChecked(),
            "use_test": ui.use_test_checkbox.isChecked(),
        }

    def collect_train_parameters(self):
        p = self.collect_common_parameters()
        p["epochs"] = int(self.ui.training_epochs_input.text())
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
        ui.hidden_layers_input.setText("9")
        ui.delayed_inputs_input.setText("3")
        ui.delayed_outputs_input.setText("2")
        ui.training_samples_input.setText("100000")
        ui.max_plant_input.setText("4")
        ui.min_plant_input.setText("-1")
        ui.max_interval_left_input.setText("60")
        ui.max_plant_output.setText("100000")
        ui.min_plant_output.setText("0")
        ui.max_interval_right_input.setText("0")
        ui.training_epochs_input.setText("100")
        ui.use_validation_checkbox.setChecked(True)
        ui.use_test_checkbox.setChecked(True)
        self.update_buttons_state()

    def closeEvent(self, event):
        """Ensure child training windows are closed when config window closes."""
        if getattr(self, "train_win", None) is not None:
            try:
                self.train_win.close()
            except Exception:
                pass
        super().closeEvent(event)


# ---------------- Main ----------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = ModelConfigWindow()
    win.show()
    sys.exit(app.exec_())
