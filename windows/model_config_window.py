# -*- coding: utf-8 -*-
import sys, os
from PyQt5 import QtCore, QtGui, QtWidgets
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.model_config_ui import Ui_MainWindow  # UI generated from Qt Designer
from windows.model_train_window import ModelTrainWindow
from backend.system_workspace import workspace


class ModelConfigWindow(QtWidgets.QMainWindow):
    """Model configuration window plus utilities."""

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
        self.restore_saved_parameters()
        self.update_buttons_state()

        # Placeholder for child window
        self.train_win = None

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

        self.ui.import_weight_btn.clicked.connect(self.open_network_weight)
        self.ui.generate_data_btn.clicked.connect(self.run_generate_data)
        self.ui.import_data_btn.clicked.connect(self.open_network_weight)
        self.ui.cancel_btn.clicked.connect(self.close)
        self.ui.train_btn.clicked.connect(self.run_model_train)
        self.ui.ok_btn.clicked.connect(self.close)
        self.ui.default_btn.clicked.connect(self.set_default_parameters)
        self.ui.save_btn.clicked.connect(self.handle_save)

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
            "min_int_r": float(ui.max_interval_right_input.text()),
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

        if getattr(self, "train_win", None) is None:
            self.train_win = ModelTrainWindow(parent=self)
            self.destroyed.connect(self.train_win.close)
            self.train_win.destroyed.connect(lambda: setattr(self, "train_win", None))

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
            self,
            "Select Config File",
            "",
            "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Could not read JSON file:\n{e}")
            return

        ok, msg = self._validate_config_dict(cfg)
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Invalid Config", msg)
            return

        self._apply_config_dict(cfg)
        
        QtWidgets.QMessageBox.information(self, "Loaded", "Configuration loaded.")
        self.update_buttons_state()

    def _validate_config_dict(self, cfg):
        required_nums = [
            "hidden", "delay_in", "delay_out", "train_samples",
            "max_in_l", "min_in_l", "max_int_l",
            "min_in_r", "max_in_r", "min_int_r", "epochs"
        ]
        for key in required_nums:
            if key not in cfg:
                return False, f"Missing numeric field: {key}"
            if not isinstance(cfg[key], (int, float)):
                return False, f"Field '{key}' must be a number."
        for key in ["use_val", "use_test"]:
            if key in cfg and not isinstance(cfg[key], bool):
                return False, f"Field '{key}' must be true/false."
        return True, ""

    def _apply_config_dict(self, cfg):
        mapping = {
            "hidden": self.ui.hidden_layers_input,
            "delay_in": self.ui.delayed_inputs_input,
            "delay_out": self.ui.delayed_outputs_input,
            "train_samples": self.ui.training_samples_input,
            "max_in_l": self.ui.max_plant_input,
            "min_in_l": self.ui.min_plant_input,
            "max_int_l": self.ui.max_interval_left_input,
            "min_in_r": self.ui.max_plant_output,
            "max_in_r": self.ui.min_plant_output,
            "min_int_r": self.ui.max_interval_right_input,
            "epochs": self.ui.training_epochs_input,
        }
        for key, widget in mapping.items():
            if key in cfg:
                widget.setText(str(cfg[key]))
        if "use_val" in cfg:
            self.ui.use_validation_checkbox.setChecked(bool(cfg.get("use_val")))
        if "use_test" in cfg:
            self.ui.use_test_checkbox.setChecked(bool(cfg.get("use_test")))

    def handle_save(self):
        if not self.all_valid():
            QtWidgets.QMessageBox.warning(self, "Invalid input")
            return
        workspace.narma_config = self.collect_train_parameters()

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

    def restore_saved_parameters(self):
        """Populate fields from saved workspace configuration if available."""
        cfg = getattr(workspace, "narma_config", {}) or {}
        mapping = {
            "hidden": self.ui.hidden_layers_input,
            "delay_in": self.ui.delayed_inputs_input,
            "delay_out": self.ui.delayed_outputs_input,
            "train_samples": self.ui.training_samples_input,
            "max_in_l": self.ui.max_plant_input,
            "min_in_l": self.ui.min_plant_input,
            "max_int_l": self.ui.max_interval_left_input,
            "min_in_r": self.ui.max_plant_output,
            "max_in_r": self.ui.min_plant_output,
            "min_int_r": self.ui.max_interval_right_input,
            "epochs": self.ui.training_epochs_input,
        }
        for key, widget in mapping.items():
            if key in cfg:
                widget.setText(str(cfg[key]))
        if "use_val" in cfg:
            self.ui.use_validation_checkbox.setChecked(bool(cfg.get("use_val")))
        if "use_test" in cfg:
            self.ui.use_test_checkbox.setChecked(bool(cfg.get("use_test")))

    def closeEvent(self, event):
        if self.train_win is not None and self.train_win.isVisible():
            self.train_win.close()
            self.train_win = None
        super().closeEvent(event)


# ---------------- Main ----------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = ModelConfigWindow()
    win.show()
    sys.exit(app.exec_())
