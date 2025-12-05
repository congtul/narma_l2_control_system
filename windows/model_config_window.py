# -*- coding: utf-8 -*-
import sys, os, json
from PyQt5 import QtCore, QtGui, QtWidgets
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.model_config_ui import Ui_MainWindow  # UI generated from Qt Designer
from windows.model_train_window import ModelTrainWindow
from backend.system_workspace import workspace
from backend.narma_l2_model import NARMA_L2_Controller
from backend import utils
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import copy
from backend.system_workspace import workspace


class ModelConfigWindow(QtWidgets.QMainWindow):
    """Model configuration window plus utilities."""

    def __init__(self, parent=None, current_role=None):
        super().__init__(parent)
        self.current_role = current_role
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Internal state
        self.required_edits_all = [
            self.ui.hidden_layers_input, self.ui.delayed_inputs_input, self.ui.delayed_outputs_input,
            self.ui.training_samples_input, self.ui.max_plant_input, self.ui.min_plant_input,
            self.ui.min_interval_input, self.ui.max_plant_output, self.ui.min_plant_output,
            self.ui.max_interval_input, self.ui.training_epochs_input
        ]
        self.required_edits_no_epoch = [w for w in self.required_edits_all if w is not self.ui.training_epochs_input]

        self._setup_validators()
        self._connect_signals()
        self.restore_saved_parameters()
        self.update_buttons_state()
        self._apply_role_permissions()

        # Placeholder for child window
        self.train_win = None
        if workspace.first_save_config:
            self.ui.delayed_inputs_input.setText(str(workspace.narma_config.get("nu")))
            self.ui.delayed_outputs_input.setText(str(workspace.narma_config.get("ny")))
            self.ui.hidden_layers_input.setText(str(workspace.narma_config.get("hidden_size")))
            self.ui.training_epochs_input.setText(str(workspace.narma_config.get("training_epochs")))
            self.ui.training_samples_input.setText(str(workspace.narma_config.get("training_sample_size")))
            self.ui.max_plant_input.setText(str(workspace.narma_config.get("max_control")))
            self.ui.min_plant_input.setText(str(workspace.narma_config.get("min_control")))
            self.ui.max_plant_output.setText(str(workspace.narma_config.get("max_output")))
            self.ui.min_plant_output.setText(str(workspace.narma_config.get("min_output")))
            self.ui.min_interval_input.setText(str(workspace.narma_config.get("min_interval")))
            self.ui.max_interval_input.setText(str(workspace.narma_config.get("max_interval")))
            self.ui.use_validation_checkbox.setChecked(workspace.narma_config.get("use_validation"))
            self.ui.use_test_checkbox.setChecked(workspace.narma_config.get("use_test_data"))
            self.ui.train_btn.setEnabled(True)


    # ---------------- Validators ----------------
    def _setup_validators(self):
        int_validator = QtGui.QIntValidator(-1_000_000, 1_000_000, self)
        double_validator = QtGui.QDoubleValidator(-1e9, 1e9, 6, self)
        double_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)

        for w in [self.ui.hidden_layers_input, self.ui.delayed_inputs_input, self.ui.delayed_outputs_input,
                  self.ui.training_samples_input, self.ui.training_epochs_input]:
            w.setValidator(int_validator)
        for w in [self.ui.max_plant_input, self.ui.min_plant_input, self.ui.min_interval_input,
                  self.ui.max_plant_output, self.ui.min_plant_output, self.ui.max_interval_input]:
            w.setValidator(double_validator)

    def _connect_signals(self):
        for w in self.required_edits_all:
            w.textChanged.connect(self.update_buttons_state)

        self.ui.import_weight_btn.clicked.connect(self.open_network_weight)
        self.ui.generate_data_btn.clicked.connect(self.run_generate_data)
        self.ui.export_data_btn.clicked.connect(self.handle_export_data)
        self.ui.import_data_btn.clicked.connect(self.handle_import_data)
        self.ui.export_weight_btn.clicked.connect(self.handle_export_weight)
        self.ui.train_btn.clicked.connect(self.run_model_train)
        self.ui.save_model_btn.clicked.connect(self.handle_save_model)
        self.ui.default_btn.clicked.connect(self.set_default_parameters)
        self.ui.save_btn.clicked.connect(self.handle_save_config)

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
        self.ui.train_btn.setEnabled(self.all_valid() and self._has_dataset())  # Train
        self.ui.generate_data_btn.setEnabled(self.all_valid_no_epoch())  # Generate Data
        base_ok = all(self._is_valid(w) for w in [
            self.ui.hidden_layers_input,
            self.ui.delayed_inputs_input,
            self.ui.delayed_outputs_input,
        ])
        self.ui.import_weight_btn.setEnabled(base_ok)

    def _has_dataset(self):
        if not hasattr(workspace, "dataset"):
            return False
        ds = workspace.dataset
        return (
            isinstance(ds, dict)
            and all(k in ds for k in ["t", "u", "y"])
            and any(len(ds[k]) > 0 for k in ["t", "u", "y"])
        )

    # ---------------- Collect parameters ----------------
    def collect_common_parameters(self):
        ui = self.ui
        return {
            "nu": int(ui.delayed_inputs_input.text()),
            "ny": int(ui.delayed_outputs_input.text()),
            "hidden_size": int(ui.hidden_layers_input.text()),
            "activation": "SiLU",
            "learning_rate": 1e-4,
            "training_epochs": int(ui.training_epochs_input.text()),
            "training_sample_size": int(ui.training_samples_input.text()),
            "backprop_batch_size": 32,
            "max_control": float(ui.max_plant_input.text()), # voltage giới hạn
            "min_control": float(ui.min_plant_input.text()),
            "max_output": float(ui.max_plant_output.text()), # tốc độ giới hạn (rad/s)
            "min_output": float(ui.min_plant_output.text()), # tốc độ giới hạn (rad/s)
            "min_interval": float(ui.min_interval_input.text()),
            "max_interval": float(ui.max_interval_input.text()),
            "sampling_time": workspace.dt,
            "patience": 10,
            "use_validation": ui.use_validation_checkbox.isChecked(),
            "use_test_data": ui.use_test_checkbox.isChecked(),
        }

    def collect_train_parameters(self):
        p = self.collect_common_parameters()
        return p

    # ---------------- Actions ----------------
    def run_model_train(self):
        if not self.all_valid():
            QtWidgets.QMessageBox.warning(self, "Invalid input", "Please fill all fields correctly.")
            return

        if getattr(self, "train_win", None) is None:
            try:
                epochs = int(self.ui.training_epochs_input.text())
            except ValueError:
                epochs = 100
            self.train_win = ModelTrainWindow(parent=self, epoch_total=epochs)
            self.destroyed.connect(self.train_win.close)
            self.train_win.destroyed.connect(lambda: setattr(self, "train_win", None))

        self.train_win.show()
        self.train_win.loss_win.show()
        geo_loss = self.train_win.loss_win.geometry()
        self.train_win.loss_win.move(geo_loss.x()+600, geo_loss.y())

    def run_generate_data(self):
        if not self.all_valid_no_epoch():
            QtWidgets.QMessageBox.warning(self, "Invalid input", "Please fill all fields (epochs optional).")
            return
        
        workspace.plant["num_disc"], workspace.plant["den_disc"] = utils.discretize_tf(
            workspace.plant["num_cont"], workspace.plant["den_cont"], workspace.dt
        )
        print("num disc:", workspace.plant["num_disc"])
        print("den disc:", workspace.plant["den_disc"])
        u_hist_list, y_hist_list = [0]*len(workspace.plant["num_disc"]), [0]*(len(workspace.plant["den_disc"])-1)
        data_samples = int(self.ui.training_samples_input.text())
        t = np.linspace(0, (data_samples-1) * workspace.dt, data_samples)
        u = utils.generate_random_control_signal_sequence(
            float(self.ui.min_plant_input.text()), float(self.ui.max_plant_input.text()),
            float(self.ui.min_interval_input.text()), float(self.ui.max_interval_input.text()),
            t
        )
        y = np.zeros(data_samples)
        for i in range(data_samples):
            y[i] = utils.plant_response(
                workspace.plant["num_disc"], workspace.plant["den_disc"],
                u_hist_list, y_hist_list
            )

            if y[i] > float(self.ui.max_plant_output.text()):
                y[i] = float(self.ui.max_plant_output.text())
            elif y[i] < float(self.ui.min_plant_output.text()):
                y[i] = float(self.ui.min_plant_output.text())

            u_hist_list = [u[i]] + u_hist_list[:-1]
            y_hist_list = [y[i]] + y_hist_list[:-1]

        accepted = self._show_dataset_preview_dialog(t, u, y)
        if accepted:
            workspace.dataset = {"t": t, "u": u, "y": y}
            QtWidgets.QMessageBox.information(self, "Saved", "Dataset accepted.")
            self.update_buttons_state()

    def handle_export_data(self):
        if not hasattr(workspace, "dataset") or workspace.dataset is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "No dataset available to export. Generate data first.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Dataset",
            "",
            "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            t = workspace.dataset["t"]
            u = workspace.dataset["u"]
            y = workspace.dataset["y"]

            data = np.column_stack((t, u, y))
            np.savetxt(path, data, delimiter=",", header="t,u,y", comments="")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Could not save dataset:\n{e}")
            return

        QtWidgets.QMessageBox.information(self, "Saved", "Dataset saved successfully.")

    def handle_import_data(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Dataset",
            "",
            "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            t = data[:, 0]
            u = data[:, 1]
            y = data[:, 2]

            workspace.dataset = {"t": t, "u": u, "y": y}

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Could not load dataset:\n{e}")
            return

        QtWidgets.QMessageBox.information(self, "Loaded", "Dataset loaded successfully.")
        self.update_buttons_state()

    def _show_dataset_preview_dialog(self, t, u, y):
        """Show dataset preview with Accept/Reject buttons. Return True if accepted."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Dataset Preview")
        layout = QtWidgets.QVBoxLayout(dlg)
        canvas = FigureCanvas(Figure(figsize=(8, 5)))
        layout.addWidget(canvas)

        fig = canvas.figure
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        fig.tight_layout(h_pad=1.0)

        ax1.plot(t, u, label="Control signal u(t)")
        ax1.set_ylabel("u")
        ax1.grid(True)

        ax2.plot(t, y, label="Plant output y(t)", color='orange')
        ax2.set_xlabel("t (s)")
        ax2.set_ylabel("y")
        ax2.grid(True)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch(1)
        btn_accept = QtWidgets.QPushButton("Accept dataset")
        btn_reject = QtWidgets.QPushButton("Reject dataset")
        btn_layout.addWidget(btn_accept)
        btn_layout.addWidget(btn_reject)
        btn_layout.addStretch(1)
        layout.addLayout(btn_layout)

        btn_accept.clicked.connect(dlg.accept)
        btn_reject.clicked.connect(dlg.reject)

        return dlg.exec_() == QtWidgets.QDialog.Accepted

    def open_network_weight(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Network Weight File",
            "",
            "Pytorch Files (*.pth)"
        )
        if not path:
            return

        try:
            state_dict = torch.load(path, map_location="cpu")
            # Validate shapes before loading to avoid noisy torch traceback
            f_state = state_dict.get("f")
            g_state = state_dict.get("g")
            if not f_state or not g_state:
                raise ValueError("Missing 'f' or 'g' keys in checkpoint.")

            def _check_shapes(target_module, incoming_state):
                for name, param in incoming_state.items():
                    if name not in target_module.state_dict():
                        raise ValueError(f"Unexpected key '{name}' in checkpoint.")
                    tgt = target_module.state_dict()[name]
                    if tgt.shape != param.shape:
                        raise ValueError(
                            f"Shape mismatch for {name}: checkpoint {tuple(param.shape)} vs model {tuple(tgt.shape)}"
                        )

            _check_shapes(workspace.narma_model.f, f_state)
            _check_shapes(workspace.narma_model.g, g_state)

            workspace.narma_model.f.load_state_dict(f_state)
            workspace.narma_model.g.load_state_dict(g_state)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Failed to load weights:\n{e}")
            return

        QtWidgets.QMessageBox.information(self, "Loaded", "Weights loaded successfully.")

    def _validate_weight_file(self, cfg):
        # required meta
        for key in ["ny", "nu", "hidden", "f", "g"]:
            if key not in cfg:
                return False, f"Missing field: {key}"
        # compare with UI inputs
        try:
            ny = int(self.ui.delayed_outputs_input.text())
            nu = int(self.ui.delayed_inputs_input.text())
            hidden = int(self.ui.hidden_layers_input.text())
        except ValueError:
            return False, "Invalid architecture fields in UI."
        if cfg["ny"] != ny or cfg["nu"] != nu or cfg["hidden"] != hidden:
            return False, "Weight file architecture does not match current settings."

        def _check_block(block, name):
            for key in ["w1", "b1", "w2", "b2"]:
                if key not in block:
                    return False, f"Missing {name}.{key}"
            w1 = torch.tensor(block["w1"])
            b1 = torch.tensor(block["b1"])
            w2 = torch.tensor(block["w2"])
            b2 = torch.tensor(block["b2"]).reshape(-1)
            if w1.shape != (hidden, ny + nu):
                return False, f"{name}.w1 shape must be ({hidden}, {ny+nu})"
            if b1.shape != (hidden,):
                return False, f"{name}.b1 shape must be ({hidden},)"
            if w2.shape != (1, hidden):
                return False, f"{name}.w2 shape must be (1, {hidden})"
            if b2.shape != (1,):
                return False, f"{name}.b2 shape must be (1,)"
            return True, (w1, b1, w2, b2)

        ok_f, msg_or_tensors_f = _check_block(cfg["f"], "f")
        if not ok_f:
            return False, msg_or_tensors_f
        ok_g, msg_or_tensors_g = _check_block(cfg["g"], "g")
        if not ok_g:
            return False, msg_or_tensors_g

        cfg["_parsed_f"] = msg_or_tensors_f
        cfg["_parsed_g"] = msg_or_tensors_g
        return True, ""

    def _store_weight_file(self, cfg):
        # Store structured weights for later loading
        workspace.narma_weights = {
            "ny": cfg.get("ny"),
            "nu": cfg.get("nu"),
            "hidden": cfg.get("hidden"),
            "f": cfg.get("f", {}),
            "g": cfg.get("g", {}),
        }

    def handle_save_config(self):
        if not self.all_valid():
            QtWidgets.QMessageBox.warning(self, "Invalid input")
            return
        config = self.collect_train_parameters()
        workspace.narma_config = config

        try:
            # Create a new NARMA-L2 controller based on the saved config
            workspace.narma_model = NARMA_L2_Controller(
                ny=config["ny"],
                nu=config["nu"],
                hidden=config["hidden_size"],
                epsilon=config["learning_rate"],
                max_control=config["max_control"],
                min_control=config["min_control"],
                max_output=config["max_output"],
                min_output=config["min_output"],
                epochs=config["training_epochs"],
                lr=config["learning_rate"],
                batch_size=config["backprop_batch_size"],
                patience=config["patience"],
                default_model=False
            )
            QtWidgets.QMessageBox.information(self, "Config Saved", "Config saved and new NARMA model created.")
            workspace.first_save_config = True
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not create NARMA model:\n{e}")

    def handle_export_weight(self):
        if not hasattr(workspace, "narma_model") or workspace.narma_model is None:
            QtWidgets.QMessageBox.warning(self, "No Model", "No trained model available to export.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Network Weights",
            "",
            "PyTorch Files (*.pth)"
        )
        if not path:
            return

        try:
            torch.save({ "f": workspace.narma_model.f.state_dict(), "g": workspace.narma_model.g.state_dict() }, path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Could not export weights:\n{e}")
            return

        QtWidgets.QMessageBox.information(self, "Exported", "Network weights exported successfully.")

    def handle_save_model(self):
        if workspace.narma_model is None or workspace.temp_narma_model is None:
            QtWidgets.QMessageBox.warning(self, "No Model", "No trained model available to save.")
            return
        else:
            workspace.narma_model = copy.deepcopy(workspace.temp_narma_model)
            QtWidgets.QMessageBox.information(self, "Model Saved", "Trained model saved to workspace.")

    def set_default_parameters(self):
        default_params = workspace.get_default_narma_l2_params()
        ui = self.ui
        ui.hidden_layers_input.setText(str(default_params["hidden_size"]))
        ui.delayed_inputs_input.setText(str(default_params["nu"]))
        ui.delayed_outputs_input.setText(str(default_params["ny"]))
        ui.training_samples_input.setText(str(default_params.get("training_sample_size")))
        ui.max_plant_input.setText(str(default_params.get("max_control")))
        ui.min_plant_input.setText(str(default_params.get("min_control")))
        ui.min_interval_input.setText(str(default_params.get("min_interval")))
        ui.max_plant_output.setText(str(default_params.get("max_output")))
        ui.min_plant_output.setText(str(default_params.get("min_output")))
        ui.max_interval_input.setText(str(default_params.get("max_interval")))
        ui.training_epochs_input.setText(str(default_params.get("training_epochs")))
        ui.use_validation_checkbox.setChecked(default_params.get("use_validation", True))
        ui.use_test_checkbox.setChecked(default_params.get("use_test_data", True))
        workspace.set_default_narma_l2_model()
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
            "max_int_l": self.ui.min_interval_input,
            "min_in_r": self.ui.max_plant_output,
            "max_in_r": self.ui.min_plant_output,
            "min_int_r": self.ui.max_interval_input,
            "epochs": self.ui.training_epochs_input,
        }
        for key, widget in mapping.items():
            if key in cfg:
                widget.setText(str(cfg[key]))
        if "use_val" in cfg:
            self.ui.use_validation_checkbox.setChecked(bool(cfg.get("use_val")))
        if "use_test" in cfg:
            self.ui.use_test_checkbox.setChecked(bool(cfg.get("use_test")))

    # ---------------- Permissions ----------------
    def _apply_role_permissions(self):
        """
        Non-admin users can adjust training data/parameters but cannot manually edit network architecture;
        they can still apply defaults to populate those fields.
        """
        if self.current_role == "admin":
            return

        for w in [
            self.ui.hidden_layers_input,
            self.ui.delayed_inputs_input,
            self.ui.delayed_outputs_input,
        ]:
            w.setReadOnly(True)
            w.setEnabled(True)
            w.setStyleSheet("background-color: #b3b1b1;")
        # Keep default button enabled so they can load default architecture
        # self.ui.default_btn.setEnabled(True)

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
