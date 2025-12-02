from PyQt5 import QtWidgets, QtGui


class OnlineTrainingConfigDialog(QtWidgets.QDialog):
    """
    Collects online training configuration before starting simulation.
    Fields: model, learning rate, batch size, epoch count.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Online Training Config")
        self.setModal(True)
        self._config = {"training_online": True}

        # Inputs
        self.model_edit = QtWidgets.QLineEdit(self)
        self.model_edit.setPlaceholderText("Model name or path")

        self.lr_edit = QtWidgets.QLineEdit(self)
        self.lr_edit.setValidator(QtGui.QDoubleValidator(bottom=0.0))
        self.lr_edit.setPlaceholderText("Learning rate (e.g. 5e-5)")

        int_validator = QtGui.QIntValidator(1, 1_000_000, self)
        self.batch_edit = QtWidgets.QLineEdit(self)
        self.batch_edit.setValidator(int_validator)
        self.batch_edit.setPlaceholderText("Batch size (e.g. 5)")

        self.epoch_edit = QtWidgets.QLineEdit(self)
        self.epoch_edit.setValidator(int_validator)
        self.epoch_edit.setPlaceholderText("Epochs per batch (e.g. 2)")

        # Buttons
        self.start_btn = QtWidgets.QPushButton("Start Train Online", self)
        self.start_btn.setEnabled(False)
        self.offline_btn = QtWidgets.QPushButton("Train Offline", self)

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow("Model:", self.model_edit)
        form.addRow("Learning rate:", self.lr_edit)
        form.addRow("Batch size:", self.batch_edit)
        form.addRow("Epoch:", self.epoch_edit)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(self.offline_btn)
        btn_row.addWidget(self.start_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(btn_row)

        # Signals
        for w in [self.model_edit, self.lr_edit, self.batch_edit, self.epoch_edit]:
            w.textChanged.connect(self._update_start_enabled)
        self.start_btn.clicked.connect(self._accept_online)
        self.offline_btn.clicked.connect(self._accept_offline)

    def _inputs_valid(self) -> bool:
        if not self.model_edit.text().strip():
            return False
        for widget in [self.lr_edit, self.batch_edit, self.epoch_edit]:
            text = widget.text().strip()
            if not text:
                return False
            validator = widget.validator()
            if validator:
                state, _, _ = validator.validate(text, 0)
                if state != QtGui.QValidator.Acceptable:
                    return False
        return True

    def _update_start_enabled(self):
        self.start_btn.setEnabled(self._inputs_valid())

    def _accept_online(self):
        if not self._inputs_valid():
            return
        self._config = {
            "training_online": True,
            "model": self.model_edit.text().strip(),
            "lr": float(self.lr_edit.text().strip()),
            "batch_size": int(self.batch_edit.text().strip()),
            "epoch": int(self.epoch_edit.text().strip()),
        }
        self.accept()

    def _accept_offline(self):
        # Offline training: disable online updates and close
        self._config = {
            "training_online": False,
            "model": self.model_edit.text().strip(),
            "lr": self.lr_edit.text().strip(),
            "batch_size": self.batch_edit.text().strip(),
            "epoch": self.epoch_edit.text().strip(),
        }
        self.accept()

    def get_config(self):
        return self._config
