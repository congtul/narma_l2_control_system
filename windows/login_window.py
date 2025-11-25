import json
import os
from PyQt5 import QtWidgets


class LoginDialog(QtWidgets.QDialog):
    """Simple login dialog with username/password validated against file DB."""

    def __init__(self, parent=None, db_path=None):
        super().__init__(parent)
        self.setWindowTitle("Login")
        self.setModal(True)
        self.db_path = db_path
        self._db = self._load_db()
        self.username_edit = QtWidgets.QLineEdit(self)
        self.username_edit.setPlaceholderText("Enter username")
        self.password_edit = QtWidgets.QLineEdit(self)
        self.password_edit.setPlaceholderText("Enter password")
        self.password_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self._selected_role = None

        form = QtWidgets.QFormLayout()
        form.addRow("Username:", self.username_edit)
        form.addRow("Password:", self.password_edit)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        btn_box.accepted.connect(self._accept)
        btn_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(btn_box)

    def _accept(self):
        if not self.username_edit.text().strip():
            QtWidgets.QMessageBox.warning(self, "Missing info", "Please enter a username.")
            return
        if not self.password_edit.text():
            QtWidgets.QMessageBox.warning(self, "Missing info", "Please enter a password.")
            return
        username = self.username_edit.text().strip()
        password = self.password_edit.text()
        role = self._authenticate(username, password)
        if role is None:
            QtWidgets.QMessageBox.critical(self, "Invalid credentials", "Incorrect username or password.")
            return
        self._selected_role = role
        self.accept()

    def _load_db(self):
        if self.db_path is None:
            return []
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("users", [])
        except Exception:
            return []

    def _authenticate(self, username, password):
        for user in self._db:
            if user.get("username") == username and user.get("password") == password:
                return user.get("role")
        return None

    def get_result(self):
        return (
            self.username_edit.text().strip(),
            self._selected_role,
        )
