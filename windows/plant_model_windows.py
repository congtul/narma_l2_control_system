# windows/plant_model_window.py

from PyQt5.QtWidgets import QDialog, QApplication, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import os, sys
import matplotlib.pyplot as plt
from io import BytesIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.plant_model_ui import Ui_plant_model
from backend.system_workspace import workspace

class PlantModelWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Load UI
        self.ui = Ui_plant_model()
        self.ui.setupUi(self)

        # Default model
        self.ui.custom_mode_box.setDisabled(True)

        mode = workspace.plant.get("mode", "dc_motor")
        if mode == "default":
            self.ui.default_box.setChecked(True)
            self.on_default_checked(Qt.Checked)
        elif mode == "custom":
            self.ui.custom_box.setChecked(True)
            self.on_custom_checked(Qt.Checked)
            self.ui.num_custom_coeff.setText(workspace.plant.get("num_custom", ""))
            self.ui.den_custom_coeff.setText(workspace.plant.get("den_custom", ""))

        # Tạo num/den mặc định từ workspace nếu có
        self.num_list = workspace.plant.get("num_cont", [0.01])
        self.den_list = workspace.plant.get("den_cont", [0.005, 0.07, 0.2])
        # Hiển thị TF
        pixmap = self.tf_to_png()
        self.ui.transfer_function_res.setPixmap(pixmap)

        # Load motor params nếu có trong workspace
        plant_data = workspace.plant
        if all(k in plant_data for k in ["L","R","Kb","Km","Kf","J","Td"]):
            self.ui.motor_L.setText(str(plant_data["L"]))
            self.ui.motor_R.setText(str(plant_data["R"]))
            self.ui.motor_Kb.setText(str(plant_data["Kb"]))
            self.ui.motor_Km.setText(str(plant_data["Km"]))
            self.ui.motor_Kf.setText(str(plant_data["Kf"]))
            self.ui.motor_J.setText(str(plant_data["J"]))
            self.ui.motor_Td.setText(str(plant_data["Td"]))

        # Signal-slot
        self.ui.default_box.stateChanged.connect(self.on_default_checked)
        self.ui.custom_box.stateChanged.connect(self.on_custom_checked)
        self.ui.apply_button.clicked.connect(self.handle_apply)
        self.ui.save_button.clicked.connect(self.handle_save)
        self.ui.close_button.clicked.connect(self.close)

    # ---------------- Helpers ----------------
    def parse_coeff_list(self, text):
        """Chuyển string nhập vào thành list float"""
        text = text.strip().replace('[', '').replace(']', '')
        text = text.replace(',', ' ')
        items = text.split()
        return [float(x) for x in items]

    def tf_to_png(self):
        """Tạo PNG từ num/den và trả về QPixmap"""
        def list_to_poly(coeffs):
            terms = []
            n = len(coeffs)
            for i, c in enumerate(coeffs):
                if c == 0:
                    continue
                power = n - i - 1
                term = f"{c:.3g}"
                if power > 0:
                    term += "s"
                    if power > 1:
                        term += f"^{power}"
                terms.append(term)
            return " + ".join(terms) if terms else "0"

        num_str = list_to_poly(self.num_list)
        den_str = list_to_poly(self.den_list)
        latex_str = r"$G(s) = \frac{%s}{%s}$" % (num_str, den_str)

        fig, ax = plt.subplots(figsize=(4,1))
        ax.text(0.5, 0.5, latex_str, fontsize=14, ha='center', va='center')
        ax.axis('off')

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)

        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        return pixmap

    # ---------------- Handlers ----------------
    def handle_apply(self):
        try:
            if self.ui.custom_box.isChecked():
                self.num_list = self.parse_coeff_list(self.ui.num_custom_coeff.text())
                self.den_list = self.parse_coeff_list(self.ui.den_custom_coeff.text())
            else:
                L = float(self.ui.motor_L.text())
                R = float(self.ui.motor_R.text())
                Kb = float(self.ui.motor_Kb.text())
                Km = float(self.ui.motor_Km.text())
                Kf = float(self.ui.motor_Kf.text())
                J = float(self.ui.motor_J.text())
                Td = float(self.ui.motor_Td.text())

                self.num_list = [Km]
                self.den_list = [L*J, L*Kf + R*J, R*Kf + Km*Kb]

            pixmap = self.tf_to_png()
            self.ui.transfer_function_res.setPixmap(pixmap)
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values.")

    def handle_save(self):
        """Lưu vào workspace, an toàn với cả default/custom mode"""
        try:
            if self.ui.custom_box.isChecked():
                # lưu custom mode
                workspace.plant["mode"] = "custom"
                workspace.plant["num_custom"] = self.ui.num_custom_coeff.text()
                workspace.plant["den_custom"] = self.ui.den_custom_coeff.text()
            else:
                workspace.plant["L"] = float(self.ui.motor_L.text())
                workspace.plant["R"] = float(self.ui.motor_R.text())
                workspace.plant["Kb"] = float(self.ui.motor_Kb.text())
                workspace.plant["Km"] = float(self.ui.motor_Km.text())
                workspace.plant["Kf"] = float(self.ui.motor_Kf.text())
                workspace.plant["J"] = float(self.ui.motor_J.text())
                workspace.plant["Td"] = float(self.ui.motor_Td.text())
                
                workspace.plant["mode"] = "dc_motor"
                if self.ui.default_box.isChecked():
                    workspace.plant["mode"] = "default"

            # lưu TF liên tục luôn
            workspace.plant["num_cont"] = self.num_list
            workspace.plant["den_cont"] = self.den_list

            print(f"[INFO] Workspace plant updated: num={self.num_list}, den={self.den_list}")
        except ValueError:
            QMessageBox.warning(self, "Save Error", "Cannot save: invalid numeric values.")

    # ---------------- Checkboxes ----------------
    def on_default_checked(self, state):
        is_checked = state == Qt.Checked
        self.ui.dc_motor_box.setDisabled(is_checked)
        self.ui.custom_box.setDisabled(is_checked)

        if is_checked:
            defaults = workspace.get_default_dc_motor_params()
            self.ui.motor_L.setText(str(defaults['L']))
            self.ui.motor_R.setText(str(defaults['R']))
            self.ui.motor_Kb.setText(str(defaults['Kb']))
            self.ui.motor_Km.setText(str(defaults['Km']))
            self.ui.motor_Kf.setText(str(defaults['Kf']))
            self.ui.motor_J.setText(str(defaults['J']))
            self.ui.motor_Td.setText(str(defaults['Td']))

    def on_custom_checked(self, state):
        is_checked = state == Qt.Checked
        self.ui.dc_motor_box.setDisabled(is_checked)
        self.ui.default_box.setDisabled(is_checked)
        self.ui.custom_mode_box.setDisabled(not is_checked)


# ---------------- Main ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PlantModelWindow()
    win.show()
    sys.exit(app.exec_())
