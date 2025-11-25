# windows/plant_model_window.py

from PyQt5.QtWidgets import QDialog, QApplication
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from io import BytesIO
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
from ui.plant_model_ui import Ui_plant_model
from PyQt5.QtWidgets import QMessageBox
from backend.system_workspace import workspace

class PlantModelDefault:
    """Chứa giá trị mặc định và các tính toán plant"""
    def get_default_dc_motor_params(self):
        return {
            'L': 0.5,
            'R': 2,
            'Kb': 0.1,
            'Km': 0.1,
            'Kf': 0.2,
            'J': 0.02,
            'Td': 0.01
        }

class PlantModelWindow(QDialog):
    def __init__(self, parent=None, default_model=None):
        super().__init__(parent)

        # Load UI
        self.ui = Ui_plant_model()
        self.ui.setupUi(self)

        # Load default model
        self.default_model = default_model
        self.ui.custom_mode_box.setDisabled(True)

        # Backend: signal-slot
        # default checkbox
        self.ui.default_box.stateChanged.connect(self.on_default_checked)
        self.ui.custom_box.stateChanged.connect(self.on_custom_checked)
        self.ui.apply_button.clicked.connect(self.handle_apply)
        self.ui.close_button.clicked.connect(self.close)


    def parse_coeff_list(self, text):
        """
        Chuyển string nhập vào thành list float
        Hỗ trợ:
            - '[5, 10]'
            - '5,10'
            - '5 10'
        """
        # Xóa dấu [ ] nếu có
        text = text.strip().replace('[', '').replace(']', '')
        # Thay , hoặc space bằng space duy nhất
        text = text.replace(',', ' ')
        # Tách chuỗi thành list
        items = text.split()
        # Convert sang float
        return [float(x) for x in items]

    def tf_to_png(self):
        """Tạo PNG từ num/den và trả về QPixmap để show lên QLabel"""
        # Chuyển list thành chuỗi polynomial dạng LaTeX
        def list_to_poly(coeffs):
            terms = []
            n = len(coeffs)
            for i, c in enumerate(coeffs):
                if c == 0:
                    continue
                power = n - i - 1
                term = f"{c:.3g}"  # 3 chữ số thập phân
                if power > 0:
                    term += f"s"
                    if power > 1:
                        term += f"^{power}"
                terms.append(term)
            return " + ".join(terms) if terms else "0"

        num_str = list_to_poly(self.num_list)
        den_str = list_to_poly(self.den_list)

        latex_str = r"$G(s) = \frac{%s}{%s}$" % (num_str, den_str)

        # Vẽ hình bằng matplotlib
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

            # ===================== Update backend workspace =====================
            workspace.plant["num"] = self.num_list
            workspace.plant["den"] = self.den_list
            print(f"[INFO] Workspace plant updated: num={self.num_list}, den={self.den_list}")
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values.")

    def on_default_checked(self, state):
        is_checked = state == 2  # Qt.Checked == 2
        self.ui.dc_motor_box.setDisabled(is_checked)
        self.ui.custom_box.setDisabled(is_checked)

        if is_checked and self.default_model:
            # Lấy giá trị mặc định từ backend model
            defaults = self.default_model.get_default_dc_motor_params()
            # Gán vào các line_edit
            self.ui.motor_L.setText(str(defaults['L']))
            self.ui.motor_R.setText(str(defaults['R']))
            self.ui.motor_Kb.setText(str(defaults['Kb']))
            self.ui.motor_Km.setText(str(defaults['Km']))
            self.ui.motor_Kf.setText(str(defaults['Kf']))
            self.ui.motor_J.setText(str(defaults['J']))
            self.ui.motor_Td.setText(str(defaults['Td']))
    
    def on_custom_checked(self, state):
        is_checked = state == 2  # Qt.Checked == 2
        self.ui.dc_motor_box.setDisabled(is_checked)
        self.ui.default_box.setDisabled(is_checked)
        self.ui.custom_mode_box.setDisabled(not is_checked)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    default_model = PlantModelDefault()
    win = PlantModelWindow(default_model=default_model)
    win.show()
    sys.exit(app.exec_())
