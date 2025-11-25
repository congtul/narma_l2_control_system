from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QThread
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.main_ui import Ui_MainWindow as Ui_Main
from windows.input_window import InputWindow
from windows.plant_model_windows import PlantModelWindow, PlantModelDefault
from windows.output_graph_windows import OutputGraphWindow
from windows.user_guide_window import UserGuideWindow
from windows.model_config_window import ModelConfigWindow
from backend.simulation_worker import SimulationWorker
from windows.login_window import LoginDialog


class MainApp(QtWidgets.QMainWindow, Ui_Main):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Main Window")
        self.setFixedSize(self.size())

        self._bg_pix = QtGui.QPixmap(".\\resources\\images\\2.jpg")
        self._update_background()

        # Login / Logout state
        self.current_user = None  # {"username": str, "role": "admin|user"}
        self.login_btn = QtWidgets.QPushButton("Login", self)
        self.login_btn.setGeometry(20, 50, 120, 30)  # under "Main UI"
        self.login_btn.setStyleSheet("background-color: #ffffff; border: 1px solid #999; border-radius: 6px;")
        self.login_btn.clicked.connect(self.handle_login_logout)
        self._user_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources", "users.json"))

        # === Thêm nút Restart nếu chưa có trong UI ===
        if not hasattr(self, "Restart_btn"):
            self.Restart_btn = QtWidgets.QPushButton("Restart", self)
            self.Restart_btn.setGeometry(self.Run_btn.geometry().x() + 120, self.Run_btn.geometry().y(),
                                         100, self.Run_btn.geometry().height())
            self.Restart_btn.setStyleSheet("background-color: orange; color: white; font-weight: bold; border-radius: 5px;")
            self.Restart_btn.show()

        # === Các nút cần filter double click ===
        for btn in [
            self.input_btn,
            self.ANN_controller_btn,
            self.DC_motor_btn,
            self.Output_btn,
            self.Run_btn,
            self.User_guide_btn,
        ]:
            btn.installEventFilter(self)

        # === State control ===
        self.running = False

        # === buffer dữ liệu cho output graph ===
        self.output_data_ready = False
        self.output_t = []
        self.output_r = []
        self.output_y = []
        self.output_y_pred = []
        self.output_u = []

        # === tạo OutputGraphWindow ngay nhưng không show ===
        self.output_window = OutputGraphWindow()
        self.output_window.hide()

        # === Connect signals ===
        self.Run_btn.clicked.connect(self.toggle_run)
        self.Restart_btn.clicked.connect(self.restart_simulation)

        # === Setup icon / style ===
        self.icon_start = QtGui.QIcon(".\\resources\\images\\start_button.jpg")
        self.icon_stop = QtGui.QIcon(".\\resources\\images\\stop_button.jpg")
        self.Restart_btn.setIcon(QtGui.QIcon(".\\resources\\images\\restart_button.png"))
        self.Run_btn.setIconSize(QtCore.QSize(50, 50))
        self.Restart_btn.setIconSize(QtCore.QSize(50, 50))

        self.update_run_button()
        self.update_login_button()

    # ---------------- Event Filter ----------------
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonDblClick and event.button() == QtCore.Qt.LeftButton:
            if obj == self.input_btn:
                self.open_input_window()
            elif obj == self.ANN_controller_btn:
                self.open_ann_controller_window()
            elif obj == self.DC_motor_btn:
                self.open_dc_motor_window()
            elif obj == self.Output_btn:
                self.open_output_window()
            elif obj == self.User_guide_btn:
                self.open_user_guide_window()
            return True
        return super().eventFilter(obj, event)

    # ---------------- Open Windows ----------------
    def open_input_window(self):
        if not self._ensure_access():
            return
        self.input_window = InputWindow(parent=self)
        self.input_window.main_window_ref = self
        self.input_window.show()

    def open_dc_motor_window(self):
        if not self._ensure_access():
            return
        default_model = PlantModelDefault()
        self.dc_motor_window = PlantModelWindow(parent=self, default_model=default_model)
        self.dc_motor_window.show()

    def open_output_window(self):
        if not self._ensure_access():
            return
        self.output_window.show()

    def open_user_guide_window(self):
        if not self._ensure_access():
            return
        self.user_guide_window = UserGuideWindow(parent=self)
        self.user_guide_window.show()

    def open_ann_controller_window(self):
        if not self._ensure_access(require_admin=True):
            return
        self.model_config_window = ModelConfigWindow(parent=self)
        self.model_config_window.show()

    def closeEvent(self, event):
        for w in self.findChildren(QWidget):
            try:
                w.close()
            except:
                pass
        event.accept()

    # ---------------- Simulation Control ----------------
    def toggle_run(self):
        """Toggle giữa Start / Stop"""
        self.running = not self.running
        if self.running:
            self.start_simulation()
        else:
            self.stop_simulation()
        self.update_run_button()

    def start_simulation(self):
        """Bắt đầu giả lập dữ liệu real-time cho OutputGraphWindow"""
        # Lấy thời gian chạy từ QLineEdit
        try:
            run_time = 600.0
            if run_time <= 0: raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter a valid positive run time.")
            self.running = False
            self.update_run_button()
            return

        # === Tạo thread + worker ===
        self.sim_thread = QThread()
        self.sim_worker = SimulationWorker(dt=0.05)
        self.sim_worker.moveToThread(self.sim_thread)

        # Signal-slot
        self.sim_thread.started.connect(self.sim_worker.run)
        self.sim_worker.data_ready.connect(self.output_window.append_data)
        self.sim_worker.finished.connect(self.sim_thread.quit)

        self.sim_thread.start()

        # === Timer dừng theo run_time ===
        self.stop_timer = QtCore.QTimer(self)
        self.stop_timer.setSingleShot(True)
        self.stop_timer.timeout.connect(self.stop_simulation)
        self.stop_timer.start(int(run_time * 1000))

        print(f"[INFO] Simulation started for {run_time} seconds.")
    
    def stop_simulation(self):
        """Dừng mô phỏng"""
        if hasattr(self, "sim_worker"):
            self.sim_worker.stop()  
        self.running = False
        self.update_run_button()
        print("[INFO] Simulation stopped.")

    def restart_simulation(self):
        """Reset mô phỏng về trạng thái ban đầu"""
        # Dừng tất cả các timer đang chạy
        self.stop_simulation()

        if hasattr(self, "sim_worker"):
            self.sim_worker.reset()  # reset clock

        # Xóa đồ thị nếu có hàm clear
        if hasattr(self.output_window, "clear_graph"):
            self.output_window.clear_graph()

        self.running = False
        # Reset lại nút Run
        self.update_run_button()

        # In log
        print("[INFO] Simulation restarted (reset).")

    def update_run_button(self):
        """Cập nhật giao diện nút Run (Start/Stop)"""
        if self.running:
            # self.Run_btn.setText("Stop")
            self.Run_btn.setIcon(self.icon_stop)
        else:
            # self.Run_btn.setText("Start")
            self.Run_btn.setIcon(self.icon_start)

    # ---------------- Login / Logout ----------------
    def handle_login_logout(self):
        if self.current_user:
            self.current_user = None
            QtWidgets.QMessageBox.information(self, "Logged out", "You have been logged out.")
        else:
            dlg = LoginDialog(self, db_path=self._user_db_path)
            if dlg.exec_() == QtWidgets.QDialog.Accepted:
                username, role = dlg.get_result()
                self.current_user = {"username": username, "role": role}
                QtWidgets.QMessageBox.information(self, "Logged in", f"Welcome {username} ({role}).")
        self.update_login_button()

    def update_login_button(self):
        if self.current_user:
            self.login_btn.setText(f"Logout ({self.current_user['role']})")
        else:
            self.login_btn.setText("Login")

    def _ensure_access(self, require_admin=False):
        if not self.current_user:
            QtWidgets.QMessageBox.warning(self, "Login required", "Please login to access this feature.")
            return False
        if require_admin and self.current_user.get("role") != "admin":
            QtWidgets.QMessageBox.critical(self, "Access denied", "Only admin users can access this feature.")
            return False
        return True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_background()

    def _update_background(self):
        if hasattr(self, "_bg_pix") and not self._bg_pix.isNull():
            scaled = self._bg_pix.scaled(
                self.main_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.main_label.setScaledContents(False)
            self.main_label.setPixmap(scaled)


# ---------------- Main ----------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_win = MainApp()
    main_win.show()
    sys.exit(app.exec_())
