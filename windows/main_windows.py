from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QWidget
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.main_ui import Ui_MainWindow as Ui_Main
from windows.input_window import InputWindow
from windows.plant_model_windows import PlantModelWindow, PlantModelDefault
from windows.output_graph_windows import OutputGraphWindow, generate_motor_data
from windows.user_guide_window import UserGuideWindow
from windows.model_config_window import ModelConfigWindow


class MainApp(QtWidgets.QMainWindow, Ui_Main):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Main Window")

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

        # === tạo timer để giả lập dữ liệu realtime ===
        self.sim_timer = QtCore.QTimer()
        self.sim_timer.timeout.connect(self.feed_sim_data)
        self.sim_index = 0

        # buffer dữ liệu giả lập
        self.output_t, self.output_r, self.output_y, self.output_y_pred, self.output_u = generate_motor_data(time_end=12)

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
        self.input_window = InputWindow(parent=self)
        self.input_window.main_window_ref = self
        self.input_window.show()

    def open_dc_motor_window(self):
        default_model = PlantModelDefault()
        self.dc_motor_window = PlantModelWindow(parent=self, default_model=default_model)
        self.dc_motor_window.show()

    def open_output_window(self):
        self.output_window.show()

    def open_user_guide_window(self):
        self.user_guide_window = UserGuideWindow(parent=self)
        self.user_guide_window.show()

    def open_ann_controller_window(self):
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
            # run_time = float(self.Run_time_input.text())
            run_time = 600.0  # Giả lập chạy 600 giây
            if run_time <= 0:
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter a valid positive run time (seconds).")
            self.running = False
            self.update_run_button()
            return
        
        # Reset chỉ số mô phỏng
        # self.sim_index = 0

        # Bắt đầu timer mô phỏng (mỗi 50 ms)
        self.sim_timer.start(50)

        # === Thêm timer dừng mô phỏng sau run_time giây ===
        self.stop_timer = QtCore.QTimer(self)
        self.stop_timer.setSingleShot(True)
        self.stop_timer.timeout.connect(self.stop_simulation)
        self.stop_timer.start(int(run_time * 1000))  # đổi giây -> mili giây

        print(f"[INFO] Simulation started for {run_time} seconds.")
    
    def stop_simulation(self):
        """Dừng mô phỏng"""
        self.sim_timer.stop()
        if hasattr(self, "stop_timer"):
            self.stop_timer.stop()
        self.running = False
        self.update_run_button()
        print("[INFO] Simulation stopped.")

    def restart_simulation(self):
        """Reset mô phỏng về trạng thái ban đầu"""
        # Dừng tất cả các timer đang chạy
        self.sim_timer.stop()
        if hasattr(self, "stop_timer"):
            self.stop_timer.stop()

        # Reset biến mô phỏng
        self.sim_index = 0
        self.running = False

        # Xóa đồ thị nếu có hàm clear
        if hasattr(self.output_window, "clear_graph"):
            self.output_window.clear_graph()

        # Reset lại nút Run
        self.update_run_button()

        # In log
        print("[INFO] Simulation restarted (reset).")

    def feed_sim_data(self):
        if self.sim_index < len(self.output_t):
            t = self.output_t[self.sim_index]
            r = self.output_r[self.sim_index]
            y = self.output_y[self.sim_index]
            y_pred = self.output_y_pred[self.sim_index]
            u = self.output_u[self.sim_index]

            # Gửi dữ liệu vào OutputGraphWindow
            self.output_window.append_data(t, r, y, y_pred, u)
            self.sim_index += 1
        else:
            self.stop_simulation()
            self.running = False
            self.update_run_button()

    def update_run_button(self):
        """Cập nhật giao diện nút Run (Start/Stop)"""
        if self.running:
            # self.Run_btn.setText("Stop")
            self.Run_btn.setIcon(self.icon_stop)
        else:
            # self.Run_btn.setText("Start")
            self.Run_btn.setIcon(self.icon_start)


# ---------------- Main ----------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_win = MainApp()
    main_win.show()
    sys.exit(app.exec_())
