from PyQt5 import QtWidgets, QtCore
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.main_ui import Ui_MainWindow as Ui_Main
from windows.input_window import InputWindow
from windows.plant_model_windows import PlantModelWindow, PlantModelDefault
from windows.output_graph_windows import OutputGraphWindow, generate_motor_data
from windows.user_guide_window import UserGuideWindow

class MainApp(QtWidgets.QMainWindow, Ui_Main):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Main Window")
        for btn in [
            self.input_btn,
            self.ANN_controller_btn,
            self.DC_motor_btn,
            self.Output_btn,
            self.Run_btn,
            self.User_guide_btn,
        ]:
            btn.installEventFilter(self)
        
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
        # just for mockup, will be replaced when in release
        self.output_t, self.output_r, self.output_y, self.output_y_pred, self.output_u = generate_motor_data(time_end=15)

        # === tạo OutputGraphWindow ngay nhưng không show ===
        self.output_window = OutputGraphWindow()
        self.output_window.hide()

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
            elif obj == self.Run_btn:
                self.run_simulation()
            elif obj == self.User_guide_btn:
                self.open_user_guide_window()
            return True
        return super().eventFilter(obj, event)

    def open_input_window(self):
        self.input_window = InputWindow()
        self.input_window.show()

    def open_dc_motor_window(self):
        default_model = PlantModelDefault()
        self.dc_motor_window = PlantModelWindow(parent=self, default_model=default_model)
        self.dc_motor_window.show()

    def open_output_window(self):
        self.output_window.show()

    def open_user_guide_window(self):
        self.user_guide_window = UserGuideWindow()
        self.user_guide_window.show()

    def open_ann_controller_window(self):
        # Placeholder for ANN controller window
        pass
    
    def run_simulation(self):
        """Bắt đầu giả lập dữ liệu real-time cho OutputGraphWindow"""
        self.sim_index = 0
        self.sim_timer.start(50)  # mỗi 50 ms gửi 1 sample

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
            self.sim_timer.stop()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main_win = MainApp()
    main_win.show()
    sys.exit(app.exec_())
