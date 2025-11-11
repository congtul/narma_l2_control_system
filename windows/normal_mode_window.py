from PyQt5 import QtWidgets
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.normal_mode_ui import Ui_MainWindow as Ui_Normal
import pyqtgraph as pg
import numpy as np

class NormalModeWindow(QtWidgets.QMainWindow, Ui_Normal):
    saved_mode = None
    saved_random_values = {"max_vel": "", "min_vel": ""}
    saved_manual_values = {"stime": "", "veloval": ""}

    def __init__(self, parent_window=None):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Normal Mode Window")
        self.parent_window = parent_window
        self.active_mode = None
        self.graph_win = None  # Thêm biến instance cho graph

        restored = self.restore_previous_state()
        if not restored:
            # Disable all editable fields initially
            self.RD_max_vel_line_edit.setEnabled(False)
            self.RD_min_vel_line_edit.setEnabled(False)
            self.man_Stime_lineedit.setEnabled(False)
            self.man_Veloval_lineedit.setEnabled(False)

        # Connect buttons
        self.Random_gen_btn.clicked.connect(self.use_random_mode)
        self.Manual_gen_btn.clicked.connect(self.use_manual_mode)
        self.Reset_all_val_btn.clicked.connect(self.reset_all_values)
        self.Apply_normalmode_btn.clicked.connect(self.apply_normal_mode)
        self.OK_normalmode_btn_2.clicked.connect(self.on_ok_clicked)

        # Connect text change signals
        self.RD_max_vel_line_edit.textChanged.connect(self.update_apply_button_state)
        self.RD_min_vel_line_edit.textChanged.connect(self.update_apply_button_state)
        self.man_Stime_lineedit.textChanged.connect(self.update_apply_button_state)
        self.man_Veloval_lineedit.textChanged.connect(self.update_apply_button_state)

    def restore_previous_state(self):
        if NormalModeWindow.saved_mode == "random":
            self.active_mode = "random"
            self.use_random_mode()
            self.RD_max_vel_line_edit.setText(str(NormalModeWindow.saved_random_values["max_vel"]))
            self.RD_min_vel_line_edit.setText(str(NormalModeWindow.saved_random_values["min_vel"]))
            self.man_Stime_lineedit.setText(str(NormalModeWindow.saved_manual_values["stime"]))
            self.man_Veloval_lineedit.setText(str(NormalModeWindow.saved_manual_values["veloval"]))
            return True
        elif NormalModeWindow.saved_mode == "manual":
            self.active_mode = "manual"
            self.use_manual_mode()
            self.man_Stime_lineedit.setText(str(NormalModeWindow.saved_manual_values["stime"]))
            self.man_Veloval_lineedit.setText(str(NormalModeWindow.saved_manual_values["veloval"]))
            self.RD_max_vel_line_edit.setText(str(NormalModeWindow.saved_random_values["max_vel"]))
            self.RD_min_vel_line_edit.setText(str(NormalModeWindow.saved_random_values["min_vel"]))
            return True
        else:
            self.active_mode = None
            self.Apply_normalmode_btn.setEnabled(False)
            return False

    def use_random_mode(self):
        self.active_mode = "random"
        self.Status_nor_label.setText("Random mode is chosen!")
        self.Status_nor_label.setStyleSheet("color: #ffa23a; font-weight: bold;")
        self.RD_max_vel_line_edit.setEnabled(True)
        self.RD_min_vel_line_edit.setEnabled(True)
        self.man_Stime_lineedit.setEnabled(False)
        self.man_Veloval_lineedit.setEnabled(False)
        self.update_apply_button_state()

    def use_manual_mode(self):
        self.active_mode = "manual"
        self.Status_nor_label.setText("Manual mode is chosen!")
        self.Status_nor_label.setStyleSheet("color: blue; font-weight: bold;")
        self.man_Stime_lineedit.setEnabled(True)
        self.man_Veloval_lineedit.setEnabled(True)
        self.RD_max_vel_line_edit.setEnabled(False)
        self.RD_min_vel_line_edit.setEnabled(False)
        self.update_apply_button_state()

    def reset_all_values(self):
        self.active_mode = None
        self.RD_max_vel_line_edit.clear()
        self.RD_min_vel_line_edit.clear()
        self.man_Stime_lineedit.clear()
        self.man_Veloval_lineedit.clear()
        self.RD_max_vel_line_edit.setEnabled(False)
        self.RD_min_vel_line_edit.setEnabled(False)
        self.man_Stime_lineedit.setEnabled(False)
        self.man_Veloval_lineedit.setEnabled(False)
        self.Apply_normalmode_btn.setEnabled(False)
        self.Status_nor_label.setText("Reset all values")
        self.Status_nor_label.setStyleSheet("color: red; font-weight: bold;")

    def update_apply_button_state(self):
        if self.active_mode == "random":
            self.Apply_normalmode_btn.setEnabled(
                bool(self.RD_max_vel_line_edit.text().strip() and
                     self.RD_min_vel_line_edit.text().strip()))
        elif self.active_mode == "manual":
            self.Apply_normalmode_btn.setEnabled(
                bool(self.man_Stime_lineedit.text().strip() and
                     self.man_Veloval_lineedit.text().strip()))
        else:
            self.Apply_normalmode_btn.setEnabled(False)

    def parse_manual_input(self, text):
        # Loại bỏ dấu [] nếu có
        text = text.replace("[", "").replace("]", "")
        # Thay tất cả dấu , bằng space
        text = text.replace(",", " ")
        # Split theo space, loại bỏ rỗng
        items = text.split()
        # Convert thành float
        return [float(x) for x in items]

    def apply_normal_mode(self):
        t_max = 10  # Thời gian tối đa để hiển thị graph, có thể tối ưu sau
        t = np.linspace(0, t_max, 1000)

        if self.active_mode == "random":
            NormalModeWindow.saved_mode = "random"
            max_vel = float(self.RD_max_vel_line_edit.text())
            min_vel = float(self.RD_min_vel_line_edit.text())
            NormalModeWindow.saved_random_values = {
                "max_vel": max_vel,
                "min_vel": min_vel,
            }

            # Parameters
            num_steps = 10  # số đoạn step
            t_max = 10
            t = np.linspace(0, t_max, 1000)
            step_len = len(t) // num_steps

            # Tạo các giá trị ngẫu nhiên cho từng step
            y = np.zeros_like(t)
            rand_values = np.random.uniform(min_vel, max_vel, size=num_steps)
            for i in range(num_steps):
                start = i * step_len
                end = (i+1) * step_len if i < num_steps-1 else len(t)
                y[start:end] = rand_values[i]

            self.plot_graph(t, y, title="Random Generator Preview")

        elif self.active_mode == "manual":
            NormalModeWindow.saved_mode = "manual"
            step_time = self.parse_manual_input(self.man_Stime_lineedit.text())
            step_value = self.parse_manual_input(self.man_Veloval_lineedit.text())
            NormalModeWindow.saved_manual_values = {
                "stime": step_time,
                "veloval": step_value,
            }

            if step_time:
                #TODO: Cân nhắc thêm thời gian cuối nếu cần
                t_max = 15
                t = np.linspace(0, t_max, 1000)
            y = np.zeros_like(t)
            for i in range(len(step_time)-1):
                y[(t >= step_time[i]) & (t < step_time[i+1])] = step_value[i]
            if step_time:
                y[t >= step_time[-1]] = step_value[-1]

            self.plot_graph(t, y, title="Manual Generator Preview")

        self.Apply_normalmode_btn.setEnabled(False)
        if self.active_mode:
            self.Status_nor_label.setText(f"{self.active_mode.capitalize()} values applied successfully!")
            self.Status_nor_label.setStyleSheet("color: green; font-weight: bold;")

    def plot_graph(self, t, y, title="Preview"):
        if not self.graph_win or not self.graph_win.isVisible():
            # Tạo PlotWidget lần đầu hoặc khi cửa sổ bị đóng
            self.graph_win = pg.PlotWidget(title=title)
            self.graph_win.setWindowTitle(title)
            self.graph_win.setBackground("#f0f0f0")  # nền xám/trắng

            # Bắt sự kiện đóng cửa sổ
            self.graph_win.closeEvent = self.graph_close_event

            self.graph_win.show()
        else:
            # Clear graph nếu còn sống
            self.graph_win.clear()
            self.graph_win.setTitle(title)

        pen = pg.mkPen(color='k', width=2)  # đường màu đen
        self.graph_win.plot(t, y, pen=pen)

    def graph_close_event(self, event):
        # Khi user đóng cửa sổ graph, set graph_win = None để Apply có thể tạo lại
        self.graph_win = None
        event.accept()

    def on_ok_clicked(self):
        if self.Apply_normalmode_btn.isEnabled():
            self.apply_normal_mode()
        self.close()

    def closeEvent(self, event):
        if self.parent_window:
            self.parent_window.show()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = NormalModeWindow()
    win.show()
    sys.exit(app.exec_())
