import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.output_graph_ui import Ui_output_graph
import control as ctl
import pandas as pd
from pyqtgraph.exporters import ImageExporter


# ===================== 1. Sinh dữ liệu mô phỏng =====================
def generate_motor_data(time_end=15):
    R, L, Km, Kb, Kf, J = 2.0, 0.5, 0.1, 0.1, 0.2, 0.02
    num = [Km]
    den = [L * J, (L * Kf + R * J), (R * Kf + Km * Kb)]
    G = ctl.tf(num, den)

    # PID controller
    Kp, Ki, Kd = 50, 70, 1
    C = ctl.tf([Kd, Kp, Ki], [1, 0])
    sys_cl = ctl.feedback(C * G, 1)

    t = np.linspace(0, time_end, 2000)
    r = np.piecewise(
        t,
        [t < 2, (t >= 2) & (t < 5), (t >= 5) & (t < 7),
         (t >= 7) & (t < 10), t >= 10],
        [0.4, 0.6, 0.1, -0.2, 0]
    )

    t, y = ctl.forced_response(sys_cl, T=t, U=r)
    e = r - y
    dt = t[1] - t[0]
    u = Kp * e + Ki * np.cumsum(e) * dt + Kd * np.gradient(e, t)

    # noise prediction
    raw_noise = np.random.normal(0, 0.03 * np.max(np.abs(y)), len(y))
    noise = np.convolve(raw_noise, np.ones(20)/20, mode='same')
    y_pred = y + noise

    return t, r, y, y_pred, u


# ===================== 2. Hàm hỗ trợ tính metrics =====================
def compute_step_metrics(t, ref, y):
    """Tính rise time, settling time, overshoot dựa trên dữ liệu"""
    if len(y) < 2:
        return np.nan, np.nan, np.nan

    steady_state = ref[-1]
    y_final = y[-1]
    y_max = np.max(y)

    # overshoot %
    overshoot = 0
    if steady_state != 0:
        overshoot = ((y_max - steady_state) / abs(steady_state)) * 100

    # rise time (tại 10%–90% giá trị steady)
    try:
        y10 = 0.1 * steady_state
        y90 = 0.9 * steady_state
        t10 = t[np.where(y >= y10)[0][0]] if np.any(y >= y10) else np.nan
        t90 = t[np.where(y >= y90)[0][0]] if np.any(y >= y90) else np.nan
        rise_time = t90 - t10 if np.isfinite(t10) and np.isfinite(t90) else np.nan
    except Exception:
        rise_time = np.nan

    # settling time (±2%)
    try:
        tol = 0.02 * abs(steady_state)
        idx = np.where(np.abs(y - steady_state) > tol)[0]
        settling_time = t[idx[-1]] if len(idx) > 0 else 0
    except Exception:
        settling_time = np.nan

    return rise_time, settling_time, overshoot


def compute_error_metrics(ref, y, y_pred):
    """Tính RMSE, MAE, corr, bias"""
    if len(y) < 2:
        return np.nan, np.nan, np.nan, np.nan
    err_pred = y - y_pred
    RMSE = np.sqrt(np.mean(err_pred**2))
    MAE = np.mean(np.abs(err_pred))
    bias = np.mean(err_pred)
    corr = np.corrcoef(y, y_pred)[0, 1] if np.std(y_pred) > 0 else np.nan
    return RMSE, MAE, corr, bias


# ===================== 3. GUI hiển thị realtime =====================
class OutputGraphWindow(QtWidgets.QWidget, Ui_output_graph):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self._init_graphs()

        # buffer dữ liệu
        self.data_t, self.data_ref, self.data_y, self.data_pred, self.data_u = [], [], [], [], []
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_graph)
        self.timer.start(50)  # update mỗi 50ms

        # để tránh vẽ liên tục quá nhanh
        self.last_draw_len = 0

        # Button actions
        self.close_button.clicked.connect(self.close)
        self.save_graph_button.clicked.connect(self.save_graph_as)

    def _init_graphs(self):
        # Tracking
        layout1 = QtWidgets.QVBoxLayout(self.tracking_graph_wid)
        self.graph_tracking = pg.PlotWidget(background="#F0F0F0")  # đổi nền ở đây
        self.graph_tracking.addLegend()
        layout1.addWidget(self.graph_tracking)
        self.curve_ref = self.graph_tracking.plot(pen='black', name="Reference")
        self.curve_out = self.graph_tracking.plot(pen='r', name="Output")
        self.graph_tracking.showGrid(x=True, y=True, alpha=0.3)

        # Prediction
        layout2 = QtWidgets.QVBoxLayout(self.predict_graph_wid)
        self.graph_pred = pg.PlotWidget(background="#F0F0F0")  # đổi nền ở đây
        self.graph_pred.addLegend()
        layout2.addWidget(self.graph_pred)
        self.curve_pred = self.graph_pred.plot(pen='r', name="Predicted")
        self.curve_real = self.graph_pred.plot(pen='black', name="Actual")
        self.graph_pred.showGrid(x=True, y=True, alpha=0.3)

    def clear_graph(self):
        self.data_t.clear()
        self.data_ref.clear()
        self.data_y.clear()
        self.data_pred.clear()
        self.data_u.clear()
        self.curve_ref.clear()
        self.curve_out.clear()
        self.curve_pred.clear()
        self.curve_real.clear()
        self.last_draw_len = 0
        self.update_graph(restart=True)

    # ===== thêm hàm này =====
    def append_data(self, t, r, y, y_pred, u):
        """Nhận thêm 1 hoặc nhiều điểm dữ liệu"""
        if isinstance(t, (list, np.ndarray)):
            self.data_t.extend(t)
            self.data_ref.extend(r)
            self.data_y.extend(y)
            self.data_pred.extend(y_pred)
            self.data_u.extend(u)
        else:
            self.data_t.append(t)
            self.data_ref.append(r)
            self.data_y.append(y)
            self.data_pred.append(y_pred)
            self.data_u.append(u)

    def update_graph(self, restart=False):
        n = len(self.data_t)
        if (n == 0 or n == self.last_draw_len) and not restart:
            return  # chưa có dữ liệu mới

        # cập nhật vẽ
        self.curve_ref.setData(self.data_t, self.data_ref)
        self.curve_out.setData(self.data_t, self.data_y)
        self.curve_pred.setData(self.data_t, self.data_pred)
        self.curve_real.setData(self.data_t, self.data_y)
        self.last_draw_len = n

        # cập nhật metrics
        t = np.array(self.data_t)
        r = np.array(self.data_ref)
        y = np.array(self.data_y)
        y_pred = np.array(self.data_pred)
        u = np.array(self.data_u)

        rt, st, os = compute_step_metrics(t, r, y)
        RMSE, MAE, corr, bias = compute_error_metrics(r, y, y_pred)
        err = abs(r[-1] - y[-1]) if len(r) > 0 else np.nan

        # cập nhật text
        self.m_rise_time.setText(f"{rt:.3f}" if not np.isnan(rt) else "-")
        self.m_settlle_time.setText(f"{st:.3f}" if not np.isnan(st) else "-")
        self.m_overshoot.setText(f"{os:.2f}" if not np.isnan(os) else "-")
        self.m_control_err.setText(f"{err:.3f}" if not np.isnan(err) else "-")
        self.m_control_input.setText(f"{u[-1]:.3f}" if len(u) > 0 else "-")
        self.RMSE.setText(f"{RMSE:.4f}")
        self.MAE.setText(f"{MAE:.4f}")
        self.corr.setText(f"{corr:.3f}" if not np.isnan(corr) else "-")
        self.bias.setText(f"{bias:.4f}")

    def save_graph_as(self):
        # Lưu graph tracking
        filename1, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Tracking Graph", "", "PNG Files (*.png)")
        if filename1:
            exporter1 = ImageExporter(self.graph_tracking.plotItem)
            exporter1.parameters()['width'] = 1200
            exporter1.export(filename1)

        # Lưu graph prediction
        filename2, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Prediction Graph", "", "PNG Files (*.png)")
        if filename2:
            exporter2 = ImageExporter(self.graph_pred.plotItem)
            exporter2.parameters()['width'] = 1200
            exporter2.export(filename2)

        # Lưu metrics CSV
        metrics = {
            "Time": self.data_t,
            "Reference": self.data_ref,
            "Output": self.data_y,
            "Predicted": self.data_pred,
            "Control_input": self.data_u
        }
        df = pd.DataFrame(metrics)
        csv_fname = os.path.splitext(filename1)[0]+".csv"
        df.to_csv(csv_fname, index=False)
        QtWidgets.QMessageBox.information(self, "Saved", f"Graph saved as {filename1} and {filename2}\nMetrics saved as {csv_fname}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = OutputGraphWindow()
    w.show()
    # Lấy dữ liệu từ generate_motor_data
    t_arr, r_arr, y_arr, y_pred_arr, u_arr = generate_motor_data(time_end=15)
    data_len = len(t_arr)
    index = 0  # index hiện tại

    # Giả lập timer gửi dữ liệu từng bước
    def feed_data_from_array():
        global index
        if index < data_len:
            w.append_data(t_arr[index], r_arr[index], y_arr[index], y_pred_arr[index], u_arr[index])
            index += 1
        else:
            timer.stop()  # dừng khi hết dữ liệu

    timer = QtCore.QTimer()
    timer.timeout.connect(feed_data_from_array)
    timer.start(50)  # mỗi 50 ms gửi 1 sample

    sys.exit(app.exec_())
