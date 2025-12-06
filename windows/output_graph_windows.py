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
def compute_tracking_metrics(ref, y, u):
    """General-purpose tracking metrics + control signal metrics"""
    ref = np.array(ref)
    y = np.array(y)
    u = np.array(u)

    e = ref - y

    RMSE = np.sqrt(np.mean(e**2))
    MAE = np.mean(np.abs(e))
    steady_err = np.abs(e[-1])

    # Control signal metrics
    mean_abs_u = np.mean(np.abs(u))
    rms_u = np.sqrt(np.mean(u**2))

    return RMSE, MAE, steady_err, mean_abs_u, rms_u

def compute_prediction_metrics(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    e = y - y_pred
    
    RMSE = np.sqrt(np.mean(e**2))
    MAE = np.mean(np.abs(e))
    bias = np.mean(e)
    corr = np.corrcoef(y, y_pred)[0, 1] if np.std(y_pred) > 1e-8 else np.nan

    return RMSE, MAE, corr, bias


# ===================== 3. GUI hiển thị realtime =====================
class OutputGraphWindow(QtWidgets.QWidget, Ui_output_graph):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self._init_graphs()
        self.scale_to_screen()

        # buffer dữ liệu
        self.data_t, self.data_ref, self.data_y, self.data_pred, self.data_u = [], [], [], [], []
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_graph)
        self.timer.start(50)  # update mỗi 50ms

        self.n = 0
        self.sum_e2 = 0.0
        self.sum_abs_e = 0.0

        self.sum_abs_u = 0.0
        self.sum_u2 = 0.0

        # prediction metrics
        self.sum_ep2 = 0.0
        self.sum_abs_ep = 0.0
        self.sum_bias = 0.0

        # để tránh vẽ liên tục quá nhanh
        self.last_draw_len = 0

        # Button actions
        self.close_button.clicked.connect(self.close)
        self.save_graph_button.clicked.connect(self.save_graph_as)

    def scale_to_screen(self):
        """Shrink the window to fit available screen space (similar to plant_model scaling)."""
        screen = QtWidgets.QApplication.primaryScreen()
        if not screen:
            return
        avail = screen.availableGeometry()
        w0, h0 = self.width(), self.height()
        factor = min(avail.width() / w0, avail.height() / h0, 1.0)
        if factor >= 1.0:
            return
        self.resize(int(w0 * factor), int(h0 * factor))
        for w in self.findChildren(QtWidgets.QWidget):
            g = w.geometry()
            w.setGeometry(
                int(g.x() * factor),
                int(g.y() * factor),
                int(g.width() * factor),
                int(g.height() * factor),
            )
            f = w.font()
            ps = f.pointSize()
            if ps > 0:
                f.setPointSize(max(6, int(ps * factor)))
                w.setFont(f)

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

        # RESET METRICS
        self.n = 0
        self.sum_e2 = 0.0
        self.sum_abs_e = 0.0

        self.sum_abs_u = 0.0
        self.sum_u2 = 0.0

        self.sum_ep2 = 0.0
        self.sum_abs_ep = 0.0
        self.sum_bias = 0.0

        self.update_graph(restart=True)

    def append_data(self, t, r, y, y_pred, u):
        t = np.atleast_1d(t)
        r = np.atleast_1d(r)
        y = np.atleast_1d(y)
        y_pred = np.atleast_1d(y_pred)
        u = np.atleast_1d(u)

        # store for plotting
        self.data_t.extend(t)
        self.data_ref.extend(r)
        self.data_y.extend(y)
        self.data_pred.extend(y_pred)
        self.data_u.extend(u)

        # incremental update
        e = r - y
        ep = y - y_pred

        m = len(e)
        self.n += m

        # tracking error accumulators
        self.sum_e2 += np.sum(e * e)
        self.sum_abs_e += np.sum(np.abs(e))

        # control effort accumulators
        self.sum_abs_u += np.sum(np.abs(u))
        self.sum_u2 += np.sum(u * u)

        # prediction error accumulators
        self.sum_ep2 += np.sum(ep * ep)
        self.sum_abs_ep += np.sum(np.abs(ep))
        self.sum_bias += np.sum(ep)

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
        if self.n > 0:
            RMSE_track = np.sqrt(self.sum_e2 / self.n)
            MAE_track = self.sum_abs_e / self.n
            mean_abs_u_track = self.sum_abs_u / self.n
            rms_u_track = np.sqrt(self.sum_u2 / self.n)
            steady_err_track = abs(self.data_ref[-1] - self.data_y[-1])

            RMSE_pred = np.sqrt(self.sum_ep2 / self.n)
            MAE_pred = self.sum_abs_ep / self.n
            bias = self.sum_bias / self.n

            # corr: compute full, very cheap
            y = np.array(self.data_y)
            yp = np.array(self.data_pred)
            corr = np.corrcoef(y, yp)[0, 1] if len(y) > 2 else np.nan
        else:
            RMSE_track = 0.0
            MAE_track = 0.0
            steady_err_track = 0.0
            mean_abs_u_track = 0.0
            rms_u_track = 0.0

            RMSE_pred = 0.0
            MAE_pred = 0.0
            corr = 0.0
            bias = 0.0

        # cập nhật text
        self.RMSE_track.setText(f"{RMSE_track:.3f}" if not np.isnan(RMSE_track) else "-")
        self.MAE_track.setText(f"{MAE_track:.3f}" if not np.isnan(MAE_track) else "-")
        self.steady_err_track.setText(f"{steady_err_track:.3f}" if not np.isnan(steady_err_track) else "-")
        self.mean_abs_u_track.setText(f"{mean_abs_u_track:.3f}" if not np.isnan(mean_abs_u_track) else "-")
        self.rms_u_track.setText(f"{rms_u_track:.3f}" if not np.isnan(rms_u_track) else "-")
        self.RMSE_pred.setText(f"{RMSE_pred:.4f}")
        self.MAE_pred.setText(f"{MAE_pred:.4f}")
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
    from PyQt5.QtCore import QThread
    from backend.simulation_worker import SimulationWorker
    from backend.online_training_worker import OnlineTrainingWorker
    import torch
    app = QtWidgets.QApplication(sys.argv)

    # --- Tạo cửa sổ đồ thị ---
    w = OutputGraphWindow()
    w.show()

    # --- Tạo worker + thread ---
    thread = QThread()
    online_thread = QThread()
    worker = SimulationWorker()
    from backend.system_workspace import workspace
    online_worker = OnlineTrainingWorker(model=workspace.narma_model)

    worker.moveToThread(thread)
    online_worker.moveToThread(online_thread)

    yk_1 = 0
    train_online = True
    # ====== NỐI SIGNAL TỚI GRAPH WINDOW ======
    def on_data_ready(t, r, y, y_pred, u):
        global yk_1
        w.append_data(t, r, y, y_pred, u) 

        if train_online:
            # push sample to online training worker
            online_worker.push_sample(yk_1, u, y)
            yk_1 = y

    worker.data_ready.connect(on_data_ready)
    worker.finished.connect(thread.quit)

    # ====== Start worker khi thread bắt đầu ======
    thread.started.connect(worker.run)
    online_thread.started.connect(online_worker.run)

    # ====== Khi đóng cửa sổ thì stop worker ======
    def on_close():
        worker.stop()
        thread.quit()
        thread.wait()

        online_worker.stop()
        online_thread.quit()
        online_thread.wait()

    w.destroyed.connect(on_close)

    # ====== Start thread ======
    thread.start()
    print("Simulation started.")
    online_thread.start()
    print("Online training started.")

    sys.exit(app.exec_())
