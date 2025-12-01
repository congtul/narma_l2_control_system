# -*- coding: utf-8 -*-
import queue
import sys, os
import threading
import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.model_train_ui import Ui_MainWindow  # import UI đã convert
from backend.system_workspace import workspace
from backend import utils
import torch
from torch.utils.data import DataLoader
from backend.training_worker import TrainingWorker

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"


# ------------------- Data Model -------------------
class TrainingData(QtCore.QObject):
    """Data model that notifies plots whenever values change."""
    changed = QtCore.pyqtSignal(object, object, object, object, object)  # t, inp, plant, err, nn

    def __init__(self, parent=None):
        super().__init__(parent)

    def set_all(self, t, inp, plant, err, nn):
        self.changed.emit(t, inp, plant, err, nn)


# ------------------- Plot Widget -------------------
class TrainingPlotWidget(QtWidgets.QWidget):
    """4-panel matplotlib widget that listens to a TrainingData model."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.axes = self.fig.subplots(2, 2)
        titles = ["Input", "Plant Output", "Error", "NN Output"]
        for ax, t in zip(self.axes.flat, titles):
            ax.set_title(t)
            ax.set_xlabel("time (s)")
        self.lines = [ax.plot([], [], lw=1)[0] for ax in self.axes.flat]

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    def bind(self, model: TrainingData):
        model.changed.connect(self._on_changed)

    def _on_changed(self, t, inp, plant, err, nn):
        t = np.asarray(t).ravel()
        series = [np.asarray(inp).ravel(),
                  np.asarray(plant).ravel(),
                  np.asarray(err).ravel(),
                  np.asarray(nn).ravel()]

        if t.size == 0:
            for line in self.lines:
                line.set_data([], [])
            self.canvas.draw_idle()
            return

        for line, ax, y in zip(self.lines, self.axes.flat, series):
            n = min(t.size, y.size)
            if n == 0:
                line.set_data([], [])
                continue

            tt = t[:n]
            yy = y[:n]

            line.set_data(tt, yy)
            ax.set_xlim(float(tt[0]), float(tt[-1]))

            ymin = float(np.nanmin(yy))
            ymax = float(np.nanmax(yy))
            pad = (ymax - ymin) * 0.15 if ymax > ymin else 0.1
            ax.set_ylim(ymin - pad, ymax + pad)

        self.canvas.draw_idle()


class TrainingPlotWindow(QtWidgets.QMainWindow):
    """Top-level window containing the 2x2 plot grid and a public model."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Result Visualization")
        self.resize(900, 700)
        self.model = TrainingData(self)
        widget = TrainingPlotWidget(self)
        widget.bind(self.model)
        self.setCentralWidget(widget)

    def update_plots(self, t, inp, plant, err, nn):
        self.model.set_all(t, inp, plant, err, nn)


# ------------------- Main Window -------------------
class ModelTrainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None, epoch_total=100):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Tạo cửa sổ plot riêng
        self.plot_win = TrainingPlotWindow(parent=self)
        self.plot_win.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)        

        # --- Các biến nội bộ ---
        self.epoch_total = int(epoch_total)
        self.epoch_count = 0
        self.demo_history = {"t": [], "inp": [], "plant": [], "err": [], "nn": []}

        # --- Gắn signal ---
        self.ui.performance_btn.clicked.connect(self._demo_update)
        self.ui.performance_btn.clicked.connect(self.plot_win.show)
        self.ui.stop_btn.clicked.connect(self._stop_training)
        self.ui.cancel_btn.clicked.connect(self.close)

        X, Y, U = utils.build_narma_dataset(workspace.dataset["y"], workspace.dataset["u"], ny=workspace.narma_model.ny, nu=workspace.narma_model.nu)   

        X_train, X_tmp, Y_train, Y_tmp, U_train, U_tmp = train_test_split(
        X.numpy(), Y.numpy(), U.numpy(), test_size=0.2, shuffle=False)
        X_val, X_test, Y_val, Y_test, U_val, U_test = train_test_split(
            X_tmp, Y_tmp, U_tmp, test_size=0.5, shuffle=False)
        
        train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32), torch.tensor(U_train, dtype=torch.float32))
        val_data = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32), torch.tensor(U_val, dtype=torch.float32))
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.Y_test = torch.tensor(Y_test, dtype=torch.float32)
        self.U_test = torch.tensor(U_test, dtype=torch.float32)

        # Tạo QThread worker
        self.worker = TrainingWorker(
            controller=workspace.narma_model,
            train_ds=train_ds,
            val_ds=val_data
        )

        # Connect signal vào GUI
        self.worker.epoch_signal.connect(self._epoch_callback)
        self.worker.finished_signal.connect(self._training_finished)

        # Start training
        self.worker.start()

    def _epoch_callback(self, epoch, train_loss, val_loss):
        # Update progress bar
        self._set_epoch_count(epoch, self.epoch_total)

        # Update GUI labels (nếu có)
        self.ui.train_loss_label.setText(f"{train_loss:.6f}")
        if val_loss is not None:
            self.ui.val_loss_label.setText(f"{val_loss:.6f}")

    def _training_finished(self, result):
        if result["status"] == "ok":
            QtWidgets.QMessageBox.information(self, "Done", "Training completed successfully!")
        elif result["status"] == "stopped":
            QtWidgets.QMessageBox.warning(self, "Stopped", "Training was stopped.")
        else:
            QtWidgets.QMessageBox.critical(self, "Error", result.get("message", "Unknown error"))


    # --- Các hàm phụ ---
    def _set_epoch_count(self, current: int, total: int = None):
        if total is not None:
            self.ui.progress_bar.setRange(0, int(total))
            self.ui.progress_max_label.setText(str(int(total)))
        self.ui.progress_bar.setValue(int(current))

    def _stop_training(self):
        if hasattr(self, "worker"):
            self.worker.stop()
        self._reset_demo_history()

    def _demo_update(self):
        """
        Vẽ đồ thị performance theo tập test:
        - NN output = y_pred (recursive)
        - plant = y_test thực
        - inp = u_test
        """

        model = workspace.narma_model
        model.eval()

        # ----- Lấy test data -----
        Y_test = self.Y_test.cpu().numpy().ravel()
        U_test = self.U_test.cpu().numpy().ravel()

        ny = workspace.narma_model.ny
        nu = workspace.narma_model.nu
        dt = workspace.dt

        delay = max(ny, nu)

        # ----- Khởi tạo lịch sử -----
        y_hist_seq = list(Y_test[:ny])
        u_hist_seq = list(U_test[:nu])

        y_pred = []

        # ----- Predict recursive giống code bạn đưa -----
        for k in range(delay, len(Y_test)):
            y_hist_t = torch.tensor(y_hist_seq[-ny:], dtype=torch.float32)
            u_hist_t = torch.tensor(u_hist_seq[-nu:], dtype=torch.float32)

            # Forward NARMA
            with torch.no_grad():
                yk = model.narma_forward(torch.cat([y_hist_t, u_hist_t]), U_test[k])

            y_pred.append(float(yk))

            # Cập nhật history
            y_hist_seq.append(Y_test[k])
            u_hist_seq.append(U_test[k])

        # ----- Convert numpy -----
        y_pred = np.array(y_pred)
        plant = Y_test[delay:]
        inp = U_test[delay:]

        # ----- Tạo time vector -----
        t = np.arange(len(plant)) * dt

        # ----- Error -----
        err = plant - y_pred

        # ----- Cập nhật plot -----
        self.plot_win.update_plots(
            t,
            inp,
            plant,
            err,
            y_pred
        )

    def _reset_demo_history(self):
        self.demo_history = {"t": [], "inp": [], "plant": [], "err": [], "nn": []}


# ------------------- Main -------------------
if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    dataset_path = "test_result/dataset.csv"
    data = np.loadtxt(dataset_path, delimiter=",", skiprows=1)
    workspace.dataset["t"] = data[:, 0]
    workspace.dataset["u"] = data[:, 1]
    workspace.dataset["y"] = data[:, 2]
    win = ModelTrainWindow()
    win.show()
    sys.exit(app.exec_())
