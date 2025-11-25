# -*- coding: utf-8 -*-
import sys, os
import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.model_train_ui import Ui_MainWindow  # import UI đã convert

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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Tạo cửa sổ plot riêng
        self.plot_win = TrainingPlotWindow(parent=self)
        self.plot_win.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        # --- Các biến nội bộ ---
        self.epoch_total = 100
        self.epoch_count = 0

        # --- Gắn signal ---
        self.ui.performance_btn.clicked.connect(self._demo_update)
        self.ui.performance_btn.clicked.connect(self.plot_win.show)
        self.ui.stop_btn.clicked.connect(self._stop_training)
        self.ui.cancel_btn.clicked.connect(self.close)

        # --- Timer mô phỏng ---
        self.demo_timer = QtCore.QTimer()
        self.demo_timer.setInterval(80)
        self.demo_timer.timeout.connect(self._tick)
        QtCore.QTimer.singleShot(0, lambda: (self._set_epoch_count(0, self.epoch_total), self.demo_timer.start()))

    # --- Các hàm phụ ---
    def _set_epoch_count(self, current: int, total: int = None):
        if total is not None:
            self.ui.progress_bar.setRange(0, int(total))
        self.ui.progress_bar.setValue(int(current))

    def _tick(self):
        if self.epoch_count < self.epoch_total:
            self.epoch_count += 1
            self._set_epoch_count(self.epoch_count, self.epoch_total)
        else:
            self.demo_timer.stop()

    def _stop_training(self):
        self.demo_timer.stop()

    def _demo_update(self):
        t = np.linspace(0, 50, 500)
        inp = np.sin(0.4*t) + 0.2*np.random.randn(len(t))
        plant = np.sin(0.4*t + 0.6) + 0.3*np.random.randn(len(t))
        err = (plant - inp) * 0.01
        nn = inp + 0.1*np.random.randn(len(t))
        self.plot_win.update_plots(t, inp, plant, err, nn)


# ------------------- Main -------------------
if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    win = ModelTrainWindow()
    win.show()
    sys.exit(app.exec_())
