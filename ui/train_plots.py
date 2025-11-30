# training_plots.py
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class TrainingData(QtCore.QObject):
    """Data model that notifies plots whenever values change."""
    changed = QtCore.pyqtSignal(object, object, object, object, object)  # t, inp, plant, err, nn

    def __init__(self, parent=None):
        super().__init__(parent)

    def set_all(self, t, inp, plant, err, nn):
        self.changed.emit(t, inp, plant, err, nn)

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
            ax.set_xlabel("epoch (s)")
        self.lines = [ax.plot([], [], lw=1)[0] for ax in self.axes.flat]

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    def bind(self, model: TrainingData):
        model.changed.connect(self._on_changed)

    def _on_changed(self, t, inp, plant, err, nn):
        # Convert everything to 1-D numpy arrays
        t = np.asarray(t).ravel()
        series = [np.asarray(inp).ravel(),
                  np.asarray(plant).ravel(),
                  np.asarray(err).ravel(),
                  np.asarray(nn).ravel()]

        # Nothing to draw
        if t.size == 0:
            for line in self.lines:
                line.set_data([], [])
            self.canvas.draw_idle()
            return

        # Plot each series (truncate to common length)
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

    # convenience forwarder
    def update_plots(self, t, inp, plant, err, nn):
        self.model.set_all(t, inp, plant, err, nn)
