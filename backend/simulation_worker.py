from PyQt5.QtCore import QObject, QThread, pyqtSignal
import numpy as np
import time
from backend.system_workspace import workspace
from backend import utils

class SimulationWorker(QObject):
    data_ready = pyqtSignal(float, float, float, float, float)
    finished = pyqtSignal()

    def __init__(self, dt=0.05, parent=None):
        super().__init__(parent)
        self.dt = dt
        self.running = True

    def run(self):
        t = 0
        workspace.plant["num_disc"], workspace.plant["den_disc"] = utils.discretize_tf(workspace.plant["num_cont"], workspace.plant["den_cont"], workspace.dt)
        u_hist = [0] * len(workspace.plant["num_disc"])
        y_hist = [0] * (len(workspace.plant["den_disc"]) - 1)
        while self.running:
            y = utils.plant_response(workspace.plant["num_disc"], workspace.plant["den_disc"], u_hist, y_hist)
            y_pred = y*0.95 + np.random.normal(0, 0.02)

            r = y + np.random.normal(0, 0.05)
            u = 5*np.sin(5 * t)
            y_hist = [y] + y_hist[:-1]
            u_hist = [u] + u_hist[:-1]

            self.data_ready.emit(t, r, y, y_pred, u)
            time.sleep(self.dt)
            t += self.dt

            if t >= workspace.run_time:
                break
        self.finished.emit()

    def stop(self):
        self.running = False

    def reset(self):
        """Reset clock or internal variables"""
        self.t = 0
