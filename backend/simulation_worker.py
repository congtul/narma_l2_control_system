import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt5.QtCore import QObject, QThread, pyqtSignal
import numpy as np
import time
from backend.system_workspace import workspace
from backend import utils
import torch


class SimulationWorker(QObject):
    data_ready = pyqtSignal(float, float, float, float, float)
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        workspace.plant["num_disc"], workspace.plant["den_disc"] = utils.discretize_tf(workspace.plant["num_cont"], workspace.plant["den_cont"], workspace.dt)

    def run(self):
        t = 0
        
        u_hist = [0] * workspace.narma_model.nu
        y_hist = [0] * workspace.narma_model.ny

        print("num disc:", workspace.plant["num_disc"])
        print("den disc:", workspace.plant["den_disc"])
        u_plant_hist = [0] * len(workspace.plant.get("num_disc", [1]))
        y_plant_hist = [0] * (len(workspace.plant.get("den_disc", [1])) - 1)
        while (self.running) and (t <= workspace.run_time):
            #TODO: tham số tham chiếu có thể thay đổi theo thời gian, port từ workspace
            r = 70
            y_hist_t = torch.tensor(y_hist, dtype=torch.float32)
            u_hist_t = torch.tensor(u_hist, dtype=torch.float32)

            u = workspace.narma_model.compute_control(y_hist_t, u_hist_t, r)
            # u = 12*np.sin(5*t) # test signal
            y_pred = workspace.narma_model.narma_forward(torch.cat([y_hist_t, u_hist_t]), u)
            y = utils.plant_response(workspace.plant["num_disc"], workspace.plant["den_disc"], u_plant_hist, y_plant_hist)

            y_hist = [y] + y_hist[:-1]
            u_hist = [u] + u_hist[:-1]

            u_plant_hist = [u] + u_plant_hist[:-1]
            y_plant_hist = [y] + y_plant_hist[:-1]

            self.data_ready.emit(t, r, y, y_pred, u)
            time.sleep(workspace.dt)
            t += workspace.dt

        self.finished.emit()

    def stop(self):
        self.running = False

    def reset(self):
        """Reset clock or internal variables"""
        self.t = 0
