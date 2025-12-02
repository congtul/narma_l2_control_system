import profile
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
        self.kp = 1
        self.ki = 1.5
        self.kd = 0.0025
        self.dt = workspace.dt
        workspace.plant["num_disc"], workspace.plant["den_disc"] = utils.discretize_tf(workspace.plant["num_cont"], workspace.plant["den_cont"], workspace.dt)

    # For backup plan
    def pid_control(self, r, y, integral_error, prev_error):
        error = r - y
        integral_error += error * self.dt
        derivative_error = (error - prev_error) / self.dt if self.dt > 0 else 0.0

        u = self.kp * error + self.ki * integral_error + self.kd * derivative_error
        return u, integral_error, error

    def run(self):
        t = 0

        u_hist = [0] * workspace.narma_model.nu
        y_hist = [0] * workspace.narma_model.ny

        print("num disc:", workspace.plant["num_disc"])
        print("den disc:", workspace.plant["den_disc"])
        u_plant_hist = [0] * len(workspace.plant.get("num_disc", [1]))
        y_plant_hist = [0] * (len(workspace.plant.get("den_disc", [1])) - 1)
        integral_error = 0.0
        prev_error = 0.0
        y = 0
        while (self.running) and (t <= workspace.run_time):
            # reference at current step and next step
            r = 100*np.sin(3*t)               # r(k)

            T = workspace.run_time
            # # profile 1
            # if t < 0.1 * T:
            #     r = 50
            # elif t < 0.2 * T:
            #     r = 100
            # elif t < 0.3 * T:
            #     r = -50
            # elif t < 0.4 * T:
            #     r = 0
            # elif t < 0.55 * T:
            #     r = 75
            # elif t < 0.70 * T:
            #     r = -100
            # elif t < 0.85 * T:
            #     r = 120
            # else:
            #     r = 20

            # profile 2
            # if t < 0.15 * T:
            #     r = 0
            # elif t < 0.30 * T:
            #     r = 50 * (t / (0.30*T))     # ramp lên 50
            # elif t < 0.45 * T:
            #     r = 50
            # elif t < 0.60 * T:
            #     r = 50 - 100 * ( (t-0.45*T) / (0.15*T) )    # ramp xuống -50
            # elif t < 0.75 * T:
            #     r = -50
            # else:
            #     r = 0

            # profile 3
            if t < 0.1*T:
                r = 0
            elif t < 0.2*T:
                r = 50
            elif t < 0.35*T:
                r = 120
            elif t < 0.50*T:
                r = -60
            elif t < 0.65*T:
                r = 80
            elif t < 0.80*T:
                r = -120
            elif t < 0.9*T:
                r = 40
            else:
                r = 0

            y_hist_t = torch.tensor(y_hist, dtype=torch.float32)
            u_hist_t = torch.tensor(u_hist, dtype=torch.float32)

            u = workspace.narma_model.compute_control(y_hist_t, u_hist_t, r)
            # low pass filter for narma control
            low_pass_coef = 0.95
            u = (1-low_pass_coef)*u_hist[0] + low_pass_coef*u
                
            # --- debug override u here ---

            # prediction (for monitoring)
            y_pred = workspace.narma_model.narma_forward(torch.cat([y_hist_t, u_hist_t]), u)

            # update plant input history then simulate plant
            u_plant_hist = [u] + u_plant_hist[:-1]
            y = utils.plant_response(workspace.plant["num_disc"], workspace.plant["den_disc"], u_plant_hist, y_plant_hist)
            y_plant_hist = [y] + y_plant_hist[:-1]

            # update controller-visible histories
            y_hist = [y] + y_hist[:-1]
            u_hist = [u] + u_hist[:-1]

            # debug
            # r = workspace.narma_model.compute_control(y_hist_t, u_hist_t, y)
            # end debug
            self.data_ready.emit(t, r, y, y_pred, u)
            time.sleep(workspace.dt)
            t += workspace.dt

        self.finished.emit()

    def stop(self):
        self.running = False

    def reset(self):
        """Reset clock or internal variables"""
        self.t = 0
