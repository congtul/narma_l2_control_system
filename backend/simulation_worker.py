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
        self.t = 0
        self.running = True
        self.kp = 1
        self.ki = 1.5
        self.kd = 0.0025
        self.dt = workspace.dt
        print("Discretizing plant transfer function...")
        print(f"workspace.plant['num_cont']: {workspace.plant['num_cont']}")
        print(f"workspace.plant['den_cont']: {workspace.plant['den_cont']}")
        print(f"workspace.dt: {workspace.dt}")
        workspace.plant["num_disc"], workspace.plant["den_disc"] = utils.discretize_tf(workspace.plant["num_cont"], workspace.plant["den_cont"], workspace.dt)
        print("num disc:", workspace.plant["num_disc"])
        print("den disc:", workspace.plant["den_disc"])
        self.u_hist = [0] * workspace.narma_model.nu
        self.y_hist = [0] * workspace.narma_model.ny
        self.u_plant_hist = [0] * len(workspace.plant.get("num_disc", [1]))
        self.y_plant_hist = [0] * (len(workspace.plant.get("den_disc", [1])) - 1)
        self.y = 0.0

    # For backup plan
    def pid_control(self, r, y, integral_error, prev_error):
        error = r - y
        integral_error += error * self.dt
        derivative_error = (error - prev_error) / self.dt if self.dt > 0 else 0.0

        u = self.kp * error + self.ki * integral_error + self.kd * derivative_error
        return u, integral_error, error

    def run(self):
        ref = getattr(workspace, "reference", {})
        if (
            not isinstance(ref, dict)
            or "t" not in ref
            or "ref" not in ref
            or len(ref.get("t", [])) == 0
            or len(ref.get("ref", [])) == 0
        ):
            print("[WARN] Reference signal is missing; simulation aborted. Please create input in the Input window.")
            self.finished.emit()
            return

        while (self.running) and (self.t <= workspace.run_time):
            # reference at current step and next step
            # r = 100*np.sin(5*self.t)
            r = np.interp(self.t, workspace.reference['t'], workspace.reference['ref'])               # r(k)

            self.y_hist_t = torch.tensor(self.y_hist, dtype=torch.float32)
            self.u_hist_t = torch.tensor(self.u_hist, dtype=torch.float32)

            u = workspace.narma_model.compute_control(self.y_hist_t, self.u_hist_t, r)
            # low pass filter for narma control
            low_pass_coef = 1
            u = (1-low_pass_coef)*self.u_hist[0] + low_pass_coef*u
                
            # --- debug override u here ---

            # prediction (for monitoring)
            y_pred = workspace.narma_model.narma_forward(torch.cat([self.y_hist_t, self.u_hist_t]), u)

            # update plant input history then simulate plant
            self.u_plant_hist = [u] + self.u_plant_hist[:-1]
            self.y = utils.plant_response(workspace.plant["num_disc"], workspace.plant["den_disc"], self.u_plant_hist, self.y_plant_hist)
            self.y_plant_hist = [self.y] + self.y_plant_hist[:-1]

            # update controller-visible histories
            self.y_hist = [self.y] + self.y_hist[:-1]
            self.u_hist = [u] + self.u_hist[:-1]
            # debug
            # r = workspace.narma_model.compute_control(y_hist_t, u_hist_t, y)
            # end debug
            self.data_ready.emit(self.t, r, self.y, y_pred, u)
            time.sleep(workspace.dt)
            self.t += workspace.dt
            # print("t from worker:", self.t)

        # print(f"Simulation worker finished. Total time: {self.t:.2f} s, {self.running=}")
        self.finished.emit()

    def stop(self):
        self.running = False

    def reset(self):
        """Reset clock or internal variables"""
        self.t = 0
