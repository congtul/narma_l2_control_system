"""
system_workspace.py
-------------------
Global workspace chứa tất cả trạng thái simulation, NN model, plant, reference signal,
và các buffer dữ liệu để UI và các thread khác truy cập.
"""

from collections import deque
from copy import deepcopy

class SystemWorkspace:
    """Singleton workspace cho toàn bộ app."""

    def __init__(self):
        self.dt = 0.01  # thời gian mẫu chung
        self.run_time = 10  # thời gian chạy mô phỏng
        # ---------------- Plant model ----------------
        # Transfer function user nhập
        self.plant = {
            "num_cont": [0.01],           # list or ndarray
            "den_cont": [0.005, 0.07, 0.2],           # list or ndarray
            "num_disc": [],           # discrete-time numerator
            "den_disc": [],           # discrete-time denominator
        }

        # ---------------- NARMA-L2 model ----------------
        self.narma_model = None   # object neural net
        self.narma_config = {}    # config dict (order, hidden units...)

        # ---------------- Reference signal ----------------
        self.reference = {
            "type": None,          # e.g., "step", "sin", "random"
            "params": {},          # dict of signal parameters
            "buffer": deque(maxlen=10000)  # generated signal
        }

        # ---------------- Runtime simulation buffers ----------------
        self.simulation = {
            "t": deque(maxlen=10000),
            "y": deque(maxlen=10000),
            "y_pred": deque(maxlen=10000),
            "u": deque(maxlen=10000),
            "r": deque(maxlen=10000),
            "running": False,
            "step_ms": 10,  # loop interval
        }

        # ---------------- Flags / mode ----------------
        self.flags = {
            "mode": "normal",  # or "advance"
            "training": False,
            "simulation_paused": False,
        }

        # ---------------- Logs ----------------
        self.logs = []

    # ---------------- Helper methods ----------------
    def reset_simulation_buffers(self):
        """Reset tất cả buffer dữ liệu mô phỏng"""
        for key in ["t", "y", "y_pred", "u", "r"]:
            self.simulation[key].clear()
        print("[INFO] Simulation buffers reset.")

    def reset_plant(self):
        """Reset plant về mặc định"""
        self.plant["num"] = []
        self.plant["den"] = []
        self.plant["ss"] = None
        print("[INFO] Plant reset.")

    def reset_reference(self):
        """Reset reference signal"""
        self.reference["buffer"].clear()
        self.reference["type"] = None
        self.reference["params"] = {}
        print("[INFO] Reference reset.")

    def reset_all(self):
        """Reset toàn bộ workspace"""
        self.reset_simulation_buffers()
        self.reset_plant()
        self.reset_reference()
        self.narma_model = None
        self.narma_config = {}
        self.flags = {
            "mode": "normal",
            "training": False,
            "simulation_paused": False,
        }
        self.logs.clear()
        print("[INFO] Workspace fully reset.")


# ---------------- Singleton instance ----------------
workspace = SystemWorkspace()
