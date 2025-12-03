"""
system_workspace.py
-------------------
Global workspace chứa tất cả trạng thái simulation, NN model, plant, reference signal,
và các buffer dữ liệu để UI và các thread khác truy cập.
"""

from collections import deque
from backend.narma_l2_model import NARMA_L2_Controller

class SystemWorkspace:
    """Singleton workspace cho toàn bộ app."""

    def __init__(self):
        self.dt = self.get_default_sampling_time()  # sampling time
        self.run_time = self.get_default_runtime()  # total simulation time
        # ---------------- Plant model ----------------
        self.plant = {
            "mode": "dc_motor",  # "dc_motor" or "custom"
        }
        self.set_default_dc_motor_plant()

        # ---------------- NARMA-L2 model ----------------
        self.set_default_narma_l2_model()
        self.dataset = {
            "t": [],
            "u": [],
            "y": [],
        }
        self.training_online = False
        self.set_default_online_training_config()

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

    #--------------------Default DC motor params--------------------#
    def get_default_dc_motor_params(self):
        return {
            'L': 0.01,     # H (điện cảm nhỏ hơn)
            'R': 1.0,      # Ω (điện trở hợp lý)
            'Kb': 0.05,    # V·s/rad
            'Km': 0.05,    # N·m/A (moment mạnh hơn)
            'Kf': 0.001,   # N·m·s/rad (ma sát nhỏ)
            'J': 0.001,    # kg·m² (inertia nhỏ, nhanh đáp ứng)
            'Td': 0.001    # s
        }

    def get_default_dc_motor_tf(self):
        para = self.get_default_dc_motor_params()
        L = para['L']
        R = para['R']
        Kb = para['Kb']
        Km = para['Km']
        Kf = para['Kf']
        J = para['J']
        Td = para['Td']

        num = [Km]
        den = [L*J, L*Kf + R*J, R*Kf + Km*Kb]
        return num, den
    
    def get_default_sampling_time(self):
        return 0.01
    
    def get_default_runtime(self):
        return 10

    def set_default_dc_motor_plant(self):
        num, den = self.get_default_dc_motor_tf()
        self.plant['num_cont'] = num
        self.plant['den_cont'] = den

    #---------------- Default narma-l2 params ----------------#
    def get_default_narma_l2_params(self):
        return {
            "nu": 4,
            "ny": 3,
            "hidden_size": 9,
            "activation": "SiLU",
            "learning_rate": 1e-4,
            "training_epochs": 200,
            "training_sample_size": 10000,
            "backprop_batch_size": 32,
            "max_control": 12.0, # voltage giới hạn
            "min_control": -12.0,
            "max_output": 160.0, # tốc độ giới hạn (rad/s)
            "min_output": -160.0, # tốc độ giới hạn (rad/s)
            "min_interval": 0.1,
            "max_interval": 1,
            "sampling_time": self.get_default_sampling_time(),
            "patience": 10,
            "use_validation": True,
            "use_test_data": True
        }

    def set_default_narma_l2_model(self):
        config = self.get_default_narma_l2_params()
        self.narma_model = NARMA_L2_Controller(
            ny=config["ny"],
            nu=config["nu"],
            hidden=config["hidden_size"],
            epsilon=config["learning_rate"],
            max_control=config["max_control"],
            min_control=config["min_control"],
            max_output=config["max_output"],
            min_output=config["min_output"],
            epochs=config["training_epochs"],
            lr=config["learning_rate"],
            batch_size=config["backprop_batch_size"],
            patience=config["patience"],
            default_model=True
        )

    def set_default_online_training_config(self):
        self.online_training_config = {
            "lr": 5e-5,
            "batch_size": 5,
            "epoch": 2
        }

    # ---------------- Sampling time helpers ----------------
    def set_sampling_time(self, dt: float):
        """
        Valid range: [0.001, 0.02] seconds.
        """
        if dt < 0.001 or dt > 0.02:
            raise ValueError("Sampling time must be within [0.001, 0.02] seconds.")

        self.dt = float(dt)
        # self.simulation["step_ms"] = max(1, int(self.dt * 1000))
        print(f"[INFO] Sampling time updated to {self.dt} s")

# ---------------- Singleton instance ----------------
workspace = SystemWorkspace()
