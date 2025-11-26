import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import backend.utils as ultils

# ---------------------------
# Plant model: f/g
# ---------------------------
class ANN_Model(nn.Module):
    """Plant model: mạng f/g"""
    def __init__(self, ny=4, nu=4, hidden=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ny + nu, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x):
        return self.net(x)  # trả về shape (batch, 1)

# ---------------------------
# NARMA-L2 Default Model (weights hardcode)
# ---------------------------
class NARMA_L2_Model:
    def __init__(self, ny=4, nu=4, hidden=10, default_model=False):
        if default_model:
            ny = 4
            nu = 4
            hidden = 10

        self.f = ANN_Model(ny, nu, hidden)
        self.g = ANN_Model(ny, nu, hidden)

        if not default_model:
            return

        # ===== Khởi tạo weights cố định =====
        #TODO: Thay bằng load từ file nếu cần
        f_w1 = torch.zeros(hidden, ny + nu)
        for i in range(hidden):
            for j in range(ny + nu):
                f_w1[i, j] = 0.01*(i+1) + 0.001*(j+1)
        f_b1 = torch.zeros(hidden)
        for i in range(hidden):
            f_b1[i] = 0.1 * (i+1)

        f_w2 = torch.zeros(1, hidden)
        for i in range(hidden):
            f_w2[0, i] = 0.05 * (i+1)
        f_b2 = torch.tensor([0.2])

        g_w1 = torch.zeros(hidden, ny + nu)
        for i in range(hidden):
            for j in range(ny + nu):
                g_w1[i, j] = 0.02*(i+1) + 0.002*(j+1)
        g_b1 = torch.zeros(hidden)
        for i in range(hidden):
            g_b1[i] = 0.15 * (i+1)

        g_w2 = torch.zeros(1, hidden)
        for i in range(hidden):
            g_w2[0, i] = 0.04 * (i+1)
        g_b2 = torch.tensor([0.25])
        # End TODO

        weight_dict_f = {
            '1.weight': f_w1,
            '1.bias': f_b1,
            '2.weight': f_w2,
            '2.bias': f_b2
        }

        weight_dict_g = {
            '1.weight': g_w1,
            '1.bias': g_b1,
            '2.weight': g_w2,
            '2.bias': g_b2
        }

        ultils.init_weights_from_arrays(self.f, weight_dict_f)
        ultils.init_weights_from_arrays(self.g, weight_dict_g)

# ---------------------------
# NARMA-L2 Controller
# ---------------------------
class NARMA_L2_Controller(nn.Module):
    def __init__(self, ny=4, nu=4, hidden=10, epsilon=1e-3, default_model=False):
        super().__init__()
        self.epsilon = epsilon
        narma_model = NARMA_L2_Model(ny, nu, hidden, default_model)
        self.f = narma_model.f
        self.g = narma_model.g

    def compute_control(self, y_hist, u_hist, y_ref_future):
        # x shape (1, ny+nu)
        x = torch.cat([y_hist, u_hist]).view(1, -1)
        f_k = self.f(x)[0, 0]  # tensor scalar
        g_k = self.g(x)[0, 0]  # tensor scalar

        # Saturation tránh chia 0
        g_k = g_k + self.epsilon * torch.sign(g_k)
        u_k = (y_ref_future - f_k) / g_k
        return u_k.item()  # trả về float

# ---------------------------
# TEST FORWARD
# ---------------------------
if __name__ == "__main__":
    # Input mẫu
    y_hist = torch.ones(4)
    u_hist = torch.ones(4)
    y_ref_future = torch.tensor(1.0)

    # Default model
    default_controller = NARMA_L2_Controller(ny=4, nu=4, hidden=10, default_model=True)
    print("Default NARMA-L2 Controller:")
    u_cmd = default_controller.compute_control(y_hist, u_hist, y_ref_future)
    print("Controller output u(k) with default model =", u_cmd)

    # New random model
    controller = NARMA_L2_Controller(ny=4, nu=4, hidden=10, default_model=False)
    print("New NARMA-L2 Controller:")
    u_cmd2 = controller.compute_control(y_hist, u_hist, y_ref_future)
    print("Controller output u(k) with new model =", u_cmd2)
