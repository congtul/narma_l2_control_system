import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import backend.utils as utils
import numpy as np
import matplotlib.pyplot as plt

# Just for debug / test
def build_narma_dataset(y_data, u_data, ny=4, nu=4):
    """
    Trả về 3 tensor:
    - X: (num_samples, ny + nu) chứa history
    - Y: (num_samples,) chứa y_next
    - U: (num_samples,) chứa u_k_actual (giá trị thực tế áp vào plant)
    
    y_data, u_data: 1D array hoặc tensor
    """
    y = torch.tensor(y_data, dtype=torch.float32)
    u = torch.tensor(u_data, dtype=torch.float32)

    N = len(y)
    delay = max(ny, nu)
    X, Y, U = [], [], []

    for k in range(delay, N):
        xk = []

        # Lấy y_history: y(k-1), ..., y(k-ny)
        for i in range(1, ny+1):
            xk.append(y[k-i])

        # Lấy u_history: u(k-1), ..., u(k-nu)
        for i in range(1, nu+1):
            xk.append(u[k-i])

        X.append(xk)
        Y.append(y[k])
        U.append(u[k])  # giá trị u(k) thực tế

    X = torch.stack([torch.tensor(xi) for xi in X])
    Y = torch.stack(Y)
    U = torch.stack(U)

    return X, Y, U

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

        utils.init_weights_from_arrays(self.f, weight_dict_f)
        utils.init_weights_from_arrays(self.g, weight_dict_g)

# ---------------------------
# NARMA-L2 Controller
# ---------------------------
class NARMA_L2_Controller(nn.Module):
    def __init__(self, ny=4, nu=4, hidden=10, epsilon=1e-3, default_model=False):
        super().__init__()
        self.ny = ny
        self.nu = nu
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
    
    def narma_forward(self, x_history, u_k, device=None):
        """
        x_history: tensor shape (batch_size, ny+nu) hoặc (ny+nu,) cho batch_size=1
        u_k: tensor shape (batch_size,) hoặc scalar
        controller: instance của NARMA_L2_Controller
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.f.to(device)
        self.g.to(device)

        # Đảm bảo x, u cùng device
        if x_history.dim() == 1:
            x_history = x_history.view(1, -1)
        x_history = x_history.to(device)

        if not torch.is_tensor(u_k):
            u_k = torch.tensor([u_k], dtype=torch.float32, device=device)
        elif u_k.dim() == 0:
            u_k = u_k.view(1)

        y_pred = self.f(x_history).squeeze() + self.g(x_history).squeeze() * u_k.squeeze()
        return y_pred
    
    def train_narma(self, dataset, u_data=None, epochs=200, lr=1e-3, batch_size=1, device=None):
        """
        dataset: TensorDataset chứa x_history, y_real
        u_data: tensor/array chứa u(k) thực tế, chỉ cần nếu train online hoặc x không có u(k)
        batch_size: 1 = online, >1 = offline
        """
        # Autodetect device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {device}")

        self.f.to(device)
        self.g.to(device)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        optim = torch.optim.Adam(list(self.f.parameters()) + list(self.g.parameters()), lr=lr)
        loss_fn = nn.MSELoss()

        for ep in range(epochs):
            total_loss = 0

            for i, (x, y_real) in enumerate(loader):
                x = x.to(device)
                y_real = y_real.to(device)

                # u_k lấy trực tiếp từ dữ liệu thực tế
                if u_data is not None:
                    start_idx = i * batch_size
                    u_k = u_data[start_idx:start_idx + batch_size].detach().clone().to(device).float()
                else:
                    u_k = x[:, -self.nu]

                f_k = self.f(x).squeeze()
                g_k = self.g(x).squeeze()
                y_pred = f_k + g_k * u_k

                # Fix size cho batch_size=1
                y_real = y_real.view_as(y_pred)

                loss = loss_fn(y_pred, y_real)

                optim.zero_grad()
                loss.backward()
                optim.step()

                total_loss += loss.item() * x.size(0)

            print(f"Epoch {ep+1}/{epochs}  Loss={total_loss/len(dataset):.6f}")


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

    print("Generating training data...")
    from backend.system_workspace import workspace
    workspace.plant["num_disc"], workspace.plant["den_disc"] = utils.discretize_tf(workspace.plant["num_cont"], workspace.plant["den_cont"], workspace.dt)
    u_hist = [0] * len(workspace.plant["num_disc"])
    y_hist = [0] * (len(workspace.plant["den_disc"]) - 1)
    t = np.linspace(0, 50, int(50/workspace.dt))
    u = np.sin(2*t)
    y = np.zeros_like(t)

    for i in range(len(t)):
        y[i] = utils.plant_response(workspace.plant["num_disc"], workspace.plant["den_disc"], u_hist, y_hist)
        y_hist = [y[i]] + y_hist[:-1]
        u_hist = [u[i]] + u_hist[:-1]

    print("Building dataset from generated data...")
    X, Y, U = build_narma_dataset(y, u, ny=4, nu=4)
    dataset = TensorDataset(X, Y)

    default_controller.train_narma(dataset, u_data=U, epochs=100, lr=1e-4, batch_size=32)

    # Lấy device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_controller.f.to(device)
    default_controller.g.to(device)

    # History buffers
    ny, nu = default_controller.ny, default_controller.nu
    delay = max(ny, nu)

    # Khởi tạo y_pred buffer
    y_pred = []

    # Dùng dữ liệu gốc y, u, t
    y_hist = list(y[:delay])  # initial y history
    u_hist = list(u[:delay])  # initial u history

    for k in range(delay, len(y)):
        # Convert history thành tensor
        y_hist_t = torch.tensor(y_hist[-ny:], dtype=torch.float32)
        u_hist_t = torch.tensor(u_hist[-nu:], dtype=torch.float32)
        
        # u_k = dữ liệu thực tế
        u_k = u[k]

        # Tính y_pred tại step k
        y_k_pred = default_controller.narma_forward(torch.cat([y_hist_t, u_hist_t]), u_k, device=device)
        y_pred.append(y_k_pred.item())

        # Cập nhật history cho step tiếp theo
        y_hist.append(y[k])  # dùng y thực tế để tiếp tục history
        u_hist.append(u_k)

    # Convert sang numpy
    y_pred_np = np.array(y_pred)
    y_real_np = y[delay:]
    t_plot = t[delay:]

    # ---- Plot 1: y_real vs y_pred ----
    plt.figure(figsize=(10, 4))
    plt.plot(t_plot, y_real_np, label="y_real", color="blue")
    plt.plot(t_plot, y_pred_np, label="y_pred", color="red", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("y")
    plt.title("NARMA-L2 Model Prediction vs Real Data")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---- Plot 2: prediction error ----
    error = y_real_np - y_pred_np
    plt.figure(figsize=(10, 4))
    plt.plot(t_plot, error, label="prediction error", color="purple")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.title("Prediction Error of NARMA-L2 Model")
    plt.grid(True)
    plt.show()