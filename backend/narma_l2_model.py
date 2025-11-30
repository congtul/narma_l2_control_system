import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import backend.utils as utils
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Dataset builder
# ---------------------------
def build_narma_dataset(y_data, u_data, ny=4, nu=4):
    """
    X: (num_samples, ny+nu) chứa history
    Y: (num_samples,) chứa y_next
    U: (num_samples,) chứa u_k_actual
    """
    y = torch.tensor(y_data, dtype=torch.float32)
    u = torch.tensor(u_data, dtype=torch.float32)
    delay = max(ny, nu)

    X = torch.stack([torch.cat([y[k-ny:k], u[k-nu:k]]) for k in range(delay, len(y))])
    Y = y[delay:]
    U = u[delay:]
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
        return self.net(x)

# ---------------------------
# NARMA-L2 Default Model
# ---------------------------
class NARMA_L2_Model:
    def __init__(self, ny=4, nu=4, hidden=10, default_model=False):
        if default_model:
            ny, nu, hidden = 4, 4, 10

        self.f = ANN_Model(ny, nu, hidden)
        self.g = ANN_Model(ny, nu, hidden)

        if not default_model:
            return

        # Khởi tạo weights cố định
        def gen_weights(h, n, a, b, bias_base):
            w1 = torch.tensor([[a*(i+1)+b*(j+1) for j in range(n)] for i in range(h)], dtype=torch.float32)
            b1 = torch.tensor([bias_base*(i+1) for i in range(h)], dtype=torch.float32)
            w2 = torch.tensor([[0.05*(i+1) for i in range(h)]], dtype=torch.float32)
            b2 = torch.tensor([0.2], dtype=torch.float32)
            return w1, b1, w2, b2

        f_w1, f_b1, f_w2, f_b2 = gen_weights(hidden, ny+nu, 0.01, 0.001, 0.1)
        g_w1, g_b1, g_w2, g_b2 = gen_weights(hidden, ny+nu, 0.02, 0.002, 0.15)
        g_w2 *= 0.8; g_b2.fill_(0.25)  # chỉnh lại cho g

        utils.init_weights_from_arrays(self.f, {'1.weight': f_w1, '1.bias': f_b1, '2.weight': f_w2, '2.bias': f_b2})
        utils.init_weights_from_arrays(self.g, {'1.weight': g_w1, '1.bias': g_b1, '2.weight': g_w2, '2.bias': g_b2})

# ---------------------------
# NARMA-L2 Controller
# ---------------------------
class NARMA_L2_Controller(nn.Module):
    def __init__(self, ny=4, nu=4, hidden=10, epsilon=1e-3, default_model=False):
        super().__init__()
        self.ny, self.nu, self.epsilon = ny, nu, epsilon
        model = NARMA_L2_Model(ny, nu, hidden, default_model)
        self.f, self.g = model.f, model.g

    def compute_control(self, y_hist, u_hist, y_ref_future):
        x = torch.cat([y_hist, u_hist]).view(1, -1)
        f_k = self.f(x)[0,0]
        g_k = self.g(x)[0,0] + self.epsilon * torch.sign(self.g(x)[0,0])
        return ((y_ref_future - f_k) / g_k).item()
    
    def narma_forward(self, x_history, u_k, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.f.to(device); self.g.to(device)
        if x_history.dim()==1: x_history = x_history.view(1,-1)
        x_history = x_history.to(device)
        u_k = u_k if torch.is_tensor(u_k) else torch.tensor([u_k], dtype=torch.float32, device=device)
        return (self.f(x_history).squeeze() + self.g(x_history).squeeze()*u_k.squeeze())
    
    def train_narma(self, dataset, u_data=None, epochs=200, lr=1e-3, batch_size=1, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.f.to(device); self.g.to(device)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        optim = torch.optim.Adam(list(self.f.parameters()) + list(self.g.parameters()), lr=lr)
        loss_fn = nn.MSELoss()

        for ep in range(epochs):
            total_loss = 0
            for i, (x, y_real) in enumerate(loader):
                x, y_real = x.to(device), y_real.to(device)
                u_k = u_data[i*batch_size:(i+1)*batch_size].to(device) if u_data is not None else x[:,-self.nu]
                y_pred = self.f(x).squeeze() + self.g(x).squeeze()*u_k
                loss = loss_fn(y_pred, y_real.view_as(y_pred))
                optim.zero_grad(); loss.backward(); optim.step()
                total_loss += loss.item()*x.size(0)
            print(f"Epoch {ep+1}/{epochs}  Loss={total_loss/len(dataset):.6f}")

# ---------------------------
# TEST FORWARD
# ---------------------------
if __name__ == "__main__":
    y_hist = torch.ones(4)
    u_hist = torch.ones(4)
    y_ref_future = torch.tensor(1.0)

    default_controller = NARMA_L2_Controller(ny=4, nu=4, hidden=10, default_model=True)
    print("Default NARMA-L2 Controller:")
    print("Controller output u(k) with default model =", default_controller.compute_control(y_hist, u_hist, y_ref_future))

    print("Generating training data...")
    from backend.system_workspace import workspace
    workspace.plant["num_disc"], workspace.plant["den_disc"] = utils.discretize_tf(
        workspace.plant["num_cont"], workspace.plant["den_cont"], workspace.dt
    )
    u_hist, y_hist = [0]*len(workspace.plant["num_disc"]), [0]*(len(workspace.plant["den_disc"])-1)
    t = np.linspace(0, 50, int(50/workspace.dt))
    u, y = np.sin(2*t), np.zeros_like(t)
    for i in range(len(t)):
        y[i] = utils.plant_response(workspace.plant["num_disc"], workspace.plant["den_disc"], u_hist, y_hist)
        y_hist = [y[i]] + y_hist[:-1]; u_hist = [u[i]] + u_hist[:-1]

    print("Building dataset from generated data...")
    X, Y, U = build_narma_dataset(y, u, ny=4, nu=4)
    dataset = TensorDataset(X, Y)
    default_controller.train_narma(dataset, u_data=U, epochs=100, lr=1e-3, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_controller.f.to(device); default_controller.g.to(device)

    ny, nu = default_controller.ny, default_controller.nu
    delay = max(ny, nu)
    y_hist, u_hist = list(y[:delay]), list(u[:delay])
    y_pred = []

    for k in range(delay, len(y)):
        y_hist_t = torch.tensor(y_hist[-ny:], dtype=torch.float32)
        u_hist_t = torch.tensor(u_hist[-nu:], dtype=torch.float32)
        y_pred.append(default_controller.narma_forward(torch.cat([y_hist_t, u_hist_t]), u[k], device=device).item())
        y_hist.append(y[k]); u_hist.append(u[k])

    y_pred_np, y_real_np, t_plot = np.array(y_pred), y[delay:], t[delay:]

    # ---- Plot 1: y_real vs y_pred ----
    plt.figure(figsize=(10, 4))
    plt.plot(t_plot, y_real_np, label="y_real", color="blue")
    plt.plot(t_plot, y_pred_np, label="y_pred", color="red", linestyle="--")
    plt.xlabel("Time"); plt.ylabel("y")
    plt.title("NARMA-L2 Model Prediction vs Real Data")
    plt.legend(); plt.grid(True); plt.show()

    # ---- Plot 2: prediction error ----
    plt.figure(figsize=(10, 4))
    plt.plot(t_plot, y_real_np - y_pred_np, label="prediction error", color="purple")
    plt.xlabel("Time"); plt.ylabel("Error")
    plt.title("Prediction Error of NARMA-L2 Model")
    plt.grid(True); plt.show()
