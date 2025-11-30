import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import backend.utils as utils
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Dataset builder (vectorized)
# ---------------------------
def build_narma_dataset(y_data, u_data, ny=4, nu=4):
    """
    Vectorized dataset builder using numpy sliding window view.
    Returns tensors: X (N, ny+nu), Y (N,), U (N,)
    """
    y = np.asarray(y_data, dtype=np.float32)
    u = np.asarray(u_data, dtype=np.float32)
    delay = max(ny, nu)
    n_total = len(y) - delay
    if n_total <= 0:
        return torch.empty(0, ny + nu), torch.empty(0), torch.empty(0)

    # build sliding windows for y-history and u-history
    # For y-history we need last ny values before k, i.e. y[k-ny:k] for k in [delay..len(y)-1]
    # sliding_window_view returns windows of length ny for y[0..len(y)-1]
    from numpy.lib.stride_tricks import sliding_window_view
    y_win = sliding_window_view(y, window_shape=ny)  # shape (len(y)-ny+1, ny)
    u_win = sliding_window_view(u, window_shape=nu)

    # We need the windows that end at indices k-1 for k=delay..len(y)-1 -> those correspond to rows with index = (delay-1)...(len(y)-1)-1 => start = delay-ny?
    # Simpler: pick windows starting at indices (delay - ny) .. (len(y) - ny - 1)
    # But easier: build X by stacking y[k-ny:k] and u[k-nu:k] for k in range(delay, len(y))
    # Using slices:
    y_hist_matrix = y_win[delay - ny : delay - ny + n_total]
    u_hist_matrix = u_win[delay - nu : delay - nu + n_total]

    X = np.concatenate([y_hist_matrix, u_hist_matrix], axis=1)
    Y = y[delay:]
    U = u[delay:]
    return torch.from_numpy(X).float(), torch.from_numpy(Y).float(), torch.from_numpy(U).float()

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

        utils.load_weights_from_file(self, os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_weights.pth"))

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
        # y_hist, u_hist: 1D tensors length ny and nu
        x = torch.cat([y_hist, u_hist]).view(1, -1)
        f_k = self.f(x).view(-1)[0]
        g_out = self.g(x).view(-1)[0]
        # add small epsilon in direction of sign once
        g_k = g_out + self.epsilon * torch.sign(g_out)
        return ((y_ref_future - f_k) / g_k).item()
    
    def narma_forward(self, x_history, u_k, device=None):
        # x_history: either 1D or 2D (batch, features)
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.f.to(device); self.g.to(device)
        if x_history.dim() == 1:
            x_history = x_history.view(1, -1)
        x_history = x_history.to(device)
        if not torch.is_tensor(u_k):
            u_k = torch.tensor(u_k, dtype=torch.float32, device=device)
        else:
            u_k = u_k.to(device)
        f_out = self.f(x_history).squeeze()
        g_out = self.g(x_history).squeeze()
        return f_out + g_out * u_k.squeeze()
    
    def train_narma(
        self,
        train_ds: TensorDataset,
        val_data: TensorDataset = None,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
        patience: int = 10,
        device: torch.device = None,
        use_amp: bool = False
    ):
        """
        train_ds: TensorDataset with (x, y, u) OR (x, y) and pass u separately.
        If passed TensorDataset has 3 tensors, DataLoader yields (x,y,u) directly.
        """
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # move models once
        self.f.to(device)
        self.g.to(device)

        # Ensure train_ds yields u directly: if not, expect train_ds to be (x,y) and user passed u externally.
        # We'll support the common case where train_ds is (x,y,u)
        sample_len = len(train_ds[0])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_data is not None:
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        params = list(self.f.parameters()) + list(self.g.parameters())
        optim = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, factor=0.5, patience=5, verbose=False
        )
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        scaler = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None

        for ep in range(epochs):
            self.f.train(); self.g.train()
            train_loss_acc = 0.0
            n_train = 0

            for batch in train_loader:
                # support both (x,y,u) and (x,y)
                if len(batch) == 3:
                    x, y_real, u_k = batch
                else:
                    x, y_real = batch
                    # try to extract u from last nu columns of x
                    u_k = x[:, -self.nu]

                x = x.to(device); y_real = y_real.to(device); u_k = u_k.to(device)

                optim.zero_grad()
                if scaler:
                    with torch.cuda.amp.autocast():
                        y_pred = self.f(x).squeeze() + self.g(x).squeeze() * u_k.squeeze()
                        loss = loss_fn(y_pred, y_real.view_as(y_pred))
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    y_pred = self.f(x).squeeze() + self.g(x).squeeze() * u_k.squeeze()
                    loss = loss_fn(y_pred, y_real.view_as(y_pred))
                    loss.backward()
                    optim.step()

                batch_n = x.size(0)
                train_loss_acc += loss.item() * batch_n
                n_train += batch_n

            train_loss = train_loss_acc / max(1, n_train)

            # Validation
            if val_loader is not None:
                self.f.eval(); self.g.eval()
                val_loss_acc = 0.0
                n_val = 0
                with torch.no_grad():
                    for batch in val_loader:
                        if len(batch) == 3:
                            x, y_real, u_k = batch
                        else:
                            x, y_real = batch
                            u_k = x[:, -self.nu]
                        x = x.to(device); y_real = y_real.to(device); u_k = u_k.to(device)
                        y_pred = self.f(x).squeeze() + self.g(x).squeeze() * u_k.squeeze()
                        loss = loss_fn(y_pred, y_real.view_as(y_pred))
                        batch_n = x.size(0)
                        val_loss_acc += loss.item() * batch_n
                        n_val += batch_n
                val_loss = val_loss_acc / max(1, n_val)
                scheduler.step(val_loss)

                print(f"Epoch {ep+1}/{epochs}  Train={train_loss:.6f}  Val={val_loss:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        "f": {k: v.cpu() for k, v in self.f.state_dict().items()},
                        "g": {k: v.cpu() for k, v in self.g.state_dict().items()},
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        break
            else:
                print(f"Epoch {ep+1}/{epochs}  Train={train_loss:.6f}")

        # Load best weights if available
        if best_state is not None:
            self.f.load_state_dict(best_state["f"])
            self.g.load_state_dict(best_state["g"])

# ---------------------------
# TEST FORWARD
# ---------------------------
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
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
    u_hist_list, y_hist_list = [0]*len(workspace.plant["num_disc"]), [0]*(len(workspace.plant["den_disc"])-1)
    t = np.linspace(0, 50, int(50/workspace.dt))
    u= 5*np.sin(1*t)
    y = np.zeros_like(t)
    for i in range(len(t)):
        y[i] = utils.plant_response(workspace.plant["num_disc"], workspace.plant["den_disc"], u_hist_list, y_hist_list)
        y_hist_list = [y[i]] + y_hist_list[:-1]; u_hist_list = [u[i]] + u_hist_list[:-1]

    print("Building dataset from generated data...")
    X, Y, U = build_narma_dataset(y, u, ny=4, nu=4)

    # 70% train, 15% val, 15% test
    X_train, X_tmp, Y_train, Y_tmp, U_train, U_tmp = train_test_split(
        X.numpy(), Y.numpy(), U.numpy(), test_size=0.30, shuffle=True
    )

    X_val, X_test, Y_val, Y_test, U_val, U_test = train_test_split(
        X_tmp, Y_tmp, U_tmp, test_size=0.50, shuffle=True
    )

    # tạo TensorDataset with u included so dataloader yields (x,y,u)
    train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float(), torch.tensor(U_train).float())
    val_ds   = TensorDataset(torch.tensor(X_val).float(),   torch.tensor(Y_val).float(),   torch.tensor(U_val).float())
    test_ds  = TensorDataset(torch.tensor(X_test).float(),  torch.tensor(Y_test).float(),  torch.tensor(U_test).float())

    default_controller.train_narma(train_ds, val_data=val_ds, epochs=100, lr=1e-2, batch_size=32, use_amp=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_controller.f.to(device)
    default_controller.g.to(device)

    print("\nEvaluating on Test Dataset...")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    y_pred_test = []
    y_real_test = []

    with torch.no_grad():
        for x, y_real, u_k in test_loader:
            x = x.to(device); y_real = y_real.to(device); u_k = u_k.to(device)
            y_hat = default_controller.f(x).squeeze() + default_controller.g(x).squeeze() * u_k.squeeze()
            y_pred_test.append(y_hat.item())
            y_real_test.append(y_real.item())

    y_pred_test = np.array(y_pred_test)
    y_real_test = np.array(y_real_test)

    test_mse  = np.mean((y_real_test - y_pred_test)**2)
    test_rmse = np.sqrt(test_mse)
    test_mae  = np.mean(np.abs(y_real_test - y_pred_test))

    print("\n===== TEST SET PERFORMANCE =====")
    print(f"Test MSE  = {test_mse:.6f}")
    print(f"Test RMSE = {test_rmse:.6f}")
    print(f"Test MAE  = {test_mae:.6f}")
    print("================================\n")

    ny, nu = default_controller.ny, default_controller.nu
    delay = max(ny, nu)
    y_hist_seq, u_hist_seq = list(y[:delay]), list(u[:delay])
    y_pred = []

    for k in range(delay, len(y)):
        y_hist_t = torch.tensor(y_hist_seq[-ny:], dtype=torch.float32)
        u_hist_t = torch.tensor(u_hist_seq[-nu:], dtype=torch.float32)
        y_pred.append(default_controller.narma_forward(torch.cat([y_hist_t, u_hist_t]), u[k], device=device).item())
        y_hist_seq.append(y[k]); u_hist_seq.append(u[k])

    y_pred_np, y_real_np, t_plot = np.array(y_pred), y[delay:], t[delay:]

    # ---- Plot 1: y_real vs y_pred ----
    plt.figure(figsize=(10, 4))
    plt.plot(t_plot, y_real_np, label="y_real")
    plt.plot(t_plot, y_pred_np, label="y_pred", linestyle="--")
    plt.xlabel("Time"); plt.ylabel("y")
    plt.title("NARMA-L2 Model Prediction vs Real Data")
    plt.legend(); plt.grid(True); plt.show()
