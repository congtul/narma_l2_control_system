from scipy import signal
import numpy as np
import torch
import torch.nn as nn

def discretize_tf(num_cont, den_cont, dt):
    """Chuyển TF liên tục sang difference equation rời rạc"""
    # Normalize trước
    num_cont = np.array(num_cont, dtype=np.float64)
    den_cont = np.array(den_cont, dtype=np.float64)

    num_cont = num_cont / den_cont[0]
    den_cont = den_cont / den_cont[0]

    # Rời rạc
    sysc = signal.TransferFunction(num_cont, den_cont)
    sysd = sysc.to_discrete(dt, method='zoh')

    num_disc = np.squeeze(sysd.num)
    den_disc = np.squeeze(sysd.den)

    return num_disc, den_disc

def plant_response(num_disc, den_disc, u_hist, y_hist):
    """
    Tính y[n] từ difference eq. 
    u_hist: list chứa [u[n], u[n-1], ..., u[n-k]]
    y_hist: list chứa [y[n-1], y[n-2], ..., y[n-k]]
    """
    return -np.sum(den_disc[1:] * np.array(y_hist)) + np.sum(num_disc * np.array(u_hist))

def load_weights_from_file(controller: nn.Module, file_path: str):
    """
    Load weights từ file .pth vào NARMA_L2_Controller hoặc tương tự
    file_path: chứa dict {'f': state_dict_f, 'g': state_dict_g}
    """
    state_dict = torch.load(file_path, map_location="cpu")  # tránh lỗi GPU nếu chưa set
    if 'f' in state_dict:
        controller.f.load_state_dict(state_dict['f'])
    else:
        raise KeyError("File weight không có key 'f'")
    
    if 'g' in state_dict:
        controller.g.load_state_dict(state_dict['g'])
    else:
        raise KeyError("File weight không có key 'g'")
    
    return controller

def init_weights_from_arrays(model: nn.Module, weight_dict: dict) -> nn.Module:
    linear_idx = 1
    with torch.no_grad():
        for layer in model.net:
            if isinstance(layer, nn.Linear):
                w_key = f"{linear_idx}.weight"
                b_key = f"{linear_idx}.bias"
                if w_key in weight_dict:
                    layer.weight.copy_(weight_dict[w_key])
                if b_key in weight_dict:
                    layer.bias.copy_(weight_dict[b_key])
                linear_idx += 1
    return model

def generate_random_control_signal_sequence(u_min, u_max,
                                            interval_min, interval_max,
                                            t_array):
    """
    Sinh chuỗi tín hiệu điều khiển dạng 'random hold'.

    Parameters
    ----------
    u_min : float  - giá trị nhỏ nhất của u
    u_max : float  - giá trị lớn nhất của u
    interval_min : float - thời gian giữ tối thiểu (s)
    interval_max : float - thời gian giữ tối đa (s)
    t_array : ndarray - mảng thời gian (phải đều nhau)

    Returns
    -------
    u_signal : ndarray - chuỗi tín hiệu u(t)
    """

    dt = t_array[1] - t_array[0]        # thời gian lấy mẫu
    n = len(t_array)

    u_signal = np.zeros(n)
    current_u = np.random.uniform(u_min, u_max)

    # thời gian còn lại để giữ giá trị
    hold_time_left = np.random.uniform(interval_min, interval_max)

    for i in range(n):
        if hold_time_left <= 0:
            # đổi giá trị u
            current_u = np.random.uniform(u_min, u_max)
            hold_time_left = np.random.uniform(interval_min, interval_max)

        u_signal[i] = current_u
        hold_time_left -= dt

    return u_signal

# ---------------------------
# Dataset builder (vectorized)
# ---------------------------
def build_narma_dataset(y_data, u_data, ny=4, nu=4):
    """
    Build NARMA-L2 dataset without discarding initial transient.
    Missing past values are padded with zeros.

    Returns:
        X: (N, ny + nu)   -> contains y(k),...,y(k-ny+1), u(k-1),...,u(k-nu)
        Y: (N,)           -> target y(k+1)
        U: (N,)           -> actual u(k)
    """
    y = np.asarray(y_data, dtype=np.float32)
    u = np.asarray(u_data, dtype=np.float32)

    N = len(y)

    # Output will have samples from k=0 .. N-2 (since y(k+1) must exist)
    n_total = N - 1
    X = np.zeros((n_total, ny + nu), dtype=np.float32)
    Y = np.zeros((n_total,), dtype=np.float32)
    U = np.zeros((n_total,), dtype=np.float32)

    for k in range(n_total):     # k corresponds to predicting y(k+1)
        # ----- fill y-history: y(k), y(k-1), ..., y(k-ny+1)
        for i in range(ny):
            idx = k - i
            if idx >= 0:
                X[k, i] = y[idx]
            else:
                X[k, i] = 0.0  # pad missing history

        # ----- fill u-history: u(k-1), u(k-2), ..., u(k-nu)
        for j in range(nu):
            idx = k - 1 - j
            if idx >= 0:
                X[k, ny + j] = u[idx]
            else:
                X[k, ny + j] = 0.0

        # Target output y(k+1)
        Y[k] = y[k + 1]

        # Ground-truth u(k) to train g() correctly
        U[k] = u[k]

    return (
        torch.from_numpy(X).float(),
        torch.from_numpy(Y).float(),
        torch.from_numpy(U).float(),
    )
