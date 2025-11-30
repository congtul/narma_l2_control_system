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