from scipy import signal
import numpy as np

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