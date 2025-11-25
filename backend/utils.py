from scipy import signal
import numpy as np

def discretize_tf(num_cont, den_cont, dt):
    """Chuyển TF liên tục sang difference equation rời rạc"""
    # Chuyển sang dạng lti của scipy
    sysc = signal.TransferFunction(num_cont, den_cont)
    sysd = sysc.to_discrete(dt, method='zoh')  # dùng zero-order hold
    # sysd.num, sysd.den là 2D array, flatten về 1D
    # y[n] = -sum(den_disc[k]*y[n-k-1]) + sum(num_disc[k]*u[n-k])
    num_disc = np.squeeze(sysd.num)
    den_disc = np.squeeze(sysd.den)
    return num_disc, den_disc