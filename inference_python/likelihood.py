import numpy as np

def sigmoid(m):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-np.clip(m, -30.0, 30.0)))

def node_log_likelihood(x, nx, m, theta_i, dt):
    """
    Computes the log-likelihood of transitions x -> nx for a single node i.
    
    Parameters:
    - x: Array of states at t (shape: M,)
    - nx: Array of states at t+dt (shape: M,)
    - m: Array of regulatory inputs m_i(t) = sum_j W_{ij} x_j(t) (shape: M,)
    - theta_i: Node parameters [log(gamma_i), log(V_i), log(D_i)]
    - dt: Time step
    
    Returns:
    - log_P: Total log-likelihood scalar over the M transitions.
    """
    lgamma, lV, lD = theta_i
    # Clip parameters to avoid numerical instability
    lgamma = max(-15.0, min(15.0, lgamma))
    lV = max(-15.0, min(15.0, lV))
    lD = max(-15.0, min(15.0, lD))
    
    gamma = np.exp(lgamma)
    V = np.exp(lV)
    D = np.exp(lD)
    
    # Ensure variables are safe
    x_safe = np.maximum(0.01, x)
    
    sig_m = sigmoid(m)
    
    # Drift mu
    mu = x + dt * (-gamma * x + V * sig_m)
    
    # Multiplicative CIR noise variance
    log_s = 0.5 * (lD + np.log(x_safe) + np.log(dt))
    s = np.exp(log_s)
    
    # Compute normal lpdf manually for speed:
    variance = s**2
    lpdf = -0.5 * np.log(2 * np.pi) - log_s - 0.5 * ((nx - mu)**2 / variance)
    
    # Sum over all time steps M
    total_log_P = np.sum(lpdf)
    
    if not np.isfinite(total_log_P) or np.abs(total_log_P) > 1e10:
        return -1e10
        
    return total_log_P

def compute_m(W_i, X):
    """
    Computes the regulatory input m_i for node i across all time steps.
    
    Parameters:
    - W_i: The i-th row of the weight matrix W (shape: N,)
    - X: The state matrix at time t (shape: N x M)
    
    Returns:
    - m: The regulatory input array (shape: M,)
    """
    return W_i @ X
