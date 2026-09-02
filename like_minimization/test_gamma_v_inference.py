import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

_proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
analysis_dir = os.path.join(_proj_dir, 'analysis')
if _proj_dir not in sys.path:
    sys.path.insert(0, _proj_dir)
if analysis_dir not in sys.path:
    sys.path.insert(0, analysis_dir)

from run_onsager_graph_tool_pipeline import simulate_grn_sde_pairs, sigmoid
from exp_data_trajectories import get_N_M_and_arrays

def generate_scale_free_W(N, m=2, seed=42):
    np.random.seed(seed)
    edges = []
    for i in range(m):
        for j in range(i+1, m):
            edges.append((i, j))
    degrees = np.zeros(N)
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    for i in range(m, N):
        probs = degrees[:i] / np.sum(degrees[:i])
        targets = np.random.choice(i, size=m, replace=False, p=probs)
        for t in targets:
            edges.append((i, t))
            degrees[i] += 1
            degrees[t] += 1
    W = np.zeros((N, N))
    for u, v in edges:
        if np.random.rand() > 0.5:
            u, v = v, u
        sign = np.random.choice([-1, 1])
        mag = np.random.uniform(1.0, 3.0)
        W[u, v] = sign * mag
    return W

def simulate_heterogeneous_trajectories(W, T_total, dt, dt_meas, true_gamma, true_V, D_base):
    N = W.shape[0]
    n_steps = int(T_total / dt)
    n_meas = int(T_total / dt_meas)
    x = np.random.uniform(0.1, 1.0, size=N)
    x_history = []
    t_history = []
    meas_step = int(dt_meas / dt)
    
    for step in range(n_steps):
        t = step * dt
        drift = -true_gamma * x + true_V * sigmoid(W @ x)
        noise = np.sqrt(2 * D_base * np.maximum(x, 0.01)) * np.random.normal(0, np.sqrt(dt), size=N)
        x = x + drift * dt + noise
        x = np.maximum(x, 0.01) # keep strictly positive
        if step % meas_step == 0:
            x_history.append(x.copy())
            t_history.append(t)
    return np.array(t_history), np.array(x_history).T # return X of shape (N, M)

def objective_gamma_v(params, X_prev, X_curr, dt_meas, Z, D_base):
    # params = [lgamma, lV]
    gamma = np.exp(params[0])
    V = np.exp(params[1])
    
    drift = -gamma * X_prev + V * sigmoid(Z)
    e_k = X_curr - X_prev - dt_meas * drift
    
    # Weighted least squares from CIR noise likelihood
    loss = np.sum((e_k**2) / (4.0 * D_base * X_prev * dt_meas))
    return loss

def main():
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    N = 25
    T_total = 20.0
    dt = 0.001
    dt_meas = 0.05
    D_base = 0.1
    
    print("Generating network and true heterogeneous parameters...")
    W_true = generate_scale_free_W(N, m=2, seed=42)
    
    np.random.seed(123)
    true_gamma = np.random.lognormal(mean=np.log(0.5), sigma=0.5, size=N)
    true_V = np.random.lognormal(mean=np.log(2.0), sigma=0.5, size=N)
    
    print("Simulating rich trajectories...")
    t, X = simulate_heterogeneous_trajectories(W_true, T_total, dt, dt_meas, true_gamma, true_V, D_base)
    
    M = X.shape[1]
    inferred_gamma = np.zeros(N)
    inferred_V = np.zeros(N)
    
    print("Running Likelihood Maximization for gamma and V...")
    for i in range(N):
        X_prev = X[i, :-1]
        X_curr = X[i, 1:]
        Z = W_true[i, :] @ X[:, :-1]
        
        # Initialize at incorrect values (e.g., standard defaults)
        init_params = [np.log(1.0), np.log(1.0)] 
        
        res = minimize(
            fun=objective_gamma_v,
            x0=init_params,
            args=(X_prev, X_curr, dt_meas, Z, D_base),
            method='L-BFGS-B',
            bounds=[(-5, 3), (-5, 5)] # reasonable bounds for log parameters
        )
        
        inferred_gamma[i] = np.exp(res.x[0])
        inferred_V[i] = np.exp(res.x[1])
        
    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gamma Plot
    min_g, max_g = min(true_gamma.min(), inferred_gamma.min()), max(true_gamma.max(), inferred_gamma.max())
    ax1.scatter(true_gamma, inferred_gamma, color='teal', s=80, alpha=0.8, edgecolors='k')
    ax1.plot([min_g, max_g], [min_g, max_g], 'k--', lw=2, label="Perfect Inference")
    ax1.set_title(r"Inference of Decay Rates ($\gamma$)")
    ax1.set_xlabel(r"True $\gamma$")
    ax1.set_ylabel(r"Inferred $\gamma$")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    # V Plot
    min_v, max_v = min(true_V.min(), inferred_V.min()), max(true_V.max(), inferred_V.max())
    ax2.scatter(true_V, inferred_V, color='coral', s=80, alpha=0.8, edgecolors='k')
    ax2.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2, label="Perfect Inference")
    ax2.set_title("Inference of Max Transcription ($V$)")
    ax2.set_xlabel("True $V$")
    ax2.set_ylabel("Inferred $V$")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.suptitle("Likelihood Maximization with Known Network Topology\n(Convex Optimization)", fontsize=16, y=1.05)
    plt.tight_layout()
    
    out_img = os.path.join(os.path.dirname(__file__), "gamma_v_inference.png")
    plt.savefig(out_img, bbox_inches='tight', dpi=300)
    print(f"\nSaved inference scatter plot to {out_img}")
    
    mse_gamma = np.mean((true_gamma - inferred_gamma)**2)
    mse_V = np.mean((true_V - inferred_V)**2)
    print(f"MSE of gamma: {mse_gamma:.4f}")
    print(f"MSE of V: {mse_V:.4f}")

if __name__ == "__main__":
    main()
