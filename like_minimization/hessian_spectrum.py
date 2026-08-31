import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_dir not in sys.path:
    sys.path.insert(0, _proj_dir)

from run_onsager_graph_tool_pipeline import simulate_grn_sde_pairs, sigmoid
from exp_data_trajectories import get_N_M_and_arrays

def sigmoid_prime(z):
    # Clip z to avoid overflow in exp
    z = np.clip(z, -50, 50)
    sig = 1.0 / (1.0 + np.exp(-z))
    return sig * (1.0 - sig)

def sigmoid_double_prime(z):
    z = np.clip(z, -50, 50)
    sig = 1.0 / (1.0 + np.exp(-z))
    return sig * (1.0 - sig) * (1.0 - 2.0 * sig)

def generate_random_network(N, density=0.1, seed=42):
    np.random.seed(seed)
    mask = (np.random.rand(N, N) < density).astype(float)
    np.fill_diagonal(mask, 0.0)
    signs = np.random.choice([-1, 1], size=(N, N))
    mags = np.random.uniform(0.5, 2.0, size=(N, N))
    return mask * signs * mags

def compute_hessian_node(X, dt_meas, d, V, D, node_idx, W_eval):
    """
    Computes the Hessian matrix of the negative log-likelihood for a specific node,
    evaluated at a given weight vector W_eval.
    """
    N, M = X.shape
    H = np.zeros((N, N))
    
    for k in range(1, M):
        x_prev = X[:, k-1]
        x_curr = X[node_idx, k]
        
        # Total input to the sigmoid
        z = np.dot(W_eval, x_prev)
        
        # Drift and residual error e_k
        drift = -d * x_prev[node_idx] + V * sigmoid(z)
        e_k = x_curr - x_prev[node_idx] - dt_meas * drift
        
        # The coefficient c_k = \Delta t * V * (\sigma')^2 - e_k * \sigma''
        term1 = dt_meas * V * (sigmoid_prime(z)**2)
        term2 = e_k * sigmoid_double_prime(z)
        c_k = term1 - term2
        
        # Outer product of the states: x_{k-1} * x_{k-1}^T
        # Add to the Hessian accumulator
        H += c_k * np.outer(x_prev, x_prev)
        
    H *= (V / (2.0 * D * dt_meas)) # The 1/dt_meas comes from the 1/(4D*dt) in the likelihood definition
    return H

def main():
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    N = 50
    M = 1000
    dt_meas = 0.1
    d = 0.5
    V = 2.0
    noise_std = 0.15
    D = (noise_std**2) / 2.0

    print(f"Generating N={N} network and simulating M={M} time steps...")
    W_true = generate_random_network(N=N, density=0.1, seed=123)
    s, _ = simulate_grn_sde_pairs(
        W_true, M=M, dt_sim=0.005, dt_sample=dt_meas, dt_meas=dt_meas,
        d=d, V=V, noise_std=noise_std, seed=42, t_window=(0.0, M*dt_meas)
    )
    
    _, _, X, _ = get_N_M_and_arrays(s, dt_meas)
    
    # We analyze the Hessian with respect to the incoming edges of Node 0
    node_idx = 0
    W_node_true = W_true[node_idx, :]
    
    print("\nCalculating Hessian at TRUE global minimum weights...")
    H_true = compute_hessian_node(X, dt_meas, d, V, D, node_idx, W_node_true)
    eig_true = np.linalg.eigvalsh(H_true)
    print(f"Negative eigenvalues: {(eig_true < 0).sum()} / {N}")
    
    print("Calculating Hessian at ZERO weights (Typical Initialization)...")
    W_node_zero = np.zeros(N)
    H_zero = compute_hessian_node(X, dt_meas, d, V, D, node_idx, W_node_zero)
    eig_zero = np.linalg.eigvalsh(H_zero)
    print(f"Negative eigenvalues: {(eig_zero < 0).sum()} / {N}")
    
    print("Calculating Hessian at RANDOM incorrect weights...")
    np.random.seed(999)
    W_node_rand = np.random.uniform(-3, 3, size=N)
    H_rand = compute_hessian_node(X, dt_meas, d, V, D, node_idx, W_node_rand)
    eig_rand = np.linalg.eigvalsh(H_rand)
    print(f"Negative eigenvalues: {(eig_rand < 0).sum()} / {N}")
    
    # --- Plotting ---
    plt.figure(figsize=(16, 5))
    
    def plot_spectrum(ax, eigvals, title):
        neg_eigs = eigvals[eigvals < -1e-10] # Small tolerance for numerical zero
        pos_eigs = eigvals[eigvals >= -1e-10]
        
        if len(pos_eigs) > 0:
            ax.hist(pos_eigs, bins=30, color='teal', alpha=0.8, label=f'Convex (Positive)')
        if len(neg_eigs) > 0:
            ax.hist(neg_eigs, bins=15, color='crimson', alpha=0.8, label=f'Non-Convex (Negative)')
            
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Eigenvalue Magnitude", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.axvline(0, color='k', linestyle='--', linewidth=1.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    ax1 = plt.subplot(1, 3, 1)
    plot_spectrum(ax1, eig_true, "Spectrum at True Weights\n(Global Minimum)")
    
    ax2 = plt.subplot(1, 3, 2)
    plot_spectrum(ax2, eig_zero, "Spectrum at Zero Weights\n(Typical Initialization)")
    
    ax3 = plt.subplot(1, 3, 3)
    plot_spectrum(ax3, eig_rand, "Spectrum at Random Weights\n(Rough Terrain / Saddle Point)")
    
    plt.suptitle(f"Hessian Eigenvalue Spectrum (Node {node_idx}, N={N} dimensions)", fontsize=16, y=1.05)
    
    out_img = os.path.join(os.path.dirname(__file__), "hessian_spectrum.png")
    plt.tight_layout()
    plt.savefig(out_img, bbox_inches='tight', dpi=300)
    print(f"\nSaved empirical spectrum plot to {out_img}")

if __name__ == "__main__":
    main()
