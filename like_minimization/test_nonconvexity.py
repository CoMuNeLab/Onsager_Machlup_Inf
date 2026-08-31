import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Add project root to sys.path
_proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_dir not in sys.path:
    sys.path.insert(0, _proj_dir)

from run_onsager_graph_tool_pipeline import (
    get_benchmark_10gene_network,
    simulate_grn_sde_pairs,
    sigmoid
)
from exp_data_trajectories import get_N_M_and_arrays

def generate_data():
    """Generates the data using the benchmark network."""
    W_true = get_benchmark_10gene_network()
    
    # Same parameters as exp_data_trajectories.py
    dt_meas = 0.2
    d = 0.5
    V = 2.0
    noise_std = 0.15
    M = 500
    
    print("Simulating trajectory data...")
    s, _ = simulate_grn_sde_pairs(
        W_true, M=M, dt_sim=0.001, dt_sample=0.5, dt_meas=dt_meas,
        d=d, V=V, noise_std=noise_std, seed=42, t_window=(0.0, 10.0)
    )
    
    N, M_actual, x_all, vel = get_N_M_and_arrays(s, dt_meas)
    
    # We want to fit: v_i = -d * x_i + V * sigmoid(W @ x)
    # Let Y be the target for the non-linear part: Y = v_i + d * x_i
    # Note: x_all has shape (N, M), vel has shape (N, M)
    Y = vel + d * x_all
    
    return W_true, x_all, Y, V

def objective_and_grad(W_flat, N, X, Y, V):
    """
    Computes the negative log-likelihood (MSE of the drift) and its gradient.
    W_flat: flattened weight matrix (N * N)
    """
    W = W_flat.reshape(N, N)
    
    # Forward pass
    Z = W @ X
    sig_Z = sigmoid(Z)
    Y_hat = V * sig_Z
    
    # Error
    err = Y_hat - Y
    
    # Loss: Sum of squared errors
    loss = 0.5 * np.sum(err**2)
    
    # Gradient
    # d(loss)/dW_ij = sum_t err_{i,t} * V * sig'(Z_{i,t}) * X_{j,t}
    sig_prime = sig_Z * (1.0 - sig_Z)
    grad_W = (err * V * sig_prime) @ X.T
    
    return loss, grad_W.flatten()

def run_minimization_trials(n_trials=10):
    W_true, X, Y, V = generate_data()
    N = W_true.shape[0]
    
    minima_results = []
    losses = []
    distances_to_true = []
    
    print(f"\nRunning {n_trials} trials with random initial conditions (Gradient Descent / L-BFGS-B)...")
    
    for i in range(n_trials):
        # Random initial weights in [-2, 2]
        np.random.seed(i * 100 + 42)
        W_init = np.random.uniform(-2, 2, size=(N, N)).flatten()
        
        # We use L-BFGS-B which is a highly efficient gradient-based optimizer
        res = minimize(
            fun=objective_and_grad,
            x0=W_init,
            args=(N, X, Y, V),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
        )
        
        W_opt = res.x.reshape(N, N)
        loss_opt = res.fun
        
        # Calculate distance from ground truth
        dist_to_true = np.linalg.norm(W_opt - W_true)
        
        minima_results.append(W_opt)
        losses.append(loss_opt)
        distances_to_true.append(dist_to_true)
        
        print(f"Trial {i+1:2d} | Loss: {loss_opt:.4f} | Dist to W_true: {dist_to_true:.4f} | Success: {res.success}")

    # --- Plotting the results ---
    fig = plt.figure(figsize=(20, 5))
    
    # 1. Scatter plot: Loss vs Distance to True W
    ax1 = plt.subplot(1, 4, 1)
    ax1.scatter(distances_to_true, losses, c='blue', s=100, alpha=0.7, edgecolor='k')
    ax1.set_xlabel("Distance to $W_{true}$ ($||W - W_{true}||_2$)")
    ax1.set_ylabel("Final Loss (Negative Log-Likelihood)")
    ax1.set_title("Local Minima: Loss vs. Distance")
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # Annotate points
    for i in range(n_trials):
        ax1.annotate(f"{i+1}", (distances_to_true[i], losses[i]), xytext=(5, 5), textcoords='offset points')

    # 2. Heatmap of True W
    ax2 = plt.subplot(1, 4, 2)
    vmax = max(np.max(np.abs(W_true)), np.max(np.abs(minima_results)))
    cax2 = ax2.imshow(W_true, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax2.set_title("Ground Truth Network ($W_{true}$)")
    fig.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)

    # 3. Heatmap of best loss W
    best_idx = np.argmin(losses)
    ax3 = plt.subplot(1, 4, 3)
    cax3 = ax3.imshow(minima_results[best_idx], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax3.set_title(f"Best Local Minimum (Trial {best_idx+1})\nDist: {distances_to_true[best_idx]:.1f}, Loss: {losses[best_idx]:.1f}")
    fig.colorbar(cax3, ax=ax3, fraction=0.046, pad=0.04)

    # 4. Heatmap of worst distance W
    worst_idx = np.argmax(distances_to_true)
    ax4 = plt.subplot(1, 4, 4)
    cax4 = ax4.imshow(minima_results[worst_idx], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax4.set_title(f"Worst Local Minimum (Trial {worst_idx+1})\nDist: {distances_to_true[worst_idx]:.1f}, Loss: {losses[worst_idx]:.1f}")
    fig.colorbar(cax4, ax=ax4, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_img = os.path.join(os.path.dirname(__file__), "nonconvexity_results_with_best.png")
    plt.savefig(out_img, dpi=300)
    print(f"\nSaved graphical results to {out_img}")

    # Analyze if we reached DIFFERENT minima
    print("\n--- Analysis of Non-Convexity ---")
    distances_between_minima = []
    for i in range(n_trials):
        for j in range(i+1, n_trials):
            dist = np.linalg.norm(minima_results[i] - minima_results[j])
            distances_between_minima.append(dist)
            
    avg_dist = np.mean(distances_between_minima)
    print(f"Average pairwise distance between found minima: {avg_dist:.4f}")
    
if __name__ == "__main__":
    run_minimization_trials(n_trials=10)
