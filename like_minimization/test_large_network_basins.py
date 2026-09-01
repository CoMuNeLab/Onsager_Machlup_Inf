import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import multiprocessing as mp
import time

_proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_dir not in sys.path:
    sys.path.insert(0, _proj_dir)

from analysis.run_onsager_graph_tool_pipeline import simulate_grn_sde_pairs, sigmoid
from analysis.exp_data_trajectories import get_N_M_and_arrays

def generate_random_network(N=100, density=0.05, seed=42):
    np.random.seed(seed)
    mask = (np.random.rand(N, N) < density).astype(float)
    np.fill_diagonal(mask, 0.0)
    signs = np.random.choice([-1, 1], size=(N, N))
    mags = np.random.uniform(0.5, 2.0, size=(N, N))
    W = mask * signs * mags
    return W

def objective_and_grad(W_flat, N, X, Y, V):
    W = W_flat.reshape(N, N)
    Z = W @ X
    sig_Z = sigmoid(Z)
    Y_hat = V * sig_Z
    err = Y_hat - Y
    loss = 0.5 * np.sum(err**2)
    sig_prime = sig_Z * (1.0 - sig_Z)
    grad_W = (err * V * sig_prime) @ X.T
    return loss, grad_W.flatten()

def run_single_trial(args):
    seed, N, X, Y, V = args
    np.random.seed(seed)
    # Initialize with slightly wider spread
    W_init = np.random.uniform(-1, 1, size=(N, N)).flatten()
    res = minimize(
        fun=objective_and_grad,
        x0=W_init,
        args=(N, X, Y, V),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 300, 'ftol': 1e-7, 'disp': False}
    )
    return res.x, res.fun, res.success

def main():
    N = 100
    M = 500
    dt_meas = 0.2
    d = 0.5
    V = 2.0
    noise_std = 0.15
    n_trials = 200

    print(f"Generating a random network with N={N}...")
    W_true = generate_random_network(N=N, density=0.05, seed=42)

    print("Simulating trajectory data...")
    s, _ = simulate_grn_sde_pairs(
        W_true, M=M, dt_sim=0.001, dt_sample=0.5, dt_meas=dt_meas,
        d=d, V=V, noise_std=noise_std, seed=100, t_window=(0.0, 10.0)
    )

    N_check, M_actual, x_all, vel = get_N_M_and_arrays(s, dt_meas)
    Y = vel + d * x_all

    print(f"Running {n_trials} trials in parallel (N={N} -> 10,000 parameters)...")
    start_time = time.time()
    
    args_list = [(42 + i, N, x_all, Y, V) for i in range(n_trials)]
    
    # Use half of the cores to not completely lock up the machine
    n_cores = max(1, mp.cpu_count() // 2)
    print(f"Using {n_cores} CPU cores...")
    
    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(run_single_trial, args_list)
        
    print(f"Optimization finished in {time.time() - start_time:.1f} seconds.")

    minima = np.array([r[0] for r in results])
    losses = np.array([r[1] for r in results])
    successes = np.array([r[2] for r in results])
    
    valid_idx = np.where(successes)[0]
    minima = minima[valid_idx]
    losses = losses[valid_idx]
    n_valid = len(valid_idx)
    
    print(f"Clustering {n_valid} successfully converged minima...")
    
    dist_matrix = pdist(minima, metric='euclidean')
    
    Z = linkage(dist_matrix, method='complete')
    # Use distance of 5.0 as threshold for 100x100 matrix 
    # (average element difference of 5.0 / 100 = 0.05)
    cluster_labels = fcluster(Z, t=5.0, criterion='distance')
    
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    n_clusters = len(unique_clusters)
    
    cluster_info = []
    for c, count in zip(unique_clusters, counts):
        idx = np.where(cluster_labels == c)[0]
        avg_loss = np.mean(losses[idx])
        cluster_info.append({'id': c, 'size': count, 'loss': avg_loss, 'idx': idx})
        
    cluster_info.sort(key=lambda x: x['size'], reverse=True)
    
    print(f"\n--- BASINS OF ATTRACTION ---")
    print(f"Found {n_clusters} distinct basins of attraction.")
    for i, c in enumerate(cluster_info[:15]):
        print(f"Basin {i+1:2d}: Size = {c['size']:3d} trials ({c['size']/n_valid*100:.1f}%), Avg Loss = {c['loss']:.2f}")
    if n_clusters > 15:
        print(f"... and {n_clusters - 15} more smaller basins.")
        
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: cluster size vs loss
    sizes = [c['size'] for c in cluster_info]
    avg_losses = [c['loss'] for c in cluster_info]
    
    ax1.scatter(avg_losses, sizes, alpha=0.7, edgecolors='k', s=80, c='coral')
    ax1.set_title(f"Distribution of Attraction Basins\n(N={N}, {n_valid} Trials)")
    ax1.set_xlabel("Average Basin Loss")
    ax1.set_ylabel("Number of Trials (Basin Size)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Right: CDF of basin sizes
    sorted_sizes = np.sort(sizes)[::-1]
    cumulative_coverage = np.cumsum(sorted_sizes) / n_valid * 100
    ax2.plot(range(1, len(sorted_sizes)+1), cumulative_coverage, 'o-', color='teal', linewidth=2)
    ax2.set_title("Cumulative Trial Coverage by Basin Rank")
    ax2.set_xlabel("Basin Rank (largest to smallest)")
    ax2.set_ylabel("Cumulative % of Trials")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    out_img = os.path.join(os.path.dirname(__file__), "large_network_basins.png")
    plt.savefig(out_img, dpi=300)
    print(f"\nSaved graphical results to {out_img}")

if __name__ == "__main__":
    main()
