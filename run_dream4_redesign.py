import numpy as np
import pandas as pd
import graph_tool.all as gt
import sys
import os
from scipy.interpolate import CubicSpline

# Ensure local imports work
sys.path.append(os.path.abspath('.'))

from inference_pipeline.inference.mcmc_engine import run_burn_in, run_posterior_sampling, compute_ensemble_stats
from inference_pipeline.inference.metrics import compute_reconstruction_metrics, plot_probabilistic_network
from gene_expr import GeneExprBlockState

def load_dream4_timeseries_redesigned(filepath, n_perturbations=5, timepoints=21, 
                                      log_transform=False, upsample_factor=10):
    """
    Loads DREAM4 data with transformations:
    1. Optional Log Transform
    2. Global Min-Max Normalization to [0.1, 1.0]
    3. Spline Upsampling to reduce dt
    """
    df = pd.read_csv(filepath, sep='\t')
    data = df.values
    
    # 1. Log Transform
    if log_transform:
        data = np.log1p(data) # log(1 + x)
        
    # 2. Global Min-Max Normalization -> [0.1, 1.0]
    d_min = data.min()
    d_max = data.max()
    data = 0.1 + 0.9 * (data - d_min) / (d_max - d_min)
    
    s = []
    
    # Original time points are 0, 50, 100 ... 1000 (length 21)
    t_original = np.linspace(0, 1000, timepoints)
    
    # Upsampled time points
    new_timepoints = (timepoints - 1) * upsample_factor + 1
    t_upsampled = np.linspace(0, 1000, new_timepoints)
    
    for i in range(n_perturbations):
        start_idx = i * timepoints
        traj = data[start_idx : start_idx + timepoints]
        
        # 3. Cubic Spline Upsampling
        if upsample_factor > 1:
            cs = CubicSpline(t_original, traj)
            traj_upsampled = cs(t_upsampled)
            
            # Splines can overshoot/undershoot, clip back to [0.01, 1.0]
            traj_upsampled = np.clip(traj_upsampled, 0.01, 1.0)
        else:
            traj_upsampled = traj
            
        # Form state-transition pairs
        for t in range(len(traj_upsampled) - 1):
            x_t = traj_upsampled[t]
            x_t_next = traj_upsampled[t + 1]
            s.append(np.column_stack([x_t, x_t_next]))
            
    return s

def load_dream4_gold_standard(filepath, N=10):
    gold_df = pd.read_csv(filepath, header=None, names=['G_source', 'G_target', 'edge'])
    W_true = np.zeros((N, N))
    
    for _, row in gold_df.iterrows():
        src = int(row['G_source'].replace('G', '')) - 1
        tgt = int(row['G_target'].replace('G', '')) - 1
        W_true[tgt, src] = row['edge']
        
    return W_true

def main():
    print("--- 1. Loading DREAM4 10-Node Data ---")
    N = 10
    
    ts_filepath = 'data/DREAM4/BSBL-TSK-FS_code/insilico_size10_1_timeseries.tsv'
    gold_filepath = 'data/DREAM4/DREAM4_InSilico_Size10/gold/DREAM4_GoldStandard_InSilico_Size10_1.csv'
    
    # Parameters for Redesign
    UPSAMPLE_FACTOR = 1
    LOG_TRANSFORM = False
    THETA_MODE = "global" # Learn node-specific d, V, D_base
    USE_EM_LIKELIHOOD = True
    
    s = load_dream4_timeseries_redesigned(ts_filepath, log_transform=LOG_TRANSFORM, upsample_factor=UPSAMPLE_FACTOR)
    W_true = load_dream4_gold_standard(gold_filepath)
    print(f"Loaded {len(s)} transition pairs (Upsampled {UPSAMPLE_FACTOR}x).")
    print(f"True network contains {int(W_true.sum())} edges.")
    
    print("\n--- 2. Setting Up Inference ---")
    g_init = gt.Graph(directed=True)
    g_init.add_vertex(N)
    
    # Normalized time step. If original dt=50, upsampled 10x means actual dt=5. 
    # But we can just use 1.0 as a baseline and let theta absorb the scale.
    dt_meas = 1.0 
    
    # Initial theta (we start with uninformative generic values, the local optimizer will find the best ones)
    initial_theta = [0.0, 0.0, np.log(0.05)]
    
    state = GeneExprBlockState(s, dt=dt_meas, g=g_init, state_args=dict(deg_corr=False), 
                               theta=initial_theta, use_em_likelihood=USE_EM_LIKELIHOOD, self_loops=False)
    print(f"Initial description length (empty network): {state.entropy():.4f}")
    
    print(f"Theta Mode: {THETA_MODE}")
    print(f"Likelihood: {'Euler-Maruyama' if USE_EM_LIKELIHOOD else 'Standard'}")
    
    print("\n--- 3. Running MCMC Burn-in ---")
    state = run_burn_in(state, n_burnin=200, theta_mode=THETA_MODE)
    
    print("\n--- 4. Running Posterior Sampling ---")
    n_samples = 200
    W_ensemble, theta_ensemble = run_posterior_sampling(state, N, n_samples=n_samples, delta_k=1, theta_mode=THETA_MODE)
    
    print("\n--- 5. Evaluating and Comparing Edges ---")
    edge_probabilities, weight_mean, weight_std = compute_ensemble_stats(W_ensemble)
    
    W_inferred_last = W_ensemble[-1]
    
    metrics_final = compute_reconstruction_metrics(W_true, W_inferred_last)
    print("\n[ Reconstruction Metrics (Final Sample) ]")
    for k, v in metrics_final.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    # Ensemble Thresholding (Edges with > 0.1 probability)
    W_ensemble_consensus = (edge_probabilities > 0.1).astype(int)
    metrics_ensemble = compute_reconstruction_metrics(W_true, W_ensemble_consensus)
    print("\n[ Reconstruction Metrics (Ensemble Consensus P > 0.1) ]")
    for k, v in metrics_ensemble.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    os.makedirs('images', exist_ok=True)
    plot_probabilistic_network(W_true, edge_probabilities*weight_mean, save_path="images/am_comparison_dream4_redesign.png")

    
    # Save the matrices for visualization
    np.savez('dream4_redesign_results.npz', W_true=W_true, edge_probabilities=edge_probabilities)
    print("\nDone! Plots saved to images/ directory.")

if __name__ == "__main__":
    main()
