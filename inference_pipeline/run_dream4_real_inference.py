import numpy as np
import pandas as pd
import graph_tool.all as gt
import sys
import os

# Add parent directory to sys.path to import local modules
sys.path.append(os.path.abspath('..'))

from inference_pipeline.inference.mcmc_engine import run_burn_in, run_posterior_sampling, compute_ensemble_stats
from inference_pipeline.inference.metrics import compute_reconstruction_metrics, plot_inferred_vs_true_am
from gene_expr import GeneExprBlockState

def load_dream4_timeseries(filepath, n_perturbations=5, timepoints=21):
    df = pd.read_csv(filepath, sep='\t')
    data = df.values
    s = []
    
    for i in range(n_perturbations):
        start_idx = i * timepoints
        for t in range(timepoints - 1):
            x_t = data[start_idx + t]
            x_t_next = data[start_idx + t + 1]
            # Ensure no zeroes for the log-normal / square-root noise formulation
            x_t = np.maximum(x_t, 0.01)
            x_t_next = np.maximum(x_t_next, 0.01)
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
    
    # Load Real DREAM4 data
    ts_filepath = '/tmp/BSBL-TSK-FS_code/insilico_size10_1_timeseries.tsv'
    gold_filepath = 'data/DREAM4/DREAM4_InSilico_Size10/gold/DREAM4_GoldStandard_InSilico_Size10_1.csv'
    
    s = load_dream4_timeseries(ts_filepath)
    W_true = load_dream4_gold_standard(gold_filepath)
    print(f"Loaded {len(s)} transition pairs from DREAM4 timeseries.")
    print(f"True network contains {int(W_true.sum())} edges.")
    
    print("\n--- 2. Setting Up Inference ---")
    g_init = gt.Graph(directed=True)
    g_init.add_vertex(N)
    
    # We rescale the time derivative since DREAM4 times are large (0, 50, 100).
    # The pipeline default dt_meas is 0.05. We can stick to 0.05 which acts as a 
    # re-scaling of the continuous time parameters.
    # We set dt_meas to 1.0 as a normalized time step.
    # We also initialize a few random edges to help MCMC escape the empty network local minimum.
    dt_meas = 1.0
    initial_theta = [-1.1262, -1.1414, -6.1079] # Estimated directly from DREAM4 trajectories
    state = GeneExprBlockState(s, dt=dt_meas, g=g_init, state_args=dict(deg_corr=True), theta=initial_theta)
    print(f"Initial description length (empty network): {state.entropy():.4f}")
    
    print("\n--- 3. Running MCMC Burn-in ---")
    state = run_burn_in(state, n_burnin=200, optimize_params=True)
    
    print("\n--- 4. Running Posterior Sampling ---")
    n_samples = 200
    W_ensemble, theta_ensemble = run_posterior_sampling(state, N, n_samples=n_samples, delta_k=1, optimize_params=True)
    
    print("\n--- 5. Evaluating and Comparing Edges ---")
    edge_probabilities, weight_mean, weight_std = compute_ensemble_stats(W_ensemble)
    
    W_inferred_last = W_ensemble[-1]
    
    metrics = compute_reconstruction_metrics(W_true, W_inferred_last)
    
    print("\n[ Reconstruction Metrics (Final Sample) ]")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    # Consensus Threshold
    W_ensemble_consensus = np.where(edge_probabilities > 0.01, 1.0, 0.0) # Lowered threshold just to see any weak edges found
    consensus_metrics = compute_reconstruction_metrics(W_true, W_ensemble_consensus)
    print("\n[ Reconstruction Metrics (Ensemble Consensus P > 0.01) ]")
    for k, v in consensus_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    os.makedirs('images', exist_ok=True)
    plot_inferred_vs_true_am(W_true, W_inferred_last, save_path="images/am_comparison_dream4_10node.png")
    
    # Save the matrices for visualization
    np.savez('dream4_10node_results.npz', W_true=W_true, edge_probabilities=edge_probabilities)
    
    # Generate a side-by-side heatmap plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(W_true, ax=axes[0], cmap='Greys', cbar=False, linewidths=.5, linecolor='gray')
    axes[0].set_title('True Edges (Binary)', fontsize=14)
    axes[0].set_xlabel('Source Gene')
    axes[0].set_ylabel('Target Gene')
    
    sns.heatmap(edge_probabilities, ax=axes[1], cmap='Blues', vmin=0, vmax=1, linewidths=.5, linecolor='gray')
    axes[1].set_title('Inferred Edge Probabilities (Ensemble)', fontsize=14)
    axes[1].set_xlabel('Source Gene')
    axes[1].set_ylabel('Target Gene')
    
    plt.tight_layout()
    plt.savefig('images/probabilistic_vs_true_heatmap.png', dpi=300)
    
    print("\nDone! Plots saved to images/ directory.")

if __name__ == "__main__":
    main()
