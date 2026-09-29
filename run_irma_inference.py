import os
import sys
import pandas as pd
import numpy as np
import graph_tool.all as gt
import matplotlib.pyplot as plt

from inference_pipeline.inference.mcmc_engine import run_burn_in, run_posterior_sampling, compute_ensemble_stats
from inference_pipeline.inference.metrics import compute_reconstruction_metrics, plot_probabilistic_network
from gene_expr import GeneExprBlockState

def load_and_preprocess_irma(csv_path):
    df = pd.read_csv(csv_path)
    # The first column is time
    # Time values are 'Glc', 'Gal 0', '20', '40', ...
    # We want to drop 'Gal 0' to keep dt=20 constant.
    # Glc is t=0, then 20, 40, etc.
    df = df[df['Time_min'] != 'Gal 0']
    
    # Extract just the expression values as a numpy array
    genes = ['CBF1', 'GAL4', 'SWI5', 'GAL80', 'ASH1']
    expr = df[genes].values.astype(np.float64)
    
    # Min-Max Normalize to [0.1, 1.0]
    min_val = expr.min(axis=0)
    max_val = expr.max(axis=0)
    # Avoid division by zero
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    expr = 0.1 + 0.9 * (expr - min_val) / range_val
    
    # Form pairs (x_t, x_{t+20})
    pairs = []
    for i in range(len(expr) - 1):
        pair = np.column_stack([expr[i], expr[i+1]])
        pairs.append(pair)
        
    return pairs

def load_true_network(csv_path):
    df = pd.read_csv(csv_path)
    genes = ['CBF1', 'GAL4', 'SWI5', 'GAL80', 'ASH1']
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    
    N = len(genes)
    W_true = np.zeros((N, N))
    for _, row in df.iterrows():
        src = gene_to_idx[row['source']]
        tgt = gene_to_idx[row['target']]
        W_true[tgt, src] = row['type']
    return W_true

def run_inference(dataset_name, pairs, W_true):
    print(f"\n=============================================")
    print(f"Running Inference for: {dataset_name}")
    print(f"Number of trajectory pairs: {len(pairs)}")
    print(f"=============================================")
    
    N = W_true.shape[0]
    g_init = gt.Graph(directed=True)
    g_init.add_vertex(N)
    
    # Jump-start with a random edge
    g_init.add_edge(0, 1)
    
    # Initial kinetic parameters [log(gamma), log(V), log(D)]
    initial_theta = [np.log(0.1), np.log(1.0), np.log(0.01)]
    dt = 1.0
    
    print("Initializing Block State...")
    state = GeneExprBlockState(pairs, dt=dt, g=g_init, state_args=dict(deg_corr=True), theta=initial_theta)
    
    print("Running Burn-in (100 sweeps)...")
    state = run_burn_in(state, n_burnin=100, theta_mode="global")
    
    print("Running Posterior Sampling (200 sweeps)...")
    W_ensemble, theta_ensemble = run_posterior_sampling(state, N, n_samples=200, delta_k=1, theta_mode="global")
    
    edge_probabilities, weight_mean, weight_std = compute_ensemble_stats(W_ensemble)
    
    # Metrics
    W_inferred = W_ensemble[-1]
    metrics = compute_reconstruction_metrics(W_true, W_inferred)
    
    print(f"\nResults for {dataset_name}:")
    print(f"  F1-Score: {metrics['F1-Score']:.3f}")
    print(f"  TPR (Recall): {metrics['TPR (Recall)']:.3f} | FPR: {metrics['FPR']:.3f} | Precision: {metrics['Precision']:.3f}")
    print(f"  SHD: {metrics['SHD']}")
    
    # Save probabilistic network plot
    plot_name = dataset_name.replace(" ", "_").replace("+", "and").lower()
    plot_path = f"IRMA_dataset/{plot_name}_inferred_network.png"
    plot_probabilistic_network(W_true, edge_probabilities, save_path=plot_path)
    
    return metrics

if __name__ == "__main__":
    on_pairs = load_and_preprocess_irma('IRMA_dataset/irma_switch_on.csv')
    off_pairs = load_and_preprocess_irma('IRMA_dataset/irma_switch_off.csv')
    both_pairs = on_pairs + off_pairs
    
    W_true = load_true_network('IRMA_dataset/true_network.csv')
    
    run_inference("Switch-On Only", on_pairs, W_true)
    run_inference("Switch-Off Only", off_pairs, W_true)
    run_inference("Combined", both_pairs, W_true)
