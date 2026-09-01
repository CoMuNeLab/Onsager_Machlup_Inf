#!/usr/bin/env python3
import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_python import GRNInferenceState, mcmc_sweep
from analysis.run_onsager_graph_tool_pipeline import get_benchmark_10gene_network, simulate_grn_sde_trajectory

def compute_metrics(W_true, W_pred):
    mask = ~np.eye(W_true.shape[0], dtype=bool)
    y_true = (W_true[mask] != 0).astype(int)
    y_pred = (W_pred[mask] != 0).astype(int)
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"TP": tp, "FP": fp, "FN": fn, "F1": f1}

def main():
    np.random.seed(42)
    print("=" * 60)
    print("TESTING PURE PYTHON MCMC GRN INFERENCE")
    print("=" * 60)
    
    W_true = get_benchmark_10gene_network()
    print("Simulating 10-gene trajectory...")
    s, dt_meas = simulate_grn_sde_trajectory(W_true, T_total=25.0, dt=0.01, dt_meas=0.5, seed=42)
    print(f"Trajectory shape: {s.shape}")
    
    # Format s as list of pairs for our Python state
    M_meas = s.shape[1]
    pairs = []
    for i in range(M_meas - 1):
        pair = np.column_stack([s[:, i], s[:, i+1]])
        pairs.append(pair)
        
    print(f"Number of transition pairs: {len(pairs)}")
    
    # Initialize state
    print("\nInitializing GRNInferenceState...")
    state = GRNInferenceState(pairs, dt=dt_meas)
    
    initial_S = state.get_entropy()
    print(f"Initial Entropy (Empty Graph): {initial_S:.4f}")
    
    print("\nRunning MCMC Sweeps...")
    n_sweeps = 15
    for sweep in range(1, n_sweeps + 1):
        # We start with greedy (high beta) for faster convergence in this test
        t0 = time.time()
        beta = min(10.0, sweep * 0.5) 
        dS, n_moves = mcmc_sweep(state, beta=beta, niter=1)
        S_cur = state.get_entropy()
        t1 = time.time()
        print(f"Sweep {sweep:2d}/{n_sweeps} | Beta: {beta:4.1f} | Moves: {n_moves:3d} | Entropy: {S_cur:10.4f} | Time: {t1-t0:.2f}s")
        
    print("\nExtracting Graph...")
    print(f"Final Categories K: {state.K}")
    print(f"Final Edges E: {state.E}")
    print("Final Weights:")
    print(state.W)
    
    metrics = compute_metrics(W_true, state.W)
    print("\nMetrics vs Ground Truth:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
        
    print("\nDONE!")

if __name__ == "__main__":
    main()
