#!/usr/bin/env python3
"""
End-to-End Pipeline connecting PROJECTS/BioInf/new_proj benchmark GRNs
with the new graph-tool Gene Expression Bayesian Inference extension (`gene_expr`).

Pipeline Stages:
  1. Network Definition: 10-Gene Benchmark GRN with negative feedback loops (from BioInf/new_proj).
  2. Trajectory Creation: Non-equilibrium Euler-Maruyama SDE simulation of gene expression.
  3. Graph-Tool Bayesian Inference: Joint SBM + dynamical parameter MCMC sweeps using GeneExprBlockState.
  4. Evaluation & Metrics: Structure reconstruction metrics against ground-truth W_true.
"""

import os
import sys
import numpy as np

# Add project directory to sys.path to load gene_expr extension
_proj_dir = os.path.dirname(os.path.abspath(__file__))
if _proj_dir not in sys.path:
    sys.path.insert(0, _proj_dir)

import graph_tool.all as gt
from gene_expr import GeneExprBlockState

# =========================================================================
# Stage 1: Network Definition (Benchmark 10-Gene GRN from BioInf/new_proj)
# =========================================================================
def get_benchmark_10gene_network():
    """
    Returns the benchmark 10-gene signed adjacency matrix W_true (10 x 10)
    with activating (+1) and repressing (-1) feedback loops as defined in
    PROJECTS/BioInf/new_proj/src/onsager_grn/dynamics.py.
    """
    W = np.array([
        [ 0,  0, -1,  0,  0,  0,  0,  0,  1,  0],  # Gene 1
        [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # Gene 2
        [ 0, -1,  0,  0,  0,  0,  0,  0,  0,  1],  # Gene 3
        [ 0,  0,  0,  0,  0, -1,  0,  0,  0,  0],  # Gene 4
        [ 0,  0,  0, -1,  0,  0,  0,  0,  0,  0],  # Gene 5
        [ 0,  0,  0,  0, -1,  0,  0,  0,  1,  0],  # Gene 6
        [ 1,  0,  0,  0,  0,  0,  0,  0,  0, -1],  # Gene 7
        [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0],  # Gene 8
        [ 0,  0,  0,  0,  0,  0,  0, -1,  0,  0],  # Gene 9
        [ 0,  0,  0,  0,  0,  0,  1,  1,  0, -1],  # Gene 10
    ], dtype=np.float32)
    W[:, 2] = 0.0  # knockout/zero intervention as in benchmark definition
    return W

# =========================================================================
# Stage 2: Trajectory Creation (Non-Equilibrium Euler-Maruyama SDE)
# =========================================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

def simulate_grn_sde_trajectory(W_true, T_total=500.0, dt=0.01, dt_meas=1.0,
                                d=0.5, V=2.0, noise_std=0.15, seed=42):
    """
    Simulates continuous gene expression trajectories via Euler-Maruyama SDE:
        dx_i(t) = [-d * x_i(t) + V * sigma(sum_j W_{ij} x_j(t))] dt + noise_std * dB_i(t)
    and sub-samples at interval dt_meas.
    """
    np.random.seed(seed)
    N = W_true.shape[0]
    n_steps = int(T_total / dt)
    meas_step = max(1, int(dt_meas / dt))
    
    x = np.random.uniform(0.5, 2.0, size=N).astype(np.float32)
    sqrt_dt_noise = noise_std * np.sqrt(dt)
    
    measured_trajectory = []
    
    for k in range(n_steps):
        if k % meas_step == 0:
            measured_trajectory.append(x.copy())
            
        x_clipped = np.maximum(x, 0.0)
        prod = sigmoid(W_true @ x_clipped)
        drift = -d * x_clipped + V * prod
        noise = np.random.normal(0.0, 1.0, size=N)
        x = np.maximum(x + dt * drift + sqrt_dt_noise * noise, 0.01)
        
    # Shape: (N, M) where M is the number of time measurements
    return np.array(measured_trajectory).T, dt_meas

def simulate_grn_sde_pairs(W_true, M=100, dt_sim=0.001, dt_sample=0.5, dt_meas=0.1,
                           d=0.5, V=2.0, noise_std=0.15, seed=42,
                           t_window=(0.0, 10.0)):
    """
    Simulates continuous gene expression trajectories via Euler-Maruyama SDE:
        dx_i(t) = [-d * x_i(t) + V * sigma(sum_j W_{ij} x_j(t))] dt + noise_std * dB_i(t)
    and harvests state-velocity observation pairs (x(t_m), x(t_m + dt_meas)) such that:
      1. Time intervals are decoupled:
         - dt_sim: Small numerical integration step (default 0.001) for accurate SDE simulation.
         - dt_sample: Interval between successive harvested candidate observation times along a trajectory.
         - dt_meas: Measurement interval used to estimate velocity/transition (x(t_m) -> x(t_m + dt_meas)).
      2. Invariant Base Distribution & Regime Control:
         - Samples are harvested uniformly across t_window = (t_start, t_end).
         - Enables isolating dynamical regimes: transient/relaxation (e.g. 0.0 to 4.0),
           stationary/equilibrium (e.g. 10.0 to 20.0), or mixed.
      3. Multi-Trajectory Spawning:
         - If M exceeds the number of sample points available in a single trajectory window,
           additional independent trajectories are generated from fresh initial conditions x(0) ~ U(0.5, 2.0).
         - Ensures that changing M or dt_meas never alters the state space distribution of sampled points.

    Returns:
        pairs: list of M numpy arrays of shape (N, 2), where column 0 is x(t_m) and column 1 is x(t_m + dt_meas).
        dt_meas: float, the measurement interval.
    """
    np.random.seed(seed)
    N = W_true.shape[0]
    t_start, t_end = t_window
    if t_end <= t_start:
        raise ValueError("t_window[1] (t_end) must be strictly greater than t_window[0] (t_start)")

    # Candidate observation times within t_window at step dt_sample
    sample_times = np.arange(t_start, t_end + 1e-9, dt_sample)
    if len(sample_times) == 0:
        sample_times = np.array([t_start])
    K = len(sample_times)

    # Number of independent trajectories required to harvest M pairs
    n_trajectories = int(np.ceil(M / K))

    # We integrate each trajectory up to T_max = t_end + dt_meas
    T_max = t_end + dt_meas
    n_steps = int(np.round(T_max / dt_sim))
    sqrt_dt_noise = noise_std * np.sqrt(dt_sim)

    # Precompute step indices where sample_times and sample_times + dt_meas occur
    sample_step_indices = [int(np.round(t / dt_sim)) for t in sample_times]
    next_step_indices = [int(np.round((t + dt_meas) / dt_sim)) for t in sample_times]

    all_pairs = []

    for traj_idx in range(n_trajectories):
        # Deterministic per-trajectory random generator ensures 100% invariance
        # of base states across different dt_meas and different M values.
        rng = np.random.RandomState(seed + traj_idx * 10007)
        x = rng.uniform(0.5, 2.0, size=N).astype(np.float64)
        # Store trajectory history at required step indices
        history = {}
        history[0] = x.copy()

        for step in range(1, n_steps + 1):
            x_clipped = np.maximum(x, 0.0)
            prod = sigmoid(W_true @ x_clipped)
            drift = -d * x_clipped + V * prod
            noise = rng.normal(0.0, 1.0, size=N)
            x = np.maximum(x + dt_sim * drift + sqrt_dt_noise * noise, 0.01)

            if step in sample_step_indices or step in next_step_indices:
                history[step] = x.copy()

        for s_idx, n_idx in zip(sample_step_indices, next_step_indices):
            pair = np.column_stack([history[s_idx], history[n_idx]])
            all_pairs.append(pair)
            if len(all_pairs) == M:
                break
        if len(all_pairs) == M:
            break

    return all_pairs, dt_meas

# =========================================================================
# Stage 3: Evaluation Metrics (Structure Reconstruction vs Ground Truth)
# =========================================================================
def compute_reconstruction_metrics(W_true, W_inferred):
    """
    Computes Precision, Recall (TPR), FPR, and Structural Hamming Distance (SHD)
    on binary edge presence (ignoring diagonal self-loops).
    """
    N = W_true.shape[0]
    mask = ~np.eye(N, dtype=bool)  # ignore diagonal
    
    gt_edges = (W_true[mask] != 0).astype(int)
    inf_edges = (W_inferred[mask] != 0).astype(int)
    
    tp = np.sum((gt_edges == 1) & (inf_edges == 1))
    fp = np.sum((gt_edges == 0) & (inf_edges == 1))
    fn = np.sum((gt_edges == 1) & (inf_edges == 0))
    tn = np.sum((gt_edges == 0) & (inf_edges == 0))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0
    shd = fp + fn
    
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "TPR (Recall)": tpr, "FPR": fpr,
        "Precision": precision, "F1-Score": f1,
        "SHD": shd
    }

# =========================================================================
# Main Pipeline Execution
# =========================================================================
def main():
    print("=" * 72)
    print("ONSAGER-GRN -> GRAPH-TOOL BAYESIAN INFERENCE PIPELINE")
    print("=" * 72)

    # 1. Network Definition
    print("\n[Stage 1] Loading Benchmark 10-Gene GRN Topology (BioInf/new_proj)...")
    W_true = get_benchmark_10gene_network()
    N = W_true.shape[0]
    n_true_edges = int((W_true != 0).sum())
    print(f"    - Network nodes (genes): {N}")
    print(f"    - Ground-truth directed edges: {n_true_edges} (Density: {n_true_edges/(N*(N-1)):.2f})")

    # 2. Trajectories Creation
    print("\n[Stage 2] Simulating Non-Equilibrium Euler-Maruyama SDE Trajectories...")
    T_total, dt, dt_meas = 100.0, 0.01, 0.5
    s, dt_meas = simulate_grn_sde_trajectory(
        W_true, T_total=T_total, dt=dt, dt_meas=dt_meas,
        d=0.5, V=2.0, noise_std=0.15, seed=424
    )
    print(f"    - Trajectory observation tensor shape: {s.shape} (N={s.shape[0]}, M={s.shape[1]})")
    print(f"    - Sub-sampling measurement interval dt_meas = {dt_meas}")

    # 3. Graph-Tool Bayesian Inference
    print("\n[Stage 3] Instantiating GeneExprBlockState (graph-tool C++ extension)...")
    state = GeneExprBlockState(s, dt=dt_meas)
    S_init = state.entropy()
    print(f"    - Initial posterior description length (entropy): {S_init:.4f}")

    print("\n[Stage 4] Running Bayesian MCMC Sweeps (Network Structure + Parameters)...")
    n_sweeps = 30
    best_S = S_init
    best_graph = None

    for sweep in range(1, n_sweeps + 1):
        dS, nmoves = state.mcmc_sweep(niter=10)
        S_cur = state.entropy()
        if S_cur < best_S or best_graph is None:
            best_S = S_cur
            best_graph = state.get_graph().copy()
        if sweep % 5 == 0 or sweep == 1:
            print(f"    - Sweep {sweep:2d}/{n_sweeps}: Entropy = {S_cur:8.4f} (dS = {dS:8.4f}, moves = {nmoves:4d})")

    # 4. Extract Inferred Graph Adjacency Matrix
    print("\n[Stage 5] Evaluating Reconstructed Network Structure...")
    W_inferred = np.zeros((N, N), dtype=int)
    for e in best_graph.edges():
        src, tgt = int(e.source()), int(e.target())
        if src != tgt: # record directed interaction
            W_inferred[tgt, src] = 1

    n_inf_edges = int(W_inferred.sum())
    print(f"    - Best inferred posterior entropy: {best_S:.4f}")
    print(f"    - Total inferred directed edges:   {n_inf_edges}")

    metrics = compute_reconstruction_metrics(W_true, W_inferred)
    print("\n[Stage 6] Reconstruction Benchmark Performance vs Ground Truth:")
    print(f"    - True Positives (TP):  {metrics['TP']}")
    print(f"    - False Positives (FP): {metrics['FP']}")
    print(f"    - False Negatives (FN): {metrics['FN']}")
    print(f"    - True Positive Rate:   {metrics['TPR (Recall)']:.3f}")
    print(f"    - False Positive Rate:  {metrics['FPR']:.3f}")
    print(f"    - Precision:            {metrics['Precision']:.3f}")
    print(f"    - F1-Score:             {metrics['F1-Score']:.3f}")
    print(f"    - Structural Hamming:   {metrics['SHD']}")

    print("\n=" * 72)
    print("SUCCESS: Full pipeline (Network -> Trajectory -> Graph-Tool Inference) executed!")
    print("=" * 72)

if __name__ == "__main__":
    main()
