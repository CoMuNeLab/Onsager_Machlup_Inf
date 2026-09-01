#!/usr/bin/env python3
"""
Node-Wise Analysis of GRN Inference.
Generates networks of various topologies, simulates trajectories in different regimes,
and evaluates the Graph-Tool Bayesian inference performance on a *per-node* basis.
Correlates node-wise trajectory information content (Var, DriftSq, TDMI, eSNR) 
with node-wise inference success (AUROC, AUPRC, MSE, SHD) for incoming edges.
"""

import os
import sys
import time
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, average_precision_score, mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from scipy.signal import savgol_filter

import graph_tool.all as gt
from itertools import product
from tqdm import tqdm

_proj_dir = os.path.dirname(os.path.abspath(__file__))
if _proj_dir not in sys.path:
    sys.path.insert(0, _proj_dir)

from gene_expr import GeneExprBlockState
from analysis.run_onsager_graph_tool_pipeline import simulate_grn_sde_pairs, sigmoid

def generate_network(N, topo_type):
    """Generates a random signed adjacency matrix based on topology type."""
    if topo_type == "ER":
        p = min(1.0, 3.0 / N)
        W = np.random.choice([0, 1, -1], size=(N, N), p=[1-p, p/2, p/2]).astype(np.float32)
        np.fill_diagonal(W, 0)
    elif topo_type == "BA":
        g = gt.price_network(N, m=2, directed=True)
        W = np.zeros((N, N), dtype=np.float32)
        for e in g.edges():
            s, t = int(e.source()), int(e.target())
            W[t, s] = np.random.choice([-1, 1])
    elif topo_type == "SBM":
        W = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if i != j:
                    p = 6.0 / N if ((i < N//2 and j < N//2) or (i >= N//2 and j >= N//2)) else 0.5 / N
                    if np.random.rand() < p:
                        W[i, j] = np.random.choice([-1, 1])
    else:
        raise ValueError("Unknown topology")
    return W

def compute_node_trajectory_metrics(s, W_true, dt_meas, d=0.5, V=2.0, noise_std=0.15):
    """
    Computes node-wise continuous physics and information theoretic metrics.
    s is list of (N, 2) arrays.
    Returns list of dicts (one per node).
    """
    N = s[0].shape[0]
    M = len(s)
    
    # Extract trajectory of base states x(t_m) and next states
    x_all = np.stack([p[:, 0] for p in s], axis=1) # shape (N, M)
    vel = np.stack([(p[:, 1] - p[:, 0]) / dt_meas for p in s], axis=1)
    
    metrics_per_node = []
    
    for i in range(N):
        x_i = x_all[i, :]
        v_i = vel[i, :]
        
        var_x = np.var(x_i)
        
        # Calculate drift
        prod = sigmoid(W_true[i, :] @ x_all) # shape (M,)
        drift_i = -d * x_i + V * prod
        mean_drift_sq = np.mean(drift_i ** 2)
        var_drift = np.var(drift_i)
        
        var_noise_rate = (noise_std ** 2) / dt_meas
        esnr_vel = var_drift / var_noise_rate if var_noise_rate > 0 else 0.0
        
        # TDMI
        if M > 5 and var_x > 1e-10:
            tdmi = mutual_info_regression(x_i[:-1].reshape(-1, 1), x_i[1:], random_state=42)[0]
        else:
            tdmi = 0.0
            
        metrics_per_node.append({
            "var_x": var_x,
            "mean_drift_sq": mean_drift_sq,
            "esnr_vel": esnr_vel,
            "tdmi": tdmi
        })
        
    return metrics_per_node

def evaluate_node_inference(y_true, y_prob):
    """
    Evaluates inference performance for a single node's incoming edges.
    y_true: shape (N-1,)
    y_prob: shape (N-1,)
    """
    y_pred = (y_prob >= 0.5).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    shd = fp + fn
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0
    mse = mean_squared_error(y_true, y_prob)
    
    if len(np.unique(y_true)) > 1:
        fpr_roc, tpr_roc, _ = roc_curve(y_true, y_prob)
        auroc_val = float(auc(fpr_roc, tpr_roc))
        auprc_val = float(average_precision_score(y_true, y_prob))
    else:
        auroc_val = np.nan
        auprc_val = np.nan
        
    return {
        "AUROC": auroc_val,
        "AUPRC": auprc_val,
        "MSE": mse,
        "SHD": shd,
        "F1": f1
    }

def run_node_wise_experiment():
    print("=" * 80)
    print("NODE-WISE GRN INFERENCE EXPERIMENT")
    print("=" * 80)
    
    N = 30
    topologies = ["ER", "BA", "SBM"]
    M_values = [100, 500]
    dt_meas_values = [0.1, 0.5]
    windows = [
        ("Transient", (0.0, 5.0)),
        ("Stationary", (10.0, 25.0))
    ]
    n_replicates = 1  # Number of networks per condition to get robust statistics
    
    out_csv = os.path.join(_proj_dir, "results_node_wise_inference.csv")
    fieldnames = [
        "network_id", "topology", "regime", "M", "dt_meas", "node_id", "in_degree", "out_degree",
        "var_x", "mean_drift_sq", "esnr_vel", "tdmi",
        "AUROC", "AUPRC", "MSE", "SHD", "F1"
    ]
    
    with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        network_id = 0
        total_configs = len(topologies) * n_replicates * len(windows) * len(M_values) * len(dt_meas_values)
        pbar = tqdm(total=total_configs, desc="Running Configurations")
        
        for topo in topologies:
            for rep in range(n_replicates):
                W_true = generate_network(N, topo)
                network_id += 1
                
                # Precompute node degrees
                in_degrees = np.sum(W_true != 0, axis=1)
                out_degrees = np.sum(W_true != 0, axis=0)
                
                for win_label, win_interval in windows:
                    for M in M_values:
                        for dt_m in dt_meas_values:
                            
                            # Simulate
                            s, _ = simulate_grn_sde_pairs(
                                W_true, M=M, dt_sim=0.001, dt_sample=0.5, dt_meas=dt_m,
                                d=0.5, V=2.0, noise_std=0.15, seed=42+network_id*100, t_window=win_interval
                            )
                            
                            # Trajectory metrics
                            node_traj_metrics = compute_node_trajectory_metrics(
                                s, W_true, dt_meas=dt_m, d=0.5, V=2.0, noise_std=0.15
                            )
                            
                            # Inference
                            state = GeneExprBlockState(s, dt=dt_m)
                            edge_counts = np.zeros((N, N), dtype=float)
                            n_samples = 0
                            
                            for sweep in range(1, 36):
                                state.mcmc_sweep(niter=10)
                                if sweep > 10:
                                    g = state.get_graph()
                                    for e in g.edges():
                                        src, tgt = int(e.source()), int(e.target())
                                        if src != tgt:
                                            edge_counts[tgt, src] += 1.0
                                    n_samples += 1
                                    
                            P_edge = edge_counts / max(1, n_samples)
                            
                            # Node-wise evaluation
                            for i in range(N):
                                # Mask out diagonal (self-loops)
                                mask_i = np.arange(N) != i
                                y_true = (W_true[i, mask_i] != 0).astype(int)
                                y_prob = P_edge[i, mask_i]
                                
                                inf_metrics = evaluate_node_inference(y_true, y_prob)
                                
                                row = {
                                    "network_id": network_id,
                                    "topology": topo,
                                    "regime": win_label,
                                    "M": M,
                                    "dt_meas": dt_m,
                                    "node_id": i,
                                    "in_degree": in_degrees[i],
                                    "out_degree": out_degrees[i],
                                    "var_x": node_traj_metrics[i]["var_x"],
                                    "mean_drift_sq": node_traj_metrics[i]["mean_drift_sq"],
                                    "esnr_vel": node_traj_metrics[i]["esnr_vel"],
                                    "tdmi": node_traj_metrics[i]["tdmi"],
                                    "AUROC": inf_metrics["AUROC"],
                                    "AUPRC": inf_metrics["AUPRC"],
                                    "MSE": inf_metrics["MSE"],
                                    "SHD": inf_metrics["SHD"],
                                    "F1": inf_metrics["F1"]
                                }
                                writer.writerow(row)
                                
                            f.flush()
                            pbar.update(1)
                            
        pbar.close()
        print(f"Results saved to {out_csv}")

if __name__ == "__main__":
    run_node_wise_experiment()
