#!/usr/bin/env python3
"""
Large-Scale 600-Run Factorial Batch Trajectory Experiment for Continuous-Time GRN Inference.
Systematically explores:
  - Sample Size (M): 15 levels in [20, 2000]
  - Measurement Interval (dt_meas): 10 levels in [0.02, 2.00]
  - Continuous Trajectory Physics: 4 distinct trajectory windows spanning a wide range of
    expression variance Var(x), mean squared drift magnitude <|f(x)|^2>, and empirical SNR.

Every MCMC inference run flushes a comprehensive row to `results_batch_runs.csv`.
"""

import os
import sys
import csv
import time
import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score

# Add project root to sys.path
_proj_dir = os.path.dirname(os.path.abspath(__file__))
if _proj_dir not in sys.path:
    sys.path.insert(0, _proj_dir)

import graph_tool.all as gt
from gene_expr import GeneExprBlockState
from run_onsager_graph_tool_pipeline import (
    get_benchmark_10gene_network,
    simulate_grn_sde_pairs,
    compute_reconstruction_metrics,
    sigmoid
)

def compute_continuous_trajectory_physics(s, W_true, dt_meas, d=0.5, V=2.0, noise_std=0.15):
    """
    Computes a comprehensive vector of continuous physical & dynamical order parameters
    directly from the sampled trajectory pairs s:
      - mean_var_x: Average expression variance across genes Var(x)
      - mean_var_v: Average finite-difference velocity variance Var(v)
      - mean_drift_sq: Mean squared magnitude of the deterministic regulatory drift vector <|f(x)|^2>
      - var_drift: Spatial variance of the deterministic drift vector field
      - var_noise_rate: Brownian velocity noise variance rate (2 * D / dt_meas = noise_std^2 / dt_meas)
      - empirical_snr: Velocity SNR = var_drift / var_noise_rate
    """
    if isinstance(s, (list, tuple)) and len(s) > 0 and isinstance(s[0], np.ndarray):
        N = s[0].shape[0]
        M = len(s)
        x_all = np.stack([p[:, 0] for p in s], axis=1) # shape (N, M)
        vel = np.stack([(p[:, 1] - p[:, 0]) / dt_meas for p in s], axis=1)
    else:
        N, M = s.shape
        x_all = s
        vel = (s[:, 1:] - s[:, :-1]) / dt_meas if M > 1 else np.zeros_like(s)

    mean_var_x = float(np.mean(np.var(x_all, axis=1)))
    mean_var_v = float(np.mean(np.var(vel, axis=1))) if M > 1 else 0.0

    # Compute deterministic drift at each measured base state:
    # f(x) = -d * x + V * sigmoid(W_true @ x)
    drifts = np.zeros_like(x_all)
    for t in range(M):
        x = x_all[:, t]
        prod = sigmoid(W_true @ x)
        drifts[:, t] = -d * x + V * prod

    mean_drift_sq = float(np.mean(drifts ** 2))
    var_drift = float(np.mean(np.var(drifts, axis=1)))
    var_noise_rate = float((noise_std ** 2) / dt_meas)
    empirical_snr = float(var_drift / var_noise_rate) if var_noise_rate > 0 else 0.0

    return {
        "N": int(N),
        "M": int(M),
        "mean_var_x": round(mean_var_x, 6),
        "mean_var_v": round(mean_var_v, 6),
        "mean_drift_sq": round(mean_drift_sq, 6),
        "var_drift": round(var_drift, 6),
        "var_noise_rate": round(var_noise_rate, 6),
        "empirical_snr": round(empirical_snr, 6)
    }

def run_mcmc_inference(W_true, s, dt_meas, n_sweeps=35, burn_in=10):
    """
    Runs Graph-Tool GeneExprBlockState MCMC inference on sample set s.
    Returns AUROC, AUPRC, best entropy, prob threshold metrics (P_edge >= 0.5), and best graph metrics.
    """
    if isinstance(s, (list, tuple)):
        N = s[0].shape[0]
        M = len(s)
    else:
        N, M = s.shape

    start_t = time.time()
    state = GeneExprBlockState(s, dt=dt_meas)
    
    entropy_history = [state.entropy()]
    edge_counts = np.zeros((N, N), dtype=float)
    n_samples = 0
    
    best_S = entropy_history[0]
    best_graph = state.get_graph().copy()

    for sweep in range(1, n_sweeps + 1):
        dS, nmoves = state.mcmc_sweep(niter=10)
        S_cur = state.entropy()
        entropy_history.append(S_cur)
        
        if S_cur < best_S:
            best_S = S_cur
            best_graph = state.get_graph().copy()
            
        if sweep > burn_in:
            g = state.get_graph()
            for e in g.edges():
                src, tgt = int(e.source()), int(e.target())
                if src != tgt:
                    edge_counts[tgt, src] += 1.0
            n_samples += 1

    elapsed = time.time() - start_t
    P_edge = edge_counts / max(1, n_samples)
    
    # 1. Probabilistic Threshold Metrics (P_edge >= 0.5)
    W_pred_prob = (P_edge >= 0.5).astype(int)
    prob_metrics = compute_reconstruction_metrics(W_true, W_pred_prob)
    
    # 2. Best Entropy Graph Metrics
    W_pred_best = np.zeros((N, N), dtype=int)
    for e in best_graph.edges():
        src, tgt = int(e.source()), int(e.target())
        if src != tgt:
            W_pred_best[tgt, src] = 1
    best_metrics = compute_reconstruction_metrics(W_true, W_pred_best)
    
    # 3. Continuous ROC and PR Curves (off-diagonal edges only)
    mask = ~np.eye(N, dtype=bool)
    y_true_flat = (W_true[mask] != 0).astype(int)
    y_score_flat = P_edge[mask]
    
    if len(np.unique(y_true_flat)) > 1:
        fpr, tpr, _ = roc_curve(y_true_flat, y_score_flat)
        auroc_val = float(auc(fpr, tpr))
        auprc_val = float(average_precision_score(y_true_flat, y_score_flat))
    else:
        auroc_val, auprc_val = 0.5, 0.0

    return {
        "AUROC": round(auroc_val, 4),
        "AUPRC": round(auprc_val, 4),
        "best_entropy": round(float(best_S), 2),
        "F1_prob": round(float(prob_metrics["F1-Score"]), 4),
        "Precision_prob": round(float(prob_metrics["Precision"]), 4),
        "Recall_prob": round(float(prob_metrics["TPR (Recall)"]), 4),
        "SHD_prob": int(prob_metrics["SHD"]),
        "F1_best": round(float(best_metrics["F1-Score"]), 4),
        "Precision_best": round(float(best_metrics["Precision"]), 4),
        "Recall_best": round(float(best_metrics["TPR (Recall)"]), 4),
        "SHD_best": int(best_metrics["SHD"]),
        "elapsed_sec": round(float(elapsed), 2)
    }

def main():
    print("=" * 85, flush=True)
    print("LARGE-SCALE 600-RUN FACTORIAL BATCH EXPERIMENT FOR GRN INFERENCE", flush=True)
    print("=" * 85, flush=True)

    W_true = get_benchmark_10gene_network()
    print(f"Loaded 10-gene benchmark network. Directed edges = {(W_true != 0).sum()}", flush=True)

    # Factorial Grid Design: 15 Sample Sizes x 10 Measurement Intervals x 4 Trajectory Windows = 600 runs
    M_values = [20, 50, 80, 120, 160, 200, 300, 400, 500, 650, 800, 1000, 1300, 1600, 2000]
    dt_meas_values = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00]
    windows = [
        ("Early_Relaxation", (0.0, 4.0)),
        ("Mid_Transition", (2.0, 10.0)),
        ("Mixed_Trajectory", (0.0, 20.0)),
        ("Stationary_Attractor", (10.0, 25.0))
    ]

    out_csv = os.path.join(_proj_dir, "results_batch_runs.csv")
    fieldnames = [
        "run_id", "window_label", "t_start", "t_end", "M", "dt_meas", "T_total",
        "mean_var_x", "mean_var_v", "mean_drift_sq", "var_drift", "var_noise_rate", "empirical_snr",
        "AUROC", "AUPRC", "best_entropy",
        "F1_prob", "Precision_prob", "Recall_prob", "SHD_prob",
        "F1_best", "Precision_best", "Recall_best", "SHD_best",
        "elapsed_sec"
    ]

    total_runs = len(windows) * len(M_values) * len(dt_meas_values)
    run_idx = 0
    start_batch_t = time.time()

    with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for win_label, win_interval in windows:
            for M in M_values:
                # Pre-sample invariant pairs for this (M, window) across dt_meas
                for dt_m in dt_meas_values:
                    run_idx += 1
                    s, _ = simulate_grn_sde_pairs(
                        W_true, M=M, dt_sim=0.001, dt_sample=0.5, dt_meas=dt_m,
                        d=0.5, V=2.0, noise_std=0.15, seed=424, t_window=win_interval
                    )
                    
                    phys = compute_continuous_trajectory_physics(
                        s, W_true, dt_meas=dt_m, d=0.5, V=2.0, noise_std=0.15
                    )
                    
                    inf_res = run_mcmc_inference(
                        W_true, s, dt_meas=dt_m, n_sweeps=35, burn_in=10
                    )

                    row = {
                        "run_id": run_idx,
                        "window_label": win_label,
                        "t_start": win_interval[0],
                        "t_end": win_interval[1],
                        "M": M,
                        "dt_meas": dt_m,
                        "T_total": round(M * dt_m, 2),
                        "mean_var_x": phys["mean_var_x"],
                        "mean_var_v": phys["mean_var_v"],
                        "mean_drift_sq": phys["mean_drift_sq"],
                        "var_drift": phys["var_drift"],
                        "var_noise_rate": phys["var_noise_rate"],
                        "empirical_snr": phys["empirical_snr"],
                        "AUROC": inf_res["AUROC"],
                        "AUPRC": inf_res["AUPRC"],
                        "best_entropy": inf_res["best_entropy"],
                        "F1_prob": inf_res["F1_prob"],
                        "Precision_prob": inf_res["Precision_prob"],
                        "Recall_prob": inf_res["Recall_prob"],
                        "SHD_prob": inf_res["SHD_prob"],
                        "F1_best": inf_res["F1_best"],
                        "Precision_best": inf_res["Precision_best"],
                        "Recall_best": inf_res["Recall_best"],
                        "SHD_best": inf_res["SHD_best"],
                        "elapsed_sec": inf_res["elapsed_sec"]
                    }

                    writer.writerow(row)
                    f.flush()

                    if run_idx % 10 == 0 or run_idx == 1 or run_idx == total_runs:
                        pct = (run_idx / total_runs) * 100.0
                        el_tot = time.time() - start_batch_t
                        print(f"  [{run_idx:3d}/{total_runs} ({pct:5.1f}%)] "
                              f"Win={win_label:<18} | M={M:<4} | dt_meas={dt_m:<4.2f} | "
                              f"Var(x)={phys['mean_var_x']:<6.4f} | DriftSq={phys['mean_drift_sq']:<6.4f} | "
                              f"AUROC={inf_res['AUROC']:<6.4f} | F1={inf_res['F1_prob']:<6.4f} | "
                              f"SHD={inf_res['SHD_prob']:<2d} | ElapsedTotal={el_tot:.1f}s", flush=True)

    total_elapsed = time.time() - start_batch_t
    print(f"\nSUCCESS: Complete batch of {total_runs} runs saved to {out_csv} in {total_elapsed:.1f}s!", flush=True)

if __name__ == "__main__":
    main()
