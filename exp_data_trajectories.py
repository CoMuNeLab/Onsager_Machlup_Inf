#!/usr/bin/env python3
"""
Experimental Investigation of Input Data & Trajectory Properties on GRN Inference Accuracy
using graph-tool C++ Extension (`GeneExprBlockState`).

Experiments:
  1. Measurement Interval (`dt_meas`): Fixed M=200, dt_sim=0.001, dt_sample=0.5, t_window=(0.0, 10.0), dt_meas in [0.05, 0.1, 0.2, 0.5, 1.0]
  2. Trajectory Sample Size (`M`): Fixed dt_meas=0.2, dt_sim=0.001, dt_sample=0.5, t_window=(0.0, 10.0), M in [50, 100, 200, 500, 1000]
  3. Noise Intensity (`noise_std`): Fixed M=200, dt_meas=0.2, dt_sim=0.001, dt_sample=0.5, t_window=(0.0, 10.0), noise_std in [0.05, 0.15, 0.30, 0.50]
  4. Trajectory Regimes: Comparing Transient (Relaxation), Stationary (Equilibrium), and Mixed regimes across dt_meas and M.
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

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

# Plotting style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "figure.dpi": 300,
})

def to_serializable(d):
    """
    Recursively converts NumPy integer/float/ndarray types to Python native int/float/list
    so that JSON serialization never fails.
    """
    if isinstance(d, dict):
        return {k: to_serializable(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [to_serializable(x) for x in d]
    elif isinstance(d, (np.integer, np.int32, np.int64)):
        return int(d)
    elif isinstance(d, (np.floating, np.float32, np.float64)):
        return float(d)
    elif isinstance(d, np.ndarray):
        return d.tolist()
    return d

def get_N_M_and_arrays(s, dt_meas):
    """
    Helper to extract N, M, base states x_all, and finite-difference velocities vel
    whether s is a list of (N, 2) pair arrays or a contiguous (N, M) array.
    """
    if isinstance(s, (list, tuple)) and len(s) > 0 and isinstance(s[0], np.ndarray):
        N = s[0].shape[0]
        M = len(s)
        x_all = np.stack([p[:, 0] for p in s], axis=1) # shape (N, M)
        vel = np.stack([(p[:, 1] - p[:, 0]) / dt_meas for p in s], axis=1)
        return N, M, x_all, vel
    else:
        N, M = s.shape
        x_all = s
        if M > 1:
            vel = (s[:, 1:] - s[:, :-1]) / dt_meas
        else:
            vel = np.zeros_like(s)
        return N, M, x_all, vel

def compute_physical_diagnostics(s, W_true, dt_meas, d=0.5, V=2.0, noise_std=0.15):
    """
    Computes physical & dynamical diagnostics for sample set s:
      - mean_var_x: Mean expression variance across sampled base states
      - mean_var_v: Mean finite-difference velocity variance across sampled pairs
      - empirical_snr: Ratio of deterministic drift variance to Brownian noise variance
    """
    N, M, x_all, vel = get_N_M_and_arrays(s, dt_meas)
    mean_var_x = float(np.mean(np.var(x_all, axis=1)))
    mean_var_v = float(np.mean(np.var(vel, axis=1))) if M > 1 else 0.0

    # Compute deterministic drift at each measured base state
    # dx/dt = -d * x + V * sigmoid(W_true @ x)
    drifts = np.zeros_like(x_all)
    for t in range(M):
        x = x_all[:, t]
        prod = sigmoid(W_true @ x)
        drifts[:, t] = -d * x + V * prod

    var_drift = float(np.mean(np.var(drifts, axis=1)))
    var_noise_rate = float((noise_std ** 2) / dt_meas)
    empirical_snr = float(var_drift / var_noise_rate) if var_noise_rate > 0 else 0.0

    return {
        "mean_var_x": mean_var_x,
        "mean_var_v": mean_var_v,
        "var_drift": var_drift,
        "var_noise_rate": var_noise_rate,
        "empirical_snr": empirical_snr
    }

def run_single_experiment(W_true, s, dt_meas, noise_std=0.15, n_sweeps=30, burn_in=10, label=""):
    """
    Runs MCMC inference on sample set s (list of pairs or N x M array) with measurement step dt_meas.
    Computes AUROC, AUPRC, and reconstruction metrics (at P_edge >= 0.5 and best_graph).
    """
    N, M, x_all, _ = get_N_M_and_arrays(s, dt_meas)
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
            
        if sweep % 10 == 0 or sweep == 1:
            print(f"          [Sweep {sweep:2d}/{n_sweeps}] Entropy = {S_cur:8.2f} (dS = {dS:8.2f})", flush=True)

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

    diag = compute_physical_diagnostics(s, W_true, dt_meas, noise_std=noise_std)
    
    res = {
        "label": label,
        "N": int(N),
        "M": int(M),
        "dt_meas": float(dt_meas),
        "noise_std": float(noise_std),
        "n_sweeps": int(n_sweeps),
        "burn_in": int(burn_in),
        "elapsed_sec": round(float(elapsed), 2),
        "best_entropy": float(best_S),
        "AUROC": round(auroc_val, 4),
        "AUPRC": round(auprc_val, 4),
        "prob_metrics": prob_metrics,
        "best_metrics": best_metrics,
        "diagnostics": diag,
        "entropy_history": [float(x) for x in entropy_history]
    }
    return to_serializable(res)

def experiment_1_measurement_interval(W_true):
    """
    Experiment 1: Measurement Interval (`dt_meas`) under Invariant Pair Sampling
    Fixed M = 200, dt_sim = 0.001, dt_sample = 0.5, t_window = (0.0, 10.0), dt_meas in [0.05, 0.1, 0.2, 0.5, 1.0]
    """
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 1: Measurement Interval (dt_meas) [Invariant Base Distribution]", flush=True)
    print("=" * 70, flush=True)
    
    dt_meas_values = [0.05, 0.1, 0.2, 0.5, 1.0]
    results = []
    
    for dt_m in dt_meas_values:
        print(f"\n[Exp 1] Simulating invariant pairs: M=200, dt_sim=0.001, dt_meas={dt_m}...", flush=True)
        s, _ = simulate_grn_sde_pairs(
            W_true, M=200, dt_sim=0.001, dt_sample=0.5, dt_meas=dt_m,
            d=0.5, V=2.0, noise_std=0.15, seed=424, t_window=(0.0, 10.0)
        )
        print(f"        -> Sample pairs generated: {len(s)} pairs of shape {s[0].shape}", flush=True)
        res = run_single_experiment(
            W_true, s, dt_meas=dt_m, noise_std=0.15,
            n_sweeps=30, burn_in=10, label=f"dt_meas={dt_m}"
        )
        print(f"        -> AUROC: {res['AUROC']:.4f} | AUPRC: {res['AUPRC']:.4f} | F1 (prob): {res['prob_metrics']['F1-Score']:.4f} | SHD: {res['prob_metrics']['SHD']} | Elapsed: {res['elapsed_sec']}s", flush=True)
        results.append(res)
        
    return results

def experiment_2_trajectory_length(W_true):
    """
    Experiment 2: Trajectory Sample Size (M) under Invariant Multi-Trajectory Spawning
    Fixed dt_meas = 0.2, dt_sim = 0.001, dt_sample = 0.5, t_window = (0.0, 10.0), M in [50, 100, 200, 500, 1000]
    """
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 2: Sample Size (M) [Invariant Multi-Trajectory Spawning]", flush=True)
    print("=" * 70, flush=True)
    
    dt_m = 0.2
    M_values = [50, 100, 200, 500, 1000]
    results = []
    
    for M in M_values:
        print(f"\n[Exp 2] Simulating invariant pairs: M={M}, dt_meas={dt_m} across t_window=(0.0, 10.0)...", flush=True)
        s, _ = simulate_grn_sde_pairs(
            W_true, M=M, dt_sim=0.001, dt_sample=0.5, dt_meas=dt_m,
            d=0.5, V=2.0, noise_std=0.15, seed=424, t_window=(0.0, 10.0)
        )
        print(f"        -> Sample pairs generated: {len(s)} pairs of shape {s[0].shape}", flush=True)
        res = run_single_experiment(
            W_true, s, dt_meas=dt_m, noise_std=0.15,
            n_sweeps=30, burn_in=10, label=f"M={M}"
        )
        print(f"        -> AUROC: {res['AUROC']:.4f} | AUPRC: {res['AUPRC']:.4f} | F1 (prob): {res['prob_metrics']['F1-Score']:.4f} | SHD: {res['prob_metrics']['SHD']} | Elapsed: {res['elapsed_sec']}s", flush=True)
        results.append(res)
        
    return results

def experiment_3_noise_intensity(W_true):
    """
    Experiment 3: Noise Intensity (noise_std) under Invariant Pair Sampling
    Fixed M = 200, dt_meas = 0.2, dt_sim = 0.001, dt_sample = 0.5, t_window = (0.0, 10.0), noise_std in [0.05, 0.15, 0.30, 0.50]
    """
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 3: Noise Intensity (noise_std) [Invariant Pair Sampling]", flush=True)
    print("=" * 70, flush=True)
    
    noise_values = [0.05, 0.15, 0.30, 0.50]
    results = []
    
    for n_std in noise_values:
        print(f"\n[Exp 3] Simulating invariant pairs: M=200, dt_meas=0.2, noise_std={n_std}...", flush=True)
        s, _ = simulate_grn_sde_pairs(
            W_true, M=200, dt_sim=0.001, dt_sample=0.5, dt_meas=0.2,
            d=0.5, V=2.0, noise_std=n_std, seed=424, t_window=(0.0, 10.0)
        )
        print(f"        -> Sample pairs generated: {len(s)} pairs of shape {s[0].shape}", flush=True)
        res = run_single_experiment(
            W_true, s, dt_meas=0.2, noise_std=n_std,
            n_sweeps=30, burn_in=10, label=f"noise_std={n_std}"
        )
        diag = res["diagnostics"]
        print(f"        -> AUROC: {res['AUROC']:.4f} | AUPRC: {res['AUPRC']:.4f} | F1 (prob): {res['prob_metrics']['F1-Score']:.4f} | Elapsed: {res['elapsed_sec']}s", flush=True)
        print(f"        -> MeanVar(x): {diag['mean_var_x']:.4f} | Empirical SNR: {diag['empirical_snr']:.4f}", flush=True)
        results.append(res)
        
    return results

def experiment_4_trajectory_regimes(W_true):
    """
    Experiment 4: Trajectory Regime Analysis (Transient vs. Stationary vs. Mixed)
    Compares GRN inference across three dynamical regimes:
      - 'Transient (Relaxation)': t_window = (0.0, 4.0)
      - 'Stationary (Equilibrium)': t_window = (10.0, 20.0)
      - 'Mixed': t_window = (0.0, 20.0)
    We test each regime across:
      (A) Measurement Intervals dt_meas in [0.05, 0.1, 0.2, 0.5, 1.0] (at fixed M=200)
      (B) Sample Sizes M in [50, 100, 200, 500] (at fixed dt_meas=0.2)
    """
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 4: Trajectory Regime Analysis (Transient vs. Stationary vs. Mixed)", flush=True)
    print("=" * 70, flush=True)

    regimes = {
        "Transient (Relaxation)": (0.0, 4.0),
        "Stationary (Equilibrium)": (10.0, 20.0),
        "Mixed": (0.0, 20.0)
    }

    results = {
        "by_dt_meas": {},
        "by_M": {}
    }

    # Part A: Scaling vs. dt_meas at fixed M = 200
    dt_meas_values = [0.05, 0.1, 0.2, 0.5, 1.0]
    for reg_name, window in regimes.items():
        print(f"\n--- [Exp 4A: {reg_name}] Scaling vs. dt_meas (M=200, window={window}) ---", flush=True)
        reg_res_dt = []
        for dt_m in dt_meas_values:
            s, _ = simulate_grn_sde_pairs(
                W_true, M=200, dt_sim=0.001, dt_sample=0.5, dt_meas=dt_m,
                d=0.5, V=2.0, noise_std=0.15, seed=424, t_window=window
            )
            res = run_single_experiment(
                W_true, s, dt_meas=dt_m, noise_std=0.15,
                n_sweeps=25, burn_in=8, label=f"{reg_name}_dt{dt_m}"
            )
            print(f"  [{reg_name} | dt_meas={dt_m}] AUROC={res['AUROC']:.4f}, AUPRC={res['AUPRC']:.4f}, F1={res['prob_metrics']['F1-Score']:.4f}, Var(x)={res['diagnostics']['mean_var_x']:.4f}", flush=True)
            reg_res_dt.append(res)
        results["by_dt_meas"][reg_name] = reg_res_dt

    # Part B: Scaling vs. M at fixed dt_meas = 0.2
    M_values = [50, 100, 200, 500]
    for reg_name, window in regimes.items():
        print(f"\n--- [Exp 4B: {reg_name}] Scaling vs. M (dt_meas=0.2, window={window}) ---", flush=True)
        reg_res_M = []
        for M in M_values:
            s, _ = simulate_grn_sde_pairs(
                W_true, M=M, dt_sim=0.001, dt_sample=0.5, dt_meas=0.2,
                d=0.5, V=2.0, noise_std=0.15, seed=424, t_window=window
            )
            res = run_single_experiment(
                W_true, s, dt_meas=0.2, noise_std=0.15,
                n_sweeps=25, burn_in=8, label=f"{reg_name}_M{M}"
            )
            print(f"  [{reg_name} | M={M}] AUROC={res['AUROC']:.4f}, AUPRC={res['AUPRC']:.4f}, F1={res['prob_metrics']['F1-Score']:.4f}, SHD={res['prob_metrics']['SHD']}", flush=True)
            reg_res_M.append(res)
        results["by_M"][reg_name] = reg_res_M

    return results

def generate_plots(res1, res2, res3, res4, out_dir):
    """
    Generates 4 publication figures in out_dir:
      - fig_exp1_dt_meas.png
      - fig_exp2_trajectory_length.png
      - fig_exp3_noise_intensity.png
      - fig_exp4_trajectory_regimes.png
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # --- Figure 1: Measurement Interval ---
    dt_vals = [r["dt_meas"] for r in res1]
    auroc1 = [r["AUROC"] for r in res1]
    auprc1 = [r["AUPRC"] for r in res1]
    f1_prob1 = [r["prob_metrics"]["F1-Score"] for r in res1]
    shd1 = [r["prob_metrics"]["SHD"] for r in res1]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(dt_vals, auroc1, "o-", color="#d95f02", linewidth=2.5, markersize=7, label="AUROC")
    ax1.plot(dt_vals, auprc1, "s-", color="#1b9e77", linewidth=2.5, markersize=7, label="AUPRC")
    ax1.plot(dt_vals, f1_prob1, "^-", color="#2b5c8f", linewidth=2.5, markersize=7, label="F1-Score ($P_{\\text{edge}} \\geq 0.5$)")
    ax1.set_xlabel("Measurement Interval ($dt_{\\text{meas}}$)")
    ax1.set_ylabel("Inference Metric Score")
    ax1.set_title("Exp 1: GRN Inference vs. Measurement Interval (Invariant Pairs)")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="lower left")
    
    ax2 = ax1.twinx()
    ax2.plot(dt_vals, shd1, "d--", color="#e7298a", linewidth=2.0, markersize=6, label="SHD (Right Axis)")
    ax2.set_ylabel("Structural Hamming Distance (SHD)", color="#e7298a")
    ax2.tick_params(axis='y', labelcolor="#e7298a")
    ax2.grid(False)
    
    plt.tight_layout()
    path1 = os.path.join(out_dir, "fig_exp1_dt_meas.png")
    plt.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved Figure 1: {path1}", flush=True)

    # --- Figure 2: Trajectory Sample Size ---
    M_vals = [r["M"] for r in res2]
    auroc2 = [r["AUROC"] for r in res2]
    auprc2 = [r["AUPRC"] for r in res2]
    f1_prob2 = [r["prob_metrics"]["F1-Score"] for r in res2]
    shd2 = [r["prob_metrics"]["SHD"] for r in res2]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(M_vals, auroc2, "o-", color="#d95f02", linewidth=2.5, markersize=7, label="AUROC")
    ax1.plot(M_vals, auprc2, "s-", color="#1b9e77", linewidth=2.5, markersize=7, label="AUPRC")
    ax1.plot(M_vals, f1_prob2, "^-", color="#2b5c8f", linewidth=2.5, markersize=7, label="F1-Score ($P_{\\text{edge}} \\geq 0.5$)")
    ax1.set_xlabel("Sample Size ($M$, with $dt_{\\text{meas}}=0.2$)")
    ax1.set_ylabel("Inference Metric Score")
    ax1.set_title("Exp 2: Sample Complexity under Invariant Multi-Trajectory Spawning")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="lower right")

    ax2 = ax1.twinx()
    ax2.plot(M_vals, shd2, "d--", color="#e7298a", linewidth=2.0, markersize=6, label="SHD (Right Axis)")
    ax2.set_ylabel("Structural Hamming Distance (SHD)", color="#e7298a")
    ax2.tick_params(axis='y', labelcolor="#e7298a")
    ax2.grid(False)

    plt.tight_layout()
    path2 = os.path.join(out_dir, "fig_exp2_trajectory_length.png")
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved Figure 2: {path2}", flush=True)

    # --- Figure 3: Noise Intensity ---
    n_vals = [r["noise_std"] for r in res3]
    auroc3 = [r["AUROC"] for r in res3]
    auprc3 = [r["AUPRC"] for r in res3]
    f1_prob3 = [r["prob_metrics"]["F1-Score"] for r in res3]
    mean_var3 = [r["diagnostics"]["mean_var_x"] for r in res3]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(n_vals, auroc3, "o-", color="#d95f02", linewidth=2.5, markersize=7, label="AUROC")
    ax1.plot(n_vals, auprc3, "s-", color="#1b9e77", linewidth=2.5, markersize=7, label="AUPRC")
    ax1.plot(n_vals, f1_prob3, "^-", color="#2b5c8f", linewidth=2.5, markersize=7, label="F1-Score ($P_{\\text{edge}} \\geq 0.5$)")
    ax1.set_xlabel("Stochastic Noise Intensity ($\\text{noise\\_std}$)")
    ax1.set_ylabel("Inference Metric Score")
    ax1.set_title("Exp 3: Physical Tradeoff in Noise Intensity (Invariant Pairs)")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="lower left")

    ax2 = ax1.twinx()
    ax2.plot(n_vals, mean_var3, "v--", color="#7570b3", linewidth=2.0, markersize=6, label="Mean Expression Variance $\\text{Var}(x)$")
    ax2.set_ylabel("Mean Gene Expression Variance $\\text{Var}(x)$", color="#7570b3")
    ax2.tick_params(axis='y', labelcolor="#7570b3")
    ax2.grid(False)

    plt.tight_layout()
    path3 = os.path.join(out_dir, "fig_exp3_noise_intensity.png")
    plt.savefig(path3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved Figure 3: {path3}", flush=True)

    # --- Figure 4: Trajectory Regime Comparison ---
    colors_reg = {
        "Transient (Relaxation)": "#e7298a",
        "Stationary (Equilibrium)": "#1b9e77",
        "Mixed": "#377eb8"
    }

    fig, (ax_dt, ax_M) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: AUROC vs. dt_meas across Regimes
    for reg_name, reg_list in res4["by_dt_meas"].items():
        dt_x = [r["dt_meas"] for r in reg_list]
        auc_y = [r["AUROC"] for r in reg_list]
        ax_dt.plot(dt_x, auc_y, "o-", color=colors_reg[reg_name], linewidth=2.5, markersize=7, label=reg_name)
    ax_dt.set_xlabel("Measurement Interval ($dt_{\\text{meas}}$)")
    ax_dt.set_ylabel("AUROC")
    ax_dt.set_title("(A) AUROC vs. Measurement Interval ($dt_{\\text{meas}}$ at $M=200$)")
    ax_dt.set_ylim(0.4, 1.05)
    ax_dt.grid(True, linestyle=":", alpha=0.7)
    ax_dt.legend(loc="lower left")

    # Panel B: AUROC vs. M across Regimes
    for reg_name, reg_list in res4["by_M"].items():
        M_x = [r["M"] for r in reg_list]
        auc_y = [r["AUROC"] for r in reg_list]
        ax_M.plot(M_x, auc_y, "s-", color=colors_reg[reg_name], linewidth=2.5, markersize=7, label=reg_name)
    ax_M.set_xlabel("Sample Size ($M$ at $dt_{\\text{meas}}=0.2$)")
    ax_M.set_ylabel("AUROC")
    ax_M.set_title("(B) AUROC vs. Sample Size ($M$ across Regimes)")
    ax_M.set_ylim(0.4, 1.05)
    ax_M.grid(True, linestyle=":", alpha=0.7)
    ax_M.legend(loc="lower right")

    plt.tight_layout()
    path4 = os.path.join(out_dir, "fig_exp4_trajectory_regimes.png")
    plt.savefig(path4, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved Figure 4: {path4}", flush=True)

def print_summary_tables(res1, res2, res3, res4):
    print("\n" + "=" * 80, flush=True)
    print("SUMMARY TABLE 1: MEASUREMENT INTERVAL (dt_meas) [INVARIANT PAIRS]", flush=True)
    print("=" * 80, flush=True)
    print(f"{'dt_meas':<8} | {'M':<6} | {'AUROC':<7} | {'AUPRC':<7} | {'F1 (prob)':<9} | {'SHD':<5} | {'MeanVar(x)':<10} | {'MeanVar(v)':<10}", flush=True)
    print("-" * 80, flush=True)
    for r in res1:
        d = r["diagnostics"]
        print(f"{r['dt_meas']:<8} | {r['M']:<6} | {r['AUROC']:<7.4f} | {r['AUPRC']:<7.4f} | {r['prob_metrics']['F1-Score']:<9.4f} | {r['prob_metrics']['SHD']:<5} | {d['mean_var_x']:<10.4f} | {d['mean_var_v']:<10.4f}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("SUMMARY TABLE 2: SAMPLE SIZE (M) [INVARIANT MULTI-TRAJECTORY SPAWNING]", flush=True)
    print("=" * 80, flush=True)
    print(f"{'M':<6} | {'dt_meas':<8} | {'AUROC':<7} | {'AUPRC':<7} | {'F1 (prob)':<9} | {'SHD':<5} | {'Precision':<9} | {'Recall':<8}", flush=True)
    print("-" * 80, flush=True)
    for r in res2:
        pm = r["prob_metrics"]
        print(f"{r['M']:<6} | {r['dt_meas']:<8} | {r['AUROC']:<7.4f} | {r['AUPRC']:<7.4f} | {pm['F1-Score']:<9.4f} | {pm['SHD']:<5} | {pm['Precision']:<9.4f} | {pm['TPR (Recall)']:<8.4f}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("SUMMARY TABLE 3: NOISE INTENSITY (noise_std) [INVARIANT PAIRS]", flush=True)
    print("=" * 80, flush=True)
    print(f"{'noise_std':<10} | {'AUROC':<7} | {'AUPRC':<7} | {'F1 (prob)':<9} | {'SHD':<5} | {'MeanVar(x)':<10} | {'Emp SNR':<8}", flush=True)
    print("-" * 80, flush=True)
    for r in res3:
        d = r["diagnostics"]
        print(f"{r['noise_std']:<10} | {r['AUROC']:<7.4f} | {r['AUPRC']:<7.4f} | {r['prob_metrics']['F1-Score']:<9.4f} | {r['prob_metrics']['SHD']:<5} | {d['mean_var_x']:<10.4f} | {d['empirical_snr']:<8.4f}", flush=True)
    print("=" * 80, flush=True)

    print("\n" + "=" * 80, flush=True)
    print("SUMMARY TABLE 4: TRAJECTORY REGIMES (TRANSIENT vs. STATIONARY vs. MIXED)", flush=True)
    print("=" * 80, flush=True)
    print("--- (A) AUROC & SHD vs. Measurement Interval dt_meas (M=200) ---", flush=True)
    print(f"{'Regime':<25} | {'dt_meas':<8} | {'AUROC':<7} | {'AUPRC':<7} | {'F1 (prob)':<9} | {'SHD':<5} | {'Var(x)':<8}", flush=True)
    print("-" * 80, flush=True)
    for reg_name, reg_list in res4["by_dt_meas"].items():
        for r in reg_list:
            d = r["diagnostics"]
            print(f"{reg_name:<25} | {r['dt_meas']:<8} | {r['AUROC']:<7.4f} | {r['AUPRC']:<7.4f} | {r['prob_metrics']['F1-Score']:<9.4f} | {r['prob_metrics']['SHD']:<5} | {d['mean_var_x']:<8.4f}", flush=True)

    print("\n--- (B) AUROC & SHD vs. Sample Size M (dt_meas=0.2) ---", flush=True)
    print(f"{'Regime':<25} | {'M':<6} | {'AUROC':<7} | {'AUPRC':<7} | {'F1 (prob)':<9} | {'SHD':<5} | {'Precision':<9} | {'Recall':<8}", flush=True)
    print("-" * 80, flush=True)
    for reg_name, reg_list in res4["by_M"].items():
        for r in reg_list:
            pm = r["prob_metrics"]
            print(f"{reg_name:<25} | {r['M']:<6} | {r['AUROC']:<7.4f} | {r['AUPRC']:<7.4f} | {pm['F1-Score']:<9.4f} | {pm['SHD']:<5} | {pm['Precision']:<9.4f} | {pm['TPR (Recall)']:<8.4f}", flush=True)
    print("=" * 80, flush=True)

def main():
    print("=" * 80, flush=True)
    print("GRN INFERENCE DATA & TRAJECTORY EXPERIMENTER PIPELINE (INVARIANT PAIRS)", flush=True)
    print("=" * 80, flush=True)

    out_json = os.path.join(_proj_dir, "results_data_trajectories.json")
    out_dir = os.path.join(_proj_dir, "figures")

    if "--plot-only" in sys.argv and os.path.exists(out_json):
        print(f"Loading existing experimental results from: {out_json}", flush=True)
        with open(out_json, "r") as f:
            all_results = json.load(f)
        res_exp1 = all_results["exp1_dt_meas"]
        res_exp2 = all_results["exp2_sample_size"]
        res_exp3 = all_results["exp3_noise_std"]
        res_exp4 = all_results["exp4_regimes"]
        print_summary_tables(res_exp1, res_exp2, res_exp3, res_exp4)
        generate_plots(res_exp1, res_exp2, res_exp3, res_exp4, out_dir)
        print(f"All figures saved to directory: {out_dir}", flush=True)
        print("\nSUCCESS: Plots and tables regenerated cleanly!", flush=True)
        return

    W_true = get_benchmark_10gene_network()
    print(f"Loaded benchmark network: N={W_true.shape[0]}, directed edges={(W_true != 0).sum()}", flush=True)
    all_results = {}

    # Run Experiments with incremental JSON saving
    res_exp1 = experiment_1_measurement_interval(W_true)
    all_results["exp1_dt_meas"] = res_exp1
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    res_exp2 = experiment_2_trajectory_length(W_true)
    all_results["exp2_sample_size"] = res_exp2
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    res_exp3 = experiment_3_noise_intensity(W_true)
    all_results["exp3_noise_std"] = res_exp3
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    res_exp4 = experiment_4_trajectory_regimes(W_true)
    all_results["exp4_regimes"] = res_exp4
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print Summary Tables
    print_summary_tables(res_exp1, res_exp2, res_exp3, res_exp4)
    print(f"\nSaved complete experimental results to: {out_json}", flush=True)

    # Generate Figures
    generate_plots(res_exp1, res_exp2, res_exp3, res_exp4, out_dir)
    print(f"All figures saved to directory: {out_dir}", flush=True)

    print("\nSUCCESS: All data & trajectory experiments completed!", flush=True)

if __name__ == "__main__":
    main()
